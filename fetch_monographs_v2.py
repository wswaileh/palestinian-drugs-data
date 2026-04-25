"""
fetch_monographs_v2.py
────────────────────────────────────────────────────────────────
Improvements over v1:
  1. Ingredient deduplication  →  ~1,292 fetches instead of 2,752
  2. Three OpenFDA strategies per ingredient (with fallback):
       a) openfda.substance_name:"name"   (broader than generic_name)
       b) openfda.generic_name:"name"     (v1 approach)
       c) active_ingredient:"name"        (free-text, highest recall)
  3. DailyMed API as a second source when OpenFDA misses
  4. Multi-ingredient drug decomposition
  5. Longer text capture (2000 chars) for RAG quality
  6. Resume from checkpoint (safe to Ctrl-C and restart)

Requirements:
    pip install aiohttp pandas tqdm

Usage:
    python fetch_monographs_v2.py
    python fetch_monographs_v2.py --input pharmacy_all_drugs.csv \\
                                  --output drug_monographs_v2.csv \\
                                  --concurrency 15
"""

import argparse
import asyncio
import re
import json
import os
import pandas as pd
import aiohttp
from tqdm.asyncio import tqdm as atqdm

OPENFDA_URL  = "https://api.fda.gov/drug/label.json"
DAILYMED_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
CHECKPOINT   = "fetch_checkpoint_v2.json"

TEXT_LIMIT = 2000   # chars per field (was 800 in v1)

OPENFDA_FIELDS = {
    "indications":       "indications_and_usage",
    "contraindications": "contraindications",
    "mechanism":         "clinical_pharmacology",
    "drug_interactions": "drug_interactions",
    "side_effects":      "adverse_reactions",
    "warnings":          "warnings_and_precautions",
    "warnings_alt":      "warnings",   # some labels use this key instead
}

# ── Name cleaning ──────────────────────────────────────────────────────────────

_SALT_FORMS = (
    r"hcl|hydrochloride|sodium|potassium|calcium|magnesium|acetate|phosphate|"
    r"sulfate|sulphate|fumarate|maleate|tartrate|citrate|trihydrate|monohydrate|"
    r"dihydrate|hemihydrate|besylate|mesylate|tosylate|diethylamine|valerate|"
    r"gluconate|lactate|bromide|chloride|iodide|nitrate|oxalate|succinate|"
    r"stearate|palmitate|propionate|butyrate|caproate|decanoate|enanthate|"
    r"undecanoate|cypionate|phenylpropionate|isocaproate|benzoate|hyclate|"
    r"dihydrochloride|disodium|hibenzate|pamoate|napadisylate|xinafoate|"
    r"as calcium|as sodium|as benzoate|as hcl|as hydrochloride|as base|"
    r"as potassium|as phosphate|as sulfate|as sulphate|as acetate|as citrate"
)

# Dosage form / route words that often follow the ingredient and break API search
_FORM_WORDS = (
    r"tab|tabs|tablet|tablets|cap|caps|capsule|capsules|syrup|syr|"
    r"inj|injection|injections|amp|amps|ampoule|ampoules|sol|solution|"
    r"susp|suspension|drops|drop|spray|sprays|oint|ointment|gel|cream|"
    r"lotion|patch|patches|powder|granules|sachet|sachets|effervescent|"
    r"oral|topical|inhaler|inhalation|nasal|ophthalmic|otic|"
    r"ovule|ovules|pessary|pessaries|suppository|suppositories|vial|vials"
)

# Pharmaceutical modifiers that aren't part of the ingredient name on OpenFDA
_MODIFIERS = (
    r"micronized|extended release|sustained release|immediate release|"
    r"modified release|controlled release|prolonged release|delayed release|"
    r"slow release|extended-release|sustained-release|immediate-release|"
    r"er|sr|xr|cr|mr|recombinant|conjugated|pegylated|liposomal|coated|"
    r"film coated|enteric coated|chewable|dispersible|orodispersible|"
    r"anhydrous|hemihydrate|sesquihydrate"
)

# Common British / WHO INN / typo variants → OpenFDA-friendly USAN names.
# Applied as a final normalization step on the cleaned ingredient.
_INGREDIENT_ALIASES = {
    "mesalazine":      "mesalamine",
    "aciclovir":       "acyclovir",
    "gentamycin":      "gentamicin",
    "ibuprufen":       "ibuprofen",
    "amoxycillin":     "amoxicillin",
    "glimepride":      "glimepiride",
    "ezomeprazole":    "esomeprazole",
    "formetrol":       "formoterol",
    "dypirone":        "metamizole",
    "erythropiotin":   "erythropoietin",
    "isotretinion":    "isotretinoin",
    "ethinylestradiol": "ethinyl estradiol",
    "naproxane":       "naproxen",
    "guaifenasine":    "guaifenesin",
    "orphenadrin":     "orphenadrine",
    "dicyclomide":     "dicyclomine",
    "hyoscine-n-butyl": "scopolamine butylbromide",
    "hyoscine butylbromide": "scopolamine butylbromide",
    "hyoscine butlbromide":  "scopolamine butylbromide",
    "valaciclovir":    "valacyclovir",
    "paracetamol":     "acetaminophen",
    "sulphamethoxasole": "sulfamethoxazole",
    "metoclopromide":  "metoclopramide",
    "isotretinon":     "isotretinoin",
    "psuedoephedrin":  "pseudoephedrine",
    "dextroamphetamins": "dextroamphetamine",
    "ciclosporin":     "cyclosporine",
    "salmeterol xinofoate": "salmeterol",
    "salmeterol xinafoate": "salmeterol",
}


def clean_ingredient(raw: str) -> str:
    """Return a clean active-ingredient name suitable for API search."""
    name = str(raw).strip()
    # Take first component only (before comma, +, &, or 'and')
    name = re.split(r"\s*[,+&]\s*|\s+and\s+", name, maxsplit=1)[0]
    # Strip "each <unit> contains"
    name = re.sub(r"each\s+\w+\s+contains\s+", "", name, flags=re.IGNORECASE)
    # Strip parenthetical content
    name = re.sub(r"\(.*?\)", "", name)
    # Strip dosage patterns (numbers + units) — handle units jammed to digits and %
    # First, insert a space between letter and digit (e.g. "glucagon1" -> "glucagon 1")
    name = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", name)
    name = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", name)
    # Strip dose + word-units (mg, ml, etc.)
    name = re.sub(
        r"\b\d+\.?\d*\s*(?:mg|mcg|ml|g|iu|ug|mmol|meq|unit|units|million|thousand)\b",
        "", name, flags=re.IGNORECASE,
    )
    # Strip dose + percent (no trailing \b since % is non-word)
    name = re.sub(r"\b\d+\.?\d*\s*%", "", name)
    # Strip orphan numbers left behind (e.g. "0.5", "100")
    name = re.sub(r"\b\d+\.?\d*\b", "", name)
    # Strip orphan unit words left behind (e.g. "ml" with no leading digit)
    name = re.sub(r"\b(?:mg|mcg|ml|iu|ug|mmol|meq)\b", "", name, flags=re.IGNORECASE)
    # Strip salt forms (incl. broader "as <salt>" patterns)
    name = re.sub(rf"\b({_SALT_FORMS})\b", "", name, flags=re.IGNORECASE)
    # Strip dosage form / route words (with optional leading slash, e.g. "/tab")
    name = re.sub(rf"\s*/?\s*\b({_FORM_WORDS})\b\.?", " ", name, flags=re.IGNORECASE)
    # Strip pharmaceutical modifiers
    name = re.sub(rf"\b({_MODIFIERS})\b", "", name, flags=re.IGNORECASE)
    # Strip "after reconstitution" / "for injection" / "for solution" suffixes
    name = re.sub(r"\bafter\s+reconstitution\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\bfor\s+(injection|solution|infusion|inhalation)\b", "", name, flags=re.IGNORECASE)
    # Strip stray slashes left behind
    name = re.sub(r"\s*/\s*", " ", name)
    # Strip dangling prepositions left behind by partial salt-form removal
    # (e.g. "tacrolimus as" -> "tacrolimus", "iopromide equiv. to iodine" -> "iopromide")
    name = re.sub(r"\b(as|as the|equiv\.?\s*to|equivalent\s+to)\s.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b(as|as the)\s*$", "", name, flags=re.IGNORECASE)
    # Clean up whitespace + trailing punctuation
    name = re.sub(r"\s+", " ", name).strip().strip(",./- ").lower()
    # Apply alias map (British/typo → OpenFDA-friendly USAN name)
    name = _INGREDIENT_ALIASES.get(name, name)
    return name


def split_ingredients(raw: str) -> list[str]:
    """For multi-ingredient drugs, return all cleaned components."""
    parts = re.split(r"\s*[,+&]\s*|\s+and\s+", str(raw))
    results = []
    for p in parts:
        c = clean_ingredient(p)
        if len(c) >= 4 and re.search(r"[a-zA-Z]{3,}", c):
            results.append(c)
    return results


# ── OpenFDA fetch ─────────────────────────────────────────────────────────────

def _extract_openfda(label: dict) -> dict:
    """Pull fields from a single OpenFDA label result."""
    out = {}
    for field, fda_key in OPENFDA_FIELDS.items():
        val = (label.get(fda_key) or [""])[0][:TEXT_LIMIT].strip()
        out[field] = val
    # Merge warnings / warnings_alt
    if not out["warnings"] and out.get("warnings_alt"):
        out["warnings"] = out["warnings_alt"]
    out.pop("warnings_alt", None)
    return out


def _best_label(results: list[dict]) -> dict:
    """Pick the label with the most complete fields."""
    def score(r):
        return sum(1 for k in OPENFDA_FIELDS if (r.get(k) or [""])[0].strip())
    return max(results, key=score)


async def _openfda_search(session: aiohttp.ClientSession,
                          name: str,
                          semaphore: asyncio.Semaphore) -> dict | None:
    """Try three OpenFDA strategies and return extracted fields or None."""
    strategies = [
        ("substance_name",  f'openfda.substance_name:"{name}"'),
        ("generic_name",    f'openfda.generic_name:"{name}"'),
        ("active_ingred",   f'active_ingredient:"{name}"'),
    ]
    async with semaphore:
        for _label, query in strategies:
            for attempt in range(3):
                try:
                    async with session.get(
                        OPENFDA_URL,
                        params={"search": query, "limit": 3},
                        timeout=aiohttp.ClientTimeout(total=20),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            hits = data.get("results", [])
                            if hits:
                                return _extract_openfda(_best_label(hits))
                            break  # 200 but 0 results → try next strategy
                        elif resp.status == 404:
                            break  # no results → try next strategy
                        elif resp.status == 429:
                            await asyncio.sleep(2 ** attempt + 1)
                        else:
                            await asyncio.sleep(1)
                            break
                except asyncio.TimeoutError:
                    await asyncio.sleep(1)
                except Exception:
                    await asyncio.sleep(0.5)
                    break
    return None


# ── DailyMed fetch ────────────────────────────────────────────────────────────

async def _dailymed_search(session: aiohttp.ClientSession,
                           name: str,
                           semaphore: asyncio.Semaphore) -> dict | None:
    """Query DailyMed and return basic fields (indications at minimum)."""
    async with semaphore:
        for attempt in range(2):
            try:
                async with session.get(
                    DAILYMED_URL,
                    params={"drug_name": name, "pagesize": 3},
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("data", [])
                        if items:
                            # DailyMed SPL list — return minimal signal
                            return {"_dailymed_hit": True, "indications": f"Found in DailyMed: {items[0].get('title','')[:200]}"}
                    elif resp.status in (404, 400):
                        break
                    elif resp.status == 429:
                        await asyncio.sleep(2 ** attempt + 1)
                    else:
                        break
            except Exception:
                await asyncio.sleep(0.5)
                break
    return None


# ── Per-ingredient fetch  ─────────────────────────────────────────────────────

async def fetch_ingredient(session: aiohttp.ClientSession,
                           ingredient: str,
                           semaphore: asyncio.Semaphore) -> dict:
    """Fetch monograph for one unique ingredient, trying all sources."""
    result = await _openfda_search(session, ingredient, semaphore)
    if result:
        return {"ingredient": ingredient, "source": "openfda", "found": True, **result}

    dm = await _dailymed_search(session, ingredient, semaphore)
    if dm:
        empty = {f: "" for f in ["indications","contraindications","mechanism",
                                  "drug_interactions","side_effects","warnings"]}
        empty.update(dm)
        return {"ingredient": ingredient, "source": "dailymed", "found": True, **empty}

    return {
        "ingredient": ingredient, "source": "none", "found": False,
        "indications": "", "contraindications": "", "mechanism": "",
        "drug_interactions": "", "side_effects": "", "warnings": "",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(args):
    df = pd.read_csv(args.input, encoding="utf-8-sig")

    # Build ingredient → [drug_ids] mapping
    ingredient_map: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        for ing in split_ingredients(row["scientific_name"]):
            if ing not in ingredient_map:
                ingredient_map[ing] = []
            ingredient_map[ing].append(row["drug_id"])

    # Filter trivial entries
    ingredient_map = {
        k: v for k, v in ingredient_map.items()
        if len(k) >= 4 and re.search(r"[a-zA-Z]{4,}", k)
    }

    print(f"Drugs in catalog  : {len(df)}")
    print(f"Unique ingredients: {len(ingredient_map)}")

    # Resume from checkpoint
    done: dict[str, dict] = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            done = json.load(f)
        print(f"Resuming — {len(done)} already fetched, "
              f"{len(ingredient_map) - len(done)} remaining.")

    todo = [ing for ing in ingredient_map if ing not in done]

    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 5)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_ingredient(session, ing, semaphore) for ing in todo]
        batch: list[dict] = []
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching"):
            result = await coro
            done[result["ingredient"]] = result
            batch.append(result)
            if len(batch) % 50 == 0:
                with open(CHECKPOINT, "w") as f:
                    json.dump(done, f)

    with open(CHECKPOINT, "w") as f:
        json.dump(done, f)

    # ── Expand: one row per drug in the original catalog ──────────────────
    rows = []
    for _, row in df.iterrows():
        ings = split_ingredients(row["scientific_name"])
        # Use first ingredient's monograph (primary active ingredient)
        primary = next((done[i] for i in ings if i in done and done[i]["found"]), None)
        if primary is None:
            primary = next((done[i] for i in ings if i in done), None)

        base = {
            "drug_id":        row["drug_id"],
            "trade_name":     row["trade_name"],
            "scientific_name":row["scientific_name"],
            "available":      row["available"],
            "price":          row["price"],
            "supplier":       row["supplier"],
        }
        if primary:
            base.update({
                "found":           primary["found"],
                "source":          primary["source"],
                "matched_ingredient": primary["ingredient"],
                "indications":     primary.get("indications",""),
                "contraindications":primary.get("contraindications",""),
                "mechanism":       primary.get("mechanism",""),
                "drug_interactions":primary.get("drug_interactions",""),
                "side_effects":    primary.get("side_effects",""),
                "warnings":        primary.get("warnings",""),
            })
        else:
            base.update({
                "found": False, "source": "none", "matched_ingredient": "",
                "indications": "", "contraindications": "", "mechanism": "",
                "drug_interactions": "", "side_effects": "", "warnings": "",
            })
        rows.append(base)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)

    found = out["found"].sum()
    total = len(out)
    openfda_n = (out["source"] == "openfda").sum()
    dailymed_n = (out["source"] == "dailymed").sum()
    fields = ["indications","contraindications","mechanism","drug_interactions","side_effects","warnings"]
    found_rows = out[out["found"]]

    print(f"\n{'─'*50}")
    print(f"Result  : {found}/{total} drugs with monograph data ({100*found//total}%)")
    print(f"  OpenFDA  : {openfda_n}")
    print(f"  DailyMed : {dailymed_n}")
    print(f"\nField completeness (among found):")
    for f in fields:
        n = (found_rows[f].str.len() > 10).sum()
        print(f"  {f:<20}: {n}/{len(found_rows)} ({100*n//max(len(found_rows),1)}%)")
    print(f"\nOutput     : {args.output}")
    print(f"Checkpoint : {CHECKPOINT} (safe to delete after reviewing output)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default="pharmacy_all_drugs.csv")
    parser.add_argument("--output",      default="drug_monographs_v2.csv")
    parser.add_argument("--concurrency", type=int, default=15,
                        help="Concurrent API requests (default 15; keep ≤20 to avoid 429s)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
