"""
build_corpus.py
────────────────────────────────────────────────────────────────
Convert the v2 ingredient checkpoint + pharmacy catalog into the
canonical RAG corpus layout consumed by the SciBERT pipeline:

  corpus/
    ingredients.parquet           one row per unique active ingredient
    catalog.parquet               one row per catalog drug (cleaned)
    catalog_ingredient_map.parquet  drug_id ↔ ingredient (many-to-many)
    chunks.jsonl                  section-level chunks for embedding
    coverage_report.md            human-readable stats + gap list

Granularity choice: the vector store is keyed by *active ingredient*,
not trade name. Embedding 50+ paracetamol trade names would dilute the
index and bias retrieval — the catalog availability join happens at
query time via catalog_ingredient_map.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from fetch_monographs_v2 import split_ingredients

SECTIONS = [
    "indications",
    "contraindications",
    "mechanism",
    "drug_interactions",
    "side_effects",
    "warnings",
]

# SciBERT max position is 512 wordpieces. We chunk by word count with a
# conservative ~3 wordpieces-per-word ratio for biomedical text, hence
# ~150 words per chunk leaves headroom for [CLS]/[SEP] + tokenizer slack.
CHUNK_WORDS = 150
CHUNK_OVERLAP = 30


def chunk_text(text: str, max_words: int = CHUNK_WORDS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-window chunks. Returns [] for empty input."""
    text = (text or "").strip()
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    step = max_words - overlap
    for start in range(0, len(words), step):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
    return chunks


def _normalize_field(s) -> str:
    """OpenFDA returns multi-paragraph blobs; collapse whitespace."""
    if pd.isna(s) or not s:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build(catalog_path: Path, checkpoint_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ────────────────────────────────────────────────────
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    with open(checkpoint_path) as f:
        cache: dict[str, dict] = json.load(f)

    # ── Build catalog ↔ ingredient mapping ─────────────────────────────
    map_rows = []
    drug_ings: dict[int, list[str]] = {}
    for _, row in catalog.iterrows():
        ings = split_ingredients(row["scientific_name"])
        # Filter to ingredients that survived the script's >=4-letter filter
        ings = [i for i in ings if len(i) >= 4 and re.search(r"[a-zA-Z]{4,}", i)]
        # Dedupe within a drug while preserving order (primary = first)
        seen = set()
        ordered: list[str] = []
        for i in ings:
            if i not in seen:
                seen.add(i)
                ordered.append(i)
        drug_ings[row["drug_id"]] = ordered
        for idx, ing in enumerate(ordered):
            map_rows.append({
                "drug_id": int(row["drug_id"]),
                "ingredient": ing,
                "is_primary": idx == 0,
            })
    cat_ing_map = pd.DataFrame(map_rows)

    # ── Build per-ingredient monograph table ───────────────────────────
    # Aggregate drug counts per ingredient
    ing_counts = cat_ing_map.groupby("ingredient")["drug_id"].nunique().to_dict()

    ing_rows = []
    for ing, n in ing_counts.items():
        rec = cache.get(ing, {})
        row = {
            "ingredient": ing,
            "n_drugs": n,
            "found": bool(rec.get("found", False)),
            "source": rec.get("source", "none"),
        }
        for f in SECTIONS:
            row[f] = _normalize_field(rec.get(f, ""))
        ing_rows.append(row)
    ingredients = pd.DataFrame(ing_rows).sort_values("n_drugs", ascending=False)

    # ── Clean catalog table (no monograph fields) ──────────────────────
    cat_clean = catalog.copy()
    cat_clean["drug_id"] = cat_clean["drug_id"].astype(int)

    # ── Emit RAG chunks (one record per chunk per section) ─────────────
    chunks_out = out_dir / "chunks.jsonl"
    n_chunks = 0
    with open(chunks_out, "w") as fh:
        for _, r in ingredients.iterrows():
            if not r["found"]:
                continue
            for section in SECTIONS:
                text = r[section]
                if not text:
                    continue
                for i, ch in enumerate(chunk_text(text)):
                    rec = {
                        "chunk_id": f"{r['ingredient']}::{section}::{i}",
                        "ingredient": r["ingredient"],
                        "section": section,
                        "text": ch,
                        "n_drugs_using": int(r["n_drugs"]),
                        "source": r["source"],
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_chunks += 1

    # ── Persist tabular artifacts (parquet preferred; csv fallback) ────
    def save_table(df: pd.DataFrame, name: str) -> None:
        try:
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
        except Exception:
            df.to_csv(out_dir / f"{name}.csv", index=False)

    save_table(ingredients, "ingredients")
    save_table(cat_clean, "catalog")
    save_table(cat_ing_map, "catalog_ingredient_map")

    # ── Coverage report ────────────────────────────────────────────────
    n_total_ing = len(ingredients)
    n_found = int(ingredients["found"].sum())
    n_total_drugs = len(cat_clean)
    drugs_with_any_found = (
        cat_ing_map.merge(ingredients[["ingredient", "found"]], on="ingredient")
        .groupby("drug_id")["found"].any().sum()
    )
    drugs_with_primary_found = (
        cat_ing_map[cat_ing_map["is_primary"]]
        .merge(ingredients[["ingredient", "found"]], on="ingredient")["found"].sum()
    )

    section_pop = {
        s: int(ingredients[ingredients["found"]][s].str.len().gt(10).sum())
        for s in SECTIONS
    }

    src_counts = ingredients[ingredients["found"]]["source"].value_counts().to_dict()

    # Top uncovered ingredients (highest catalog impact)
    uncovered = ingredients[~ingredients["found"]].sort_values(
        "n_drugs", ascending=False
    ).head(30)

    report = []
    report.append("# Monograph corpus coverage report\n")
    report.append(f"_Built from `{catalog_path.name}` + `{checkpoint_path.name}`._\n")

    report.append("## Catalog\n")
    report.append(f"- Catalog drugs: **{n_total_drugs}**")
    report.append(f"- Drugs with ≥1 extracted ingredient: "
                  f"**{cat_ing_map['drug_id'].nunique()}** "
                  f"({100*cat_ing_map['drug_id'].nunique()/n_total_drugs:.1f}%)")
    report.append(f"- Drugs with monograph for ≥1 ingredient: "
                  f"**{drugs_with_any_found}** "
                  f"({100*drugs_with_any_found/n_total_drugs:.1f}%)")
    report.append(f"- Drugs with monograph for the *primary* ingredient: "
                  f"**{drugs_with_primary_found}** "
                  f"({100*drugs_with_primary_found/n_total_drugs:.1f}%)\n")

    report.append("## Ingredients\n")
    report.append(f"- Unique ingredients: **{n_total_ing}**")
    report.append(f"- With any monograph data: **{n_found}** "
                  f"({100*n_found/n_total_ing:.1f}%)")
    report.append("- Source breakdown: " +
                  ", ".join(f"{k}={v}" for k, v in src_counts.items()) + "\n")

    report.append("## Field population (among found ingredients)\n")
    report.append("| section | populated | % |")
    report.append("|---|---|---|")
    for s in SECTIONS:
        n = section_pop[s]
        report.append(f"| {s} | {n} | {100*n/max(n_found,1):.1f}% |")
    report.append("")

    report.append("## RAG chunks\n")
    report.append(f"- Total chunks: **{n_chunks}**")
    report.append(f"- Chunk size: {CHUNK_WORDS} words "
                  f"(~{CHUNK_WORDS*3} wordpieces, fits SciBERT's 512 limit) "
                  f"with {CHUNK_OVERLAP}-word overlap\n")

    report.append("## Top 30 uncovered ingredients (highest catalog impact)\n")
    report.append("| ingredient | catalog drugs |")
    report.append("|---|---|")
    for _, r in uncovered.iterrows():
        report.append(f"| `{r['ingredient']}` | {int(r['n_drugs'])} |")
    report.append("")

    (out_dir / "coverage_report.md").write_text("\n".join(report))

    # ── Console summary ────────────────────────────────────────────────
    print(f"Wrote {out_dir}/")
    print(f"  ingredients               : {n_total_ing} rows")
    print(f"  catalog                   : {n_total_drugs} rows")
    print(f"  catalog_ingredient_map    : {len(cat_ing_map)} rows")
    print(f"  chunks.jsonl              : {n_chunks} chunks")
    print(f"  coverage_report.md")
    print()
    print(f"Coverage: {n_found}/{n_total_ing} ingredients ({100*n_found/n_total_ing:.1f}%) "
          f"→ {drugs_with_any_found}/{n_total_drugs} drugs "
          f"({100*drugs_with_any_found/n_total_drugs:.1f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog",    default="pharmacy_all_drugs.csv")
    p.add_argument("--checkpoint", default="fetch_checkpoint_v2.json")
    p.add_argument("--out-dir",    default="corpus")
    args = p.parse_args()
    build(Path(args.catalog), Path(args.checkpoint), Path(args.out_dir))


if __name__ == "__main__":
    main()
