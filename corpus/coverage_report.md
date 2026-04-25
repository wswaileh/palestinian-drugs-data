# Monograph corpus coverage report

_Built from `pharmacy_all_drugs.csv` + `fetch_checkpoint_v2.json`._

## Catalog

- Catalog drugs: **2752**
- Drugs with ≥1 extracted ingredient: **2723** (98.9%)
- Drugs with monograph for ≥1 ingredient: **2261** (82.2%)
- Drugs with monograph for the *primary* ingredient: **2235** (81.2%)

## Ingredients

- Unique ingredients: **1020**
- With any monograph data: **657** (64.4%)
- Source breakdown: openfda=571, dailymed=86

## Field population (among found ingredients)

| section | populated | % |
|---|---|---|
| indications | 654 | 99.5% |
| contraindications | 463 | 70.5% |
| mechanism | 457 | 69.6% |
| drug_interactions | 391 | 59.5% |
| side_effects | 469 | 71.4% |
| warnings | 257 | 39.1% |

## RAG chunks

- Total chunks: **5216**
- Chunk size: 150 words (~450 wordpieces, fits SciBERT's 512 limit) with 30-word overlap

## Top 30 uncovered ingredients (highest catalog impact)

| ingredient | catalog drugs |
|---|---|
| `etoricoxib` | 25 |
| `human normal immunoglobulin` | 10 |
| `scopolamine butylbromide` | 10 |
| `zuclopenthixol` | 9 |
| `dichlorobenzyl alcohol` | 6 |
| `fusidic acid` | 6 |
| `amylmetacresol` | 6 |
| `alfacalcidol` | 5 |
| `recombinat factor` | 4 |
| `glibenclamide` | 4 |
| `valsartan salt complex` | 4 |
| `bifonazole` | 4 |
| `calcipotriol` | 4 |
| `valsartan hct` | 4 |
| `betahistine` | 4 |
| `dapoxetine` | 4 |
| `nystatin- neomycin-garamicidin` | 4 |
| `lamotrigine disp chew` | 4 |
| `dexamethasone base` | 3 |
| `xylometazoline w v` | 3 |
| `moroctocog alpha` | 3 |
| `mebeverine` | 3 |
| `lornoxicam` | 3 |
| `carbocisteine` | 3 |
| `follitropin beta` | 3 |
| `cinnarizine` | 3 |
| `dexamethasone na` | 3 |
| `afatinib dimaleate` | 3 |
| `diltiazim` | 3 |
| `deferazirox` | 3 |
