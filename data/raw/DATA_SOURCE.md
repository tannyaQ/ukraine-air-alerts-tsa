# Data provenance

- **Source repository:** https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset
- **License:** MIT
- **Retrieved:** 2026-06-21
- **Download:** `python scripts/download_data.py`

## Files

| file | committed | granularity | starts | notes |
|------|-----------|-------------|--------|-------|
| `volunteer_data_en.csv` | yes (source of truth) | oblast | 2022-02-25 | volunteer-collected (eTryvoga) |
| `official_data_en.csv`  | no (~28 MB, .gitignore) | oblast → raion (Dec 2025) | 2022-03-15 | official; used only for crosscheck |

## Verified schema (interval level)

| column | meaning | format |
|--------|---------|--------|
| `region` | oblast / city name | e.g. `Kyiv City`, `Cherkaska oblast` |
| `started_at` | alert start | ISO 8601, UTC (`+00:00`) |
| `finished_at` | alert end | ISO 8601, UTC (`+00:00`) |
| `naive` | flag (semantics verified empirically in EDA) | boolean |

> Times are stored in **UTC**; the analysis converts them to **Europe/Kyiv**.

## Checksums (sha256)

Filled by `scripts/download_data.py` on download:

- `volunteer_data_en.csv`: `245afff21d2a00317918f6da35d17e47d67512cf5858147edb2a3453f50675dd`
- `official_data_en.csv`: `a36eb2fac7606765b67967f4b70be0a2c93d15710972c01e27427be128ab7d58`
