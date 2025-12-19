# Changelog

## v1.0.1 — 2025-12-20

- Added `ECE 34/new_class/download_datasets.py` for automated Kaggle dataset downloads (tomato, chili, potato, rice) with 80/20 split.
- Updated `README.md` (top-level) with Dataset Management section, fixed headings and code fences, removed legacy bundle options.
- Updated `ECE 34/new_class/README.md` with dataset downloader usage and a generic training section using `train_classifier.py`.
- Removed `ECE 34/new_class/DATASET_STATUS.md` (replaced by downloader script and README guidance).
- Minor formatting cleanup across docs; standardized PowerShell examples.

Notes:
- Datasets are large and not tracked in git; use `download_datasets.py` to fetch locally.
- Release tag created on branch `patch6` to include latest docs and scripts.