# Changelog
All notable changes to this project will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/).
---
## [0.1.3] - 2025-12-07

### Changed
- Update readme

### Commits
- `706601` – add filter

---

## [0.1.2] - 2025-12-07
### Added
- Dataset-level filtering support via `filter_mode`
  - `keep_safe`
  - `keep_problematic`

### Changed
- Improved dataset processing flow to support post-annotation filtering

### Commits
- `238642b` – add filter

---

## [0.1.1] - 2025-12-06
### Added
- Multi-source training pipeline support
- Evaluation strategy for risk model training
- DeBERTa-based fine-tuning integration using the Devign dataset
- Content-aware code detection in addition to column-name heuristics

### Changed
- Expanded README with training and usage documentation
- Improved heuristic detection logic for more reliable column identification

### Commits
- `ae8f56f` – eval_strategy  
- `102a09b` – multisource training  
- `8971606` – add finetuning using deberta devign  
- `ac328ed` – find code based on input as well as column names  

---

## [0.1.0] - 2025-12-06
### Added
- Core dataset row annotation logic
- `risk_score` and `is_problematic` injection
- Hugging Face `Dataset` and `DatasetDict` support
- Heuristic risk decorator
- Python package structure
- Deployment script
- Initial README

### Commits
- `457bca7` – python package decorating rows  
- `4260805` – add heuristic risk decorator to each dataset  
- `bf04fb2` – deploy script  
- `9dba1a1` – readme  
- `5d5342d` – Initial commit  

---
