Got it — here is a **clean, publication-ready README** with tightened language, fixed typos, consistent naming, and accurate technical framing. You can drop this directly into `README.md`:

---

# Dataset Risk Decorator

A lightweight, plug-and-play **risk annotation and filtering decorator for Hugging Face datasets**.
It automatically detects code-like columns, assigns a security risk score to each sample, and injects:

* `risk_score`
* `is_problematic`

directly into the dataset **without breaking the Hugging Face API**.

Designed for:

* Dataset auditing
* Safety filtering before fine-tuning
* Preference / DPO preprocessing
* Security research workflows

---

## Features

* ✅ Works with both `datasets.Dataset` and `datasets.DatasetDict`
* ✅ Drop-in decorator for any `load_dataset` function
* ✅ Automatic code column detection (name- and content-based)
* ✅ Heuristic or learned risk scoring (DeBERTa-based)
* ✅ Trainer-compatible output
* ✅ Zero changes required to downstream Hugging Face pipelines
* ✅ Optional automatic filtering (`keep_safe`, `keep_problematic`)

---

## Installation

### From PyPI (once published)

```bash
pip install dataset-risk-decorator
```

### For Local Development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

---

## Quick Start

```python
from datasets import load_dataset
from dataset_risk_decorator import (
    DatasetRiskDecorator,
    HeuristicCodeColumnDetector,
    DebertaRiskScorer,
)

detector = HeuristicCodeColumnDetector()
scorer = DebertaRiskScorer("deberta-devign-risk-model")

risk_guard = DatasetRiskDecorator(
    detector=detector,
    scorer=scorer,
    threshold=0.5,
    filter_mode="keep_safe",   # Automatically drops risky samples
)

@risk_guard
def load_data():
    return load_dataset("yahma/alpaca-cleaned")

ds = load_data()
print(ds["train"][0])
```

Each row now contains:

```text
risk_score
is_problematic
```

---

## Viewing Annotations

Print a fully annotated row:

```python
print(ds["train"][0])
```

Filter only problematic samples:

```python
risky = ds["train"].filter(lambda x: x["is_problematic"])
print(len(risky))
print(risky[0])
```

---

## Examples

Runnable examples using real datasets are available in `examples/`:

* `devign_example.py` → `DetectVul/devign`
* `cybernative_dpo_example.py` → `CyberNative/Code_Vulnerability_Security_DPO`
* `alpaca_cleaned_example.py` → `yahma/alpaca-cleaned`

Run them:

```bash
python examples/devign_example.py
python examples/cybernative_dpo_example.py
python examples/alpaca_cleaned_example.py
```

---

## Development Reset (Clean Environment)

```bash
deactivate
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

---

## Verify Installation

```bash
python -c "from dataset_risk_decorator import DatasetRiskDecorator; print('OK')"
python -c "import datasets; print('datasets OK')"
python examples/alpaca_cleaned_example.py
```

---

## Project Structure

```text
dataset-risk-decorator/
├── dataset_risk_decorator/
│   ├── __init__.py
│   └── core.py
├── examples/
├── pyproject.toml
└── README.md
```

---

## License

Apache-2.0

---

## Disclaimer

The current risk scorer is **heuristic / MVP-grade by design**.
It is intended to validate data pipelines and safety filtering mechanics prior to deploying a fully trained classifier (e.g., DeBERTa, CodeBERT, or hybrid ensembles).

---

## Maintainer

**Durinn Research**
Contact: [victorstrandmoe@gmail.com](mailto:victorstrandmoe@gmail.com)

---

If you want, I can also add:

* A **Streaming Dataset** section
* A **Trainer / HF fine-tuning integration example**
* A **BigQuery → HF ingestion example**
* Or a **Model Card–style evaluation section** for the scorer.
