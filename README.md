# Dataset Risk Decorator

A lightweight, plug-and-play **risk annotation decorator for Hugging Face datasets**.  
It automatically detects code columns, assigns a heuristic security risk score to each sample, and injects:

- `risk_score`
- `is_problematic`

directly into the dataset without breaking the Hugging Face API.

Designed for:
- Dataset auditing
- Safety filtering before fine-tuning
- Preference / DPO preprocessing
- Security research workflows

---

## Features

- ✅ Works with `datasets.Dataset` and `datasets.DatasetDict`
- ✅ Drop-in decorator for any `load_dataset` function
- ✅ Automatic code column detection (name-based)
- ✅ Heuristic risk scoring (MVP)
- ✅ Trainer-compatible output
- ✅ Zero changes required to downstream Hugging Face pipelines

---

## Installation

From PyPI (once published):

```bash
pip install dataset-risk-decorator
```

For local development:

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
    HeuristicRiskScorer,
)

detector = HeuristicCodeColumnDetector()
scorer = HeuristicRiskScorer()

risk_guard = DatasetRiskDecorator(
    detector=detector,
    scorer=scorer,
    threshold=0.5,
)

@risk_guard
def load_data():
    return load_dataset("yahma/alpaca-cleaned")

ds = load_data()
print(ds["train"][0])
```

Each row will now contain:

```text
risk_score
is_problematic
```

---

## Examples

Runnable examples using real datasets are available in `examples/`:

- `devign_example.py` → DetectVul/devign  
- `cybernative_dpo_example.py` → CyberNative/Code_Vulnerability_Security_DPO  
- `alpaca_cleaned_example.py` → yahma/alpaca-cleaned  

Run them:

```bash
python examples/devign_example.py
python examples/cybernative_dpo_example.py
python examples/alpaca_cleaned_example.py
```

---

## Viewing Annotations

Print a full annotated row:

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

The current risk scorer is **heuristic and intentionally minimal**.  
It is designed to validate data pipelines and preprocessing mechanics prior to training a learned classifier (e.g., DeBERTa).

---

## Maintainer

Durinn Research  
Contact: victorstrandmoe@gmail.com
