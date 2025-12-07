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
* ✅ Optional automatic filtering:

  * `filter_mode="none"` (annotate only)
  * `filter_mode="keep_safe"`
  * `filter_mode="keep_problematic"`

---

## Installation

### From PyPI

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

## Quick Start (Annotate Only)

In the most conservative setup, you **only annotate** the dataset and decide what to do with risky rows **in your own code**.

```python
from datasets import load_dataset
from dataset_risk_decorator import (
    DatasetRiskDecorator,
    HeuristicCodeColumnDetector,
    DebertaRiskScorer,
)

detector = HeuristicCodeColumnDetector()
scorer = DebertaRiskScorer("<model_path>")

# Annotate-only: no automatic filtering, you handle risk downstream.
risk_guard = DatasetRiskDecorator(
    detector=detector,
    scorer=scorer,
    threshold=0.5,
    filter_mode="none",   # <-- keep all rows, just add risk fields
)

@risk_guard
def load_data():
    return load_dataset("yahma/alpaca-cleaned")

ds = load_data()
train = ds["train"]

print(train[0])
```

Each row now contains:

```text
risk_score
is_problematic
```

---

## Filtering Modes

`filter_mode` controls what the **decorated loader returns**:

| `filter_mode`        | Behavior                                                               |
| -------------------- | ---------------------------------------------------------------------- |
| `"none"`             | Annotate rows; keep **all** samples.                                   |
| `"keep_safe"`        | Keep only `is_problematic == False` (low-risk samples).                |
| `"keep_problematic"` | Keep only `is_problematic == True` (high-risk samples for inspection). |

> **Important:**
>
> * `filter_mode` is applied **after** annotation inside `DatasetRiskProcessor`.
> * You can still apply additional `dataset.filter(...)` calls on top of the returned dataset if you want more complex logic.

---

## Common Usage Patterns

### 1. Keep All Data, Handle Risk in Code

You want to **keep the full dataset**, but:

* Train only on safe samples
* Log or inspect problematic ones
* Possibly export risky rows separately

```python
from datasets import load_dataset
from dataset_risk_decorator import (
    DatasetRiskDecorator,
    HeuristicCodeColumnDetector,
    DebertaRiskScorer,
)

detector = HeuristicCodeColumnDetector()
scorer = DebertaRiskScorer("<model_path>")

risk_guard = DatasetRiskDecorator(
    detector=detector,
    scorer=scorer,
    threshold=0.5,
    filter_mode="none",  # annotate-only
)

@risk_guard
def load_alpaca():
    return load_dataset("yahma/alpaca-cleaned")

ds = load_alpaca()
train = ds["train"]

safe = train.filter(lambda r: r["is_problematic"] is False)
risky = train.filter(lambda r: r["is_problematic"] is True)

print("Total:", len(train))
print("Safe samples:", len(safe))
print("Risky samples:", len(risky))

# Example: Train on safe, log risky
for row in risky.select(range(min(10, len(risky)))):
    print("\n[⚠️ RISKY SAMPLE]")
    print("risk_score:", row["risk_score"])
    print("instruction:", row.get("instruction", "")[:160], "...")
```

This pattern is ideal if you:

* Want **full visibility** of your data
* Need **custom business logic** around risky samples (e.g. manual review queues, separate storage, red-team harnesses)

---

### 2. Use `keep_safe` for Training-Ready Datasets

You want a loader that **only returns safe samples**, ready to plug into training scripts.

```python
from datasets import load_dataset
from dataset_risk_decorator import (
    DatasetRiskDecorator,
    HeuristicCodeColumnDetector,
    DebertaRiskScorer,
)

detector = HeuristicCodeColumnDetector()
scorer = DebertaRiskScorer("<model_path>")

train_guard = DatasetRiskDecorator(
    detector=detector,
    scorer=scorer,
    threshold=0.5,
    filter_mode="keep_safe",  # only low-risk rows survive
)

@train_guard
def load_safe_alpaca():
    return load_dataset("yahma/alpaca-cleaned")

ds_safe = load_safe_alpaca()
safe_train = ds_safe["train"]

print("Safe training samples:", len(safe_train))
print(safe_train[0])  # already annotated but filtered
```

This pattern is ideal if you:

* Want a **drop-in replacement** for `load_dataset`
* Want your training code to stay unchanged (just swap loader)
* Use HF Trainer / PyTorch DataLoader directly on the filtered dataset

---

### 3. Use `keep_problematic` for Offline Analysis / Red-Teaming

You want a dataset that contains only **high-risk samples** to:

* Audit model behavior
* Build test harnesses
* Red-team specific vulnerable patterns

```python
from datasets import load_dataset
from dataset_risk_decorator import (
    DatasetRiskDecorator,
    HeuristicCodeColumnDetector,
    DebertaRiskScorer,
)

detector = HeuristicCodeColumnDetector()
scorer = DebertaRiskScorer("<model_path>")

analysis_guard = DatasetRiskDecorator(
    detector=detector,
    scorer=scorer,
    threshold=0.6,
    filter_mode="keep_problematic",   # keep only risky rows
)

@analysis_guard
def load_risky_devign():
    return load_dataset("microsoft/Devign")

risky_ds = load_risky_devign()
split_name = next(iter(risky_ds.keys()))
risky_split = risky_ds[split_name]

print("Problematic rows:", len(risky_split))

for row in risky_split.select(range(min(20, len(risky_split)))):
    print("\n---")
    print("risk_score:", row["risk_score"])
    print("code:", row.get("func", "")[:160], "...")
```

This pattern is ideal if you:

* Want to build a **small, focused dataset of sketchy code**
* Use it as **negative examples** in training or evaluation
* Feed it into **specialized analysis pipelines** (static analysis, symbolic execution, etc.)

---

## Dataset-Specific Examples

### Example: `yahma/alpaca-cleaned`

**Goal:** annotate, then split into safe/risky for training vs analysis.

```python
from datasets import load_dataset
from dataset_risk_decorator import (
    DatasetRiskDecorator,
    HeuristicCodeColumnDetector,
    DebertaRiskScorer,
)

detector = HeuristicCodeColumnDetector()
scorer = DebertaRiskScorer("<model_path>")

risk_guard = DatasetRiskDecorator(
    detector=detector,
    scorer=scorer,
    threshold=0.5,
    filter_mode="none",   # annotate-only
)

@risk_guard
def load_alpaca():
    return load_dataset("yahma/alpaca-cleaned")

ds = load_alpaca()
train = ds["train"]

safe_train = train.filter(lambda r: not r["is_problematic"])
risky_train = train.filter(lambda r: r["is_problematic"])

print("Safe:", len(safe_train), "Risky:", len(risky_train))
```

---

### Example: `CyberNative/Code_Vulnerability_Security_DPO`

**Goal:** Slightly stricter threshold, inspect problematic preference data.

```python
from datasets import load_dataset
from dataset_risk_decorator import (
    DatasetRiskDecorator,
    HeuristicCodeColumnDetector,
    DebertaRiskScorer,
)

detector = HeuristicCodeColumnDetector()
scorer = DebertaRiskScorer("<model_path>")

risk_guard = DatasetRiskDecorator(
    detector=detector,
    scorer=scorer,
    threshold=0.6,
    filter_mode="none",  # keep all, we want to view both sides
)

@risk_guard
def load_cybernative_dpo():
    return load_dataset("CyberNative/Code_Vulnerability_Security_DPO")

ds = load_cybernative_dpo()
split_name = next(iter(ds.keys()))
split = ds[split_name]

problematic = split.filter(lambda x: x["is_problematic"])
print("Problematic rows:", len(problematic))

for row in problematic.select(range(min(20, len(problematic)))):
    print("\n---")
    print("risk_score:", row["risk_score"])
    print("chosen:", row.get("chosen", "")[:160], "...")
    print("rejected:", row.get("rejected", "")[:160], "...")
```

If you later want a **training-only variant**:

```python
train_guard = DatasetRiskDecorator(
    detector=detector,
    scorer=scorer,
    threshold=0.6,
    filter_mode="keep_safe",
)

@train_guard
def load_cybernative_dpo_safe():
    return load_dataset("CyberNative/Code_Vulnerability_Security_DPO")

safe_ds = load_cybernative_dpo_safe()
safe_split = safe_ds[next(iter(safe_ds.keys()))]

print("Safe rows:", len(safe_split))
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
https://huggingface.co/durinn
https://durinn-as.web.app/
---

If you want, next step I can add a small **“Integration Recipes”** section:

* HF `Trainer` integration (how to pass only safe splits)
* Multi-dataset pipelines (Devign + CyberNative + Alpaca combined)
* How to log risky rows to **WandB / MLflow** for inspection.
