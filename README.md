# Dataset Risk Decorator Examples

This directory contains runnable examples demonstrating how to use
`DatasetRiskDecorator` with real Hugging Face datasets:

- `devign_example.py` → DetectVul/devign
- `cybernative_dpo_example.py` → CyberNative/Code_Vulnerability_Security_DPO
- `alpaca_cleaned_example.py` → yahma/alpaca-cleaned

## Run

```bash
pip install datasets
python examples/devign_example.py
python examples/cybernative_dpo_example.py
python examples/alpaca_cleaned_example.py


```
deactivate
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

```


verify 
```
python -c "from dataset_risk_decorator import DatasetRiskDecorator; print('OK')"
python -c "import datasets; print('datasets OK')"
python examples/alpaca_cleaned_example.py

```