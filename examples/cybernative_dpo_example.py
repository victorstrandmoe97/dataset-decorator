"""
Example: Using DatasetRiskDecorator with CyberNative/Code_Vulnerability_Security_DPO
"""
# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from dataset_risk_decorator.core import (
    DatasetRiskDecorator,
    HeuristicCodeColumnDetector,
    DebertaRiskScorer,
)

detector = HeuristicCodeColumnDetector()
scorer = scorer = DebertaRiskScorer("deberta-devign-risk-model")

# Slightly stricter threshold for preference data
risk_guard = DatasetRiskDecorator(
    detector=detector,
    scorer=scorer,
    threshold=0.6,
    filter_mode="keep_safe"
)

@risk_guard
def load_cybernative_dpo():
    return load_dataset("CyberNative/Code_Vulnerability_Security_DPO")

if __name__ == "__main__":
    ds = load_cybernative_dpo()

    # Often this dataset has a single split
    split_name = next(iter(ds.keys()))
    split = ds[split_name]

    train = ds["train"]
    problematic = train.filter(lambda x: x["is_problematic"])

    for i in range(min(20, len(problematic))):
        row = problematic[i]
        print("\n---")
        print("risk_score:", row["risk_score"])
