"""
Example: Using DatasetRiskDecorator with yahma/alpaca-cleaned
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
scorer = scorer = DebertaRiskScorer("microsoft/deberta-v3-base")

# Default threshold = 0.5 is fine here
risk_guard = DatasetRiskDecorator(
    detector=detector,
    scorer=scorer,
    threshold=0.5,
    filter_mode="keep_safe"
)


@risk_guard
def load_alpaca():
    return load_dataset("yahma/alpaca-cleaned")

if __name__ == "__main__":
    ds = load_alpaca()
    split = ds["train"]

    train = ds["train"]
    problematic = train.filter(lambda x: x["is_problematic"])

    for i in range(min(20, len(problematic))):
        row = problematic[i]
        print("\n---")
        print("risk_score:", row["risk_score"])