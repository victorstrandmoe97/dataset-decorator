"""
Example: Using DatasetRiskDecorator with DetectVul/devign
"""
# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from dataset_risk_decorator.core import (
    DatasetRiskDecorator,
    HeuristicCodeColumnDetector,
    DebertaRiskScorer,
)

# Instantiate core components
detector = HeuristicCodeColumnDetector()
scorer = scorer = DebertaRiskScorer("deberta-devign-risk-model")

# Create decorator with default threshold = 0.5
risk_guard = DatasetRiskDecorator(
    detector=detector,
    scorer=scorer,
    threshold=0.5,
    filter_mode="keep_safe"
)

@risk_guard
def load_devign():
    # Devign typically provides train/validation/test splits
    return load_dataset("DetectVul/devign")

if __name__ == "__main__":
    ds = load_devign()

    train = ds["train"]

    # ✅ NO FILTER CALLS IN USER LAND
    print("Final train size:", len(train))

    problematic = train.filter(lambda x: x["is_problematic"])

    for i in range(min(20, len(problematic))):
        row = problematic[i]
        print("\n---")
        print("risk_score:", row["risk_score"])