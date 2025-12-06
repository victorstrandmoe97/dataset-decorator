"""
Example: Using DatasetRiskDecorator with yahma/alpaca-cleaned
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from risk_decorator import (
    DatasetRiskDecorator,
    HeuristicCodeColumnDetector,
    HeuristicRiskScorer,
)

detector = HeuristicCodeColumnDetector()
scorer = HeuristicRiskScorer()

# Default threshold = 0.5 is fine here
risk_guard = DatasetRiskDecorator(
    detector=detector,
    scorer=scorer,
    threshold=0.5,
)

@risk_guard
def load_alpaca():
    return load_dataset("yahma/alpaca-cleaned")

if __name__ == "__main__":
    ds = load_alpaca()
    split = ds["train"]

    print("Total rows:", len(split))
    print("Columns:", split.column_names)

    # Show a few annotated samples
    for i in range(3):
        row = split[i]
        print(f"\nRow {i}")
        print("risk_score:", row["risk_score"])
        print("is_problematic:", row["is_problematic"])
