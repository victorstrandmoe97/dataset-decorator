"""
risk_decorator.py

MVP: Dataset risk annotation and decorator layer for Hugging Face datasets.

- Automatically detects code-like columns (heuristically)
- Scores code snippets with a simple risk function
- Injects `risk_score` and `is_problematic` into each row
- Provides a decorator that wraps any dataset loader function

Intended usage:

    from datasets import load_dataset
    from risk_decorator import DatasetRiskDecorator, HeuristicCodeColumnDetector, HeuristicRiskScorer

    detector = HeuristicCodeColumnDetector()
    scorer = HeuristicRiskScorer()
    risk_guard = DatasetRiskDecorator(detector=detector, scorer=scorer, threshold=0.5)

    @risk_guard
    def load_devign():
        return load_dataset("DetectVul/devign")

    ds = load_devign()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Protocol,
    Union,
)

from datasets import Dataset, DatasetDict, Features

HF_Dataset = Union[Dataset, DatasetDict]


# ---------------------------------------------------------------------------
# Interfaces / Protocols
# ---------------------------------------------------------------------------


class ICodeColumnDetector(Protocol):
    """
    Detects which columns in a dataset schema contain source code.
    """

    def detect_columns(self, schema: Dict[str, Any]) -> List[str]:
        """
        Args:
            schema: Dataset features/schema dictionary (e.g. ds.features)

        Returns:
            List of column names that contain source code.
        """
        ...


class IRiskScorer(Protocol):
    """
    Scores a code snippet for security / policy risk.
    """

    def score(self, code: str) -> float:
        """
        Args:
            code: Source code snippet

        Returns:
            Risk score in the range [0.0, 1.0].
        """
        ...


class IDatasetAnnotator(Protocol):
    """
    Injects risk metadata into dataset rows.
    """

    def annotate_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            row: Single dataset example

        Returns:
            Updated row with at least:
                - "risk_score": float
                - "is_problematic": bool
        """
        ...


class IDatasetProcessor(Protocol):
    """
    Applies risk analysis to Hugging Face datasets.
    """

    def process(self, dataset: HF_Dataset) -> HF_Dataset:
        """
        Args:
            dataset: HuggingFace Dataset or DatasetDict

        Returns:
            Fully annotated Dataset / DatasetDict.
        """
        ...


class IDatasetRiskDecorator(Protocol):
    """
    Wraps dataset loader functions with risk analysis.
    """

    def __call__(
        self,
        loader_fn: Callable[..., HF_Dataset],
    ) -> Callable[..., HF_Dataset]:
        """
        Args:
            loader_fn: Any function returning a HF Dataset or DatasetDict.

        Returns:
            Wrapped function that returns an annotated dataset.
        """
        ...


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskAnnotation:
    risk_score: float
    is_problematic: bool


@dataclass(frozen=True)
class CodeDetectionResult:
    detected_columns: List[str]


@dataclass
class DatasetRiskConfig:
    threshold: float = 0.5
    fail_on_no_code_columns: bool = False


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


@dataclass
class HeuristicCodeColumnDetector(ICodeColumnDetector):
    """
    Heuristic code-column detector.

    Strategy:
    - Prefer column names containing common code-related tokens.
    - Only consider string-like columns (if dtype information is present).
    """

    name_keywords: List[str] = field(
        default_factory=lambda: [
            "code",
            "source",
            "snippet",
            "body",
            "func",
            "function",
            "solution",
            "response",
            "completion",
            "implementation",
            "output",
            "chosen",   
            "rejected", 
        ]
    )

    def detect_columns(self, schema: Dict[str, Any]) -> List[str]:
        detected: List[str] = []

        for col_name, feature in schema.items():
            lower_name = col_name.lower()

            # 1) Name-based heuristic
            if not any(k in lower_name for k in self.name_keywords):
                continue

            # 2) (Soft) type-based check: only string-like features if dtype is available
            dtype = getattr(feature, "dtype", None)
            if dtype is not None and "string" not in str(dtype):
                # if dtype is explicitly non-string, skip
                continue

            detected.append(col_name)

        return detected


@dataclass
class HeuristicRiskScorer(IRiskScorer):
    """
    Very simple heuristic risk scorer for code.

    Scoring strategy:
        - Count occurrences of "suspicious" tokens
        - Normalize by a configurable factor
        - Clamp to [0.0, 1.0]

    This is intentionally dumb but deterministic and side-effect free.
    Replace with a learned model / static analyzer for real use.
    """

    dangerous_tokens: List[str] = field(
        default_factory=lambda: [
            "eval(",
            "exec(",
            "system(",
            "subprocess.",
            "os.popen",
            "pickle.loads",
            "yaml.load(",
            "rm -rf",
            "DROP TABLE",
            "xp_cmdshell",
            "shell=True",
            "sudo ",
        ]
    )
    normalization_factor: float = 5.0

    def score(self, code: str) -> float:
        text = code or ""
        lower = text.lower()

        hits = 0
        for token in self.dangerous_tokens:
            # count non-overlapping occurrences, case-insensitive
            hits += lower.count(token.lower())

        raw_score = hits / self.normalization_factor
        if raw_score > 1.0:
            return 1.0
        if raw_score < 0.0:
            return 0.0
        return float(raw_score)


@dataclass
class DatasetAnnotator(IDatasetAnnotator):
    """
    Row-level annotator that:
    - Extracts code from one or more code columns
    - Computes a single row-level risk score (max over code columns)
    - Adds "risk_score" and "is_problematic"
    """

    scorer: IRiskScorer
    code_columns: List[str]
    threshold: float = 0.5

    def annotate_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # Collect all code snippets in this row
        codes: List[str] = []
        for col in self.code_columns:
            val = row.get(col)
            if isinstance(val, str):
                codes.append(val)

        if not codes:
            # No code in this row: risk_score = 0
            risk_score = 0.0
        else:
            # Risk = max risk across all code columns
            scores = [self.scorer.score(code) for code in codes]
            risk_score = float(max(scores)) if scores else 0.0

        annotation = RiskAnnotation(
            risk_score=risk_score,
            is_problematic=bool(risk_score >= self.threshold),
        )

        # Inject into row without mutating the original dict in place
        new_row = dict(row)
        new_row["risk_score"] = annotation.risk_score
        new_row["is_problematic"] = annotation.is_problematic

        return new_row


@dataclass
class DatasetRiskProcessor(IDatasetProcessor):
    """
    Applies risk analysis to an entire Hugging Face Dataset or DatasetDict.

    - Uses ICodeColumnDetector to identify code columns
    - Uses IRiskScorer + DatasetAnnotator to annotate rows
    """

    detector: ICodeColumnDetector
    scorer: IRiskScorer
    config: DatasetRiskConfig = field(default_factory=DatasetRiskConfig)

    def _detect_code_columns(self, dataset: Dataset) -> CodeDetectionResult:
        features: Features = dataset.features
        schema_dict: Dict[str, Any] = dict(features)
        cols = self.detector.detect_columns(schema_dict)
        return CodeDetectionResult(detected_columns=cols)

    def _annotate_single_dataset(self, dataset: Dataset) -> Dataset:
        detection = self._detect_code_columns(dataset)
        print("Detected code columns:", detection.detected_columns)

        if not detection.detected_columns:
            if self.config.fail_on_no_code_columns:
                raise ValueError("No code columns detected in dataset.")
            # No code columns → still return dataset, but risk fields will be zeroed
            annotator = DatasetAnnotator(
                scorer=self.scorer,
                code_columns=[],
                threshold=self.config.threshold,
            )
        else:
            annotator = DatasetAnnotator(
                scorer=self.scorer,
                code_columns=detection.detected_columns,
                threshold=self.config.threshold,
            )

        # map() returns a new Dataset
        return dataset.map(
            annotator.annotate_row,
            desc="Annotating dataset with risk scores",
        )

    def process(self, dataset: HF_Dataset) -> HF_Dataset:
        if isinstance(dataset, DatasetDict):
            # Process each split individually
            processed_splits: Dict[str, Dataset] = {}
            for split_name, split_ds in dataset.items():
                processed_splits[split_name] = self._annotate_single_dataset(split_ds)
            return DatasetDict(processed_splits)

        if isinstance(dataset, Dataset):
            return self._annotate_single_dataset(dataset)

        raise TypeError(
            f"DatasetRiskProcessor.process expected Dataset or DatasetDict, "
            f"got {type(dataset)!r}"
        )


@dataclass
class DatasetRiskDecorator(IDatasetRiskDecorator):
    """
    Decorator that wraps any Hugging Face dataset loader and returns
    an annotated dataset with risk metadata.

    Example:

        detector = HeuristicCodeColumnDetector()
        scorer = HeuristicRiskScorer()
        risk_guard = DatasetRiskDecorator(detector=detector, scorer=scorer, threshold=0.5)

        @risk_guard
        def load_my_data():
            return load_dataset("yahma/alpaca-cleaned")

        ds = load_my_data()
    """

    detector: ICodeColumnDetector
    scorer: IRiskScorer
    threshold: float = 0.5
    fail_on_no_code_columns: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0.0, 1.0]")

    def __call__(
        self,
        loader_fn: Callable[..., HF_Dataset],
    ) -> Callable[..., HF_Dataset]:
        processor = DatasetRiskProcessor(
            detector=self.detector,
            scorer=self.scorer,
            config=DatasetRiskConfig(
                threshold=self.threshold,
                fail_on_no_code_columns=self.fail_on_no_code_columns,
            ),
        )

        def wrapped_loader(*args: Any, **kwargs: Any) -> HF_Dataset:
            ds = loader_fn(*args, **kwargs)
            return processor.process(ds)

        # Preserve name and docstring for nicer debugging / introspection
        wrapped_loader.__name__ = getattr(loader_fn, "__name__", "wrapped_loader")
        wrapped_loader.__doc__ = getattr(loader_fn, "__doc__", None)

        return wrapped_loader


# ---------------------------------------------------------------------------
# Optional: basic sanity test when run as a script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # This block is safe to delete if you don't want any side effects.
    from datasets import Dataset

    # Tiny in-memory example
    data = {
        "instruction": [
            "Write a Python function that deletes all files in /tmp",
            "Print 'hello world'.",
        ],
        "response": [
            "import os\nos.system('rm -rf /tmp/*')",
            "print('hello world')",
        ],
    }
    ds = Dataset.from_dict(data)

    detector = HeuristicCodeColumnDetector()
    scorer = HeuristicRiskScorer()
    processor = DatasetRiskProcessor(
        detector=detector,
        scorer=scorer,
        config=DatasetRiskConfig(threshold=0.5),
    )

    annotated = processor.process(ds)
    print(annotated[0])
    print(annotated[1])
