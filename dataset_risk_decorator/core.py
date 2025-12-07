"""
risk_decorator.py

MVP: Dataset risk annotation + filtering layer for Hugging Face datasets.

- Automatically detects code-like columns (heuristically)
- Scores code snippets with a learned DeBERTa model
- Injects `risk_score` and `is_problematic` into each row
- Optionally FILTERS the dataset based on the probability score
- Provides a decorator that wraps any dataset loader function
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
    Literal,
    Optional,
)

from datasets import Dataset, DatasetDict, Features

HF_Dataset = Union[Dataset, DatasetDict]

# ---------------------------------------------------------------------------
# Interfaces / Protocols
# ---------------------------------------------------------------------------

class ICodeColumnDetector(Protocol):
    def detect_columns(self, schema: Dict[str, Any]) -> List[str]: ...


class IRiskScorer(Protocol):
    def score(self, code: str) -> float: ...


class IDebertaRiskScorer(IRiskScorer, Protocol):
    def __init__(self, model_path: str, device: str | None = None) -> None: ...
    def score(self, code: str) -> float: ...


class IDatasetAnnotator(Protocol):
    def annotate_row(self, row: Dict[str, Any]) -> Dict[str, Any]: ...


class IDatasetProcessor(Protocol):
    def process(self, dataset: HF_Dataset) -> HF_Dataset: ...


class IDatasetRiskDecorator(Protocol):
    def __call__(
        self,
        loader_fn: Callable[..., HF_Dataset],
    ) -> Callable[..., HF_Dataset]: ...

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
    filter_mode: Literal["none", "keep_safe", "keep_problematic"] = "none"

# ---------------------------------------------------------------------------
# Heuristic Code Column Detection
# ---------------------------------------------------------------------------

import re

@dataclass
class HeuristicCodeColumnDetector(ICodeColumnDetector):
    name_keywords: List[str] = field(default_factory=lambda: [
        "code", "source", "snippet", "body", "func", "function",
        "solution", "response", "completion", "implementation",
        "output", "chosen", "rejected",
    ])

    code_regexes: List[re.Pattern] = field(default_factory=lambda: [
        re.compile(r"\b(def|class|import|from)\b"),
        re.compile(r"\b(function|const|let|var)\b"),
        re.compile(r";\s*$"),
        re.compile(r"#include\b"),
        re.compile(r"\b(public|private|static)\b"),
        re.compile(r"{\s*$"),
        re.compile(r"}\s*$"),
    ])

    sample_size: int = 50
    min_hits: int = 3

    def detect_columns(self, schema: Dict[str, Any]) -> List[str]:
        return self._detect_by_name(schema)

    def detect_from_dataset(self, dataset: Dataset) -> List[str]:
        candidates = [
            col for col, feat in dataset.features.items()
            if getattr(feat, "dtype", None) is None
            or "string" in str(getattr(feat, "dtype", "")).lower()
        ]

        hit_counter: Dict[str, int] = {c: 0 for c in candidates}
        sample = dataset.select(range(min(self.sample_size, len(dataset))))

        for row in sample:
            for col in candidates:
                val = row.get(col)
                if not isinstance(val, str):
                    continue
                for rx in self.code_regexes:
                    if rx.search(val):
                        hit_counter[col] += 1

        detected = [c for c, hits in hit_counter.items() if hits >= self.min_hits]

        if not detected:
            detected = self._detect_by_name(dict(dataset.features))

        return detected

    def _detect_by_name(self, schema: Dict[str, Any]) -> List[str]:
        detected: List[str] = []

        for col_name, feature in schema.items():
            lower_name = col_name.lower()
            if not any(k in lower_name for k in self.name_keywords):
                continue

            dtype = getattr(feature, "dtype", None)
            if dtype is not None and "string" not in str(dtype).lower():
                continue

            detected.append(col_name)

        return detected

# ---------------------------------------------------------------------------
# DeBERTa Scorer
# ---------------------------------------------------------------------------

import torch
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer

@dataclass
class DebertaRiskScorer(IDebertaRiskScorer):
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = DebertaV2Tokenizer.from_pretrained(
            "microsoft/deberta-v3-base"
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, code: str) -> float:
        if not isinstance(code, str) or not code.strip():
            return 0.0

        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

        return float(probs[0, 1].item())

# ---------------------------------------------------------------------------
# Row Annotator
# ---------------------------------------------------------------------------

@dataclass
class DatasetAnnotator(IDatasetAnnotator):
    scorer: IRiskScorer
    code_columns: List[str]
    threshold: float = 0.5

    def annotate_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        codes: List[str] = []
        for col in self.code_columns:
            val = row.get(col)
            if isinstance(val, str):
                codes.append(val)

        if not codes:
            risk_score = 0.0
        else:
            risk_score = float(max(self.scorer.score(code) for code in codes))

        new_row = dict(row)
        new_row["risk_score"] = risk_score
        new_row["is_problematic"] = bool(risk_score >= self.threshold)

        return new_row

# ---------------------------------------------------------------------------
# Dataset Processor (ANNOTATE + FILTER)
# ---------------------------------------------------------------------------

@dataclass
class DatasetRiskProcessor(IDatasetProcessor):
    detector: ICodeColumnDetector
    scorer: IRiskScorer
    config: DatasetRiskConfig = field(default_factory=DatasetRiskConfig)

    def _detect_code_columns(self, dataset: Dataset) -> CodeDetectionResult:
        if hasattr(self.detector, "detect_from_dataset"):
            cols = self.detector.detect_from_dataset(dataset)
        else:
            schema_dict: Dict[str, Any] = dict(dataset.features)
            cols = self.detector.detect_columns(schema_dict)

        return CodeDetectionResult(detected_columns=cols)

    def _annotate_single_dataset(self, dataset: Dataset) -> Dataset:
        detection = self._detect_code_columns(dataset)
        print("Detected code columns:", detection.detected_columns)

        if not detection.detected_columns:
            if self.config.fail_on_no_code_columns:
                raise ValueError("No code columns detected in dataset.")
            annotator = DatasetAnnotator(self.scorer, [], self.config.threshold)
        else:
            annotator = DatasetAnnotator(
                self.scorer,
                detection.detected_columns,
                self.config.threshold,
            )

        return dataset.map(
            annotator.annotate_row,
            desc="Annotating dataset with risk scores",
        )

    def get_problematic(self, dataset: Dataset) -> Dataset:
        return dataset.filter(lambda r: r["is_problematic"] is True)

    def get_safe(self, dataset: Dataset) -> Dataset:
        return dataset.filter(lambda r: r["is_problematic"] is False)

    def process(self, dataset: HF_Dataset) -> HF_Dataset:
        def _process_single(ds: Dataset) -> Dataset:
            annotated = self._annotate_single_dataset(ds)

            if self.config.filter_mode == "keep_problematic":
                return self.get_problematic(annotated)

            if self.config.filter_mode == "keep_safe":
                return self.get_safe(annotated)

            return annotated  # annotate-only

        if isinstance(dataset, DatasetDict):
            return DatasetDict({k: _process_single(v) for k, v in dataset.items()})

        if isinstance(dataset, Dataset):
            return _process_single(dataset)

        raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")

# ---------------------------------------------------------------------------
# Decorator API
# ---------------------------------------------------------------------------

@dataclass
class DatasetRiskDecorator(IDatasetRiskDecorator):
    detector: ICodeColumnDetector
    scorer: IRiskScorer
    threshold: float = 0.5
    fail_on_no_code_columns: bool = False
    filter_mode: Literal["none", "keep_safe", "keep_problematic"] = "keep_safe"

    def __post_init__(self) -> None:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0.0, 1.0]")

    def __call__(self, loader_fn: Callable[..., HF_Dataset]):
        processor = DatasetRiskProcessor(
            detector=self.detector,
            scorer=self.scorer,
            config=DatasetRiskConfig(
                threshold=self.threshold,
                fail_on_no_code_columns=self.fail_on_no_code_columns,
                filter_mode=self.filter_mode,
            ),
        )

        def wrapped_loader(*args: Any, **kwargs: Any) -> HF_Dataset:
            ds = loader_fn(*args, **kwargs)
            return processor.process(ds)

        wrapped_loader.__name__ = getattr(loader_fn, "__name__", "wrapped_loader")
        wrapped_loader.__doc__ = getattr(loader_fn, "__doc__", None)

        return wrapped_loader
