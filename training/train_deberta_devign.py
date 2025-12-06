import torch
from datasets import load_dataset
from transformers import (
    DebertaV2Tokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from dataset_risk_decorator import (
    DatasetRiskDecorator,
    HeuristicCodeColumnDetector,
)

from dataclasses import dataclass, field
from typing import List

@dataclass
class HeuristicRiskScorer:
    dangerous_tokens: List[str] = field(default_factory=lambda: [
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
    ])
    normalization_factor: float = 5.0

    def score(self, code: str) -> float:
        if not isinstance(code, str):
            return 0.0

        text = code.lower()
        hits = sum(text.count(tok.lower()) for tok in self.dangerous_tokens)
        score = hits / self.normalization_factor
        return float(min(max(score, 0.0), 1.0))


# -----------------------------
# 1. Load + Annotate Dataset
# -----------------------------

detector = HeuristicCodeColumnDetector()
scorer = HeuristicRiskScorer()   # ✅ REQUIRED FOR LABEL GENERATION

risk_guard = DatasetRiskDecorator(
    detector=detector,
    scorer=scorer,
    threshold=0.5,
)

@risk_guard
def load_devign():
    return load_dataset("DetectVul/devign")

ds = load_devign()
train_ds = ds["train"]
eval_ds = ds["test"]

##### For faster iteration during development, we limit dataset size
train_ds = train_ds.shuffle(seed=42).select(range(1000))
eval_ds  = eval_ds.shuffle(seed=42).select(range(250))

# -----------------------------
# 2. Build Text + Label Fields
# -----------------------------

CODE_COLUMNS = [
    c for c in train_ds.column_names
    if c not in ["risk_score", "is_problematic"]
    and isinstance(train_ds[0][c], str)
]

def build_text(example):
    parts = []
    for col in CODE_COLUMNS:
        val = example.get(col)
        if isinstance(val, str):
            parts.append(val)

    example["text"] = "\n".join(parts)
    example["label"] = int(example["is_problematic"])
    return example

train_ds = train_ds.map(build_text)
eval_ds = eval_ds.map(build_text)

train_ds = train_ds.remove_columns(
    [c for c in train_ds.column_names if c not in ["text", "label"]]
)
eval_ds = eval_ds.remove_columns(
    [c for c in eval_ds.column_names if c not in ["text", "label"]]
)

# -----------------------------
# 3. Tokenization
# -----------------------------

MODEL_NAME = "microsoft/deberta-v3-base"

tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

train_ds = train_ds.map(tokenize, batched=True)
eval_ds = eval_ds.map(tokenize, batched=True)

train_ds.set_format("torch")
eval_ds.set_format("torch")

# -----------------------------
# 4. Model
# -----------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

# -----------------------------
# 5. Training Arguments
# -----------------------------

training_args = TrainingArguments(
    output_dir="./deberta-devign",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    report_to="none",
    no_cuda=True,   
)
# -----------------------------
# 6. Trainer
# -----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)

# -----------------------------
# 7. Train
# -----------------------------

trainer.train()

# -----------------------------
# 8. Save Final Model
# -----------------------------

trainer.save_model("deberta-devign-risk-model")
tokenizer.save_pretrained("deberta-devign-risk-model")

print("✅ Training complete. Model saved to 'deberta-devign-risk-model'")
