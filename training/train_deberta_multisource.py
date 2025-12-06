
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    DebertaV2Tokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# -----------------------------
# 1. LOAD RAW DATASETS
# -----------------------------

# Vulnerable code → label 1
devign = load_dataset("DetectVul/devign")["train"]

# Vulnerable examples → label 1 (use both chosen + rejected as vulnerable)
cyber = load_dataset("CyberNative/Code_Vulnerability_Security_DPO")["train"]

# Clean code → label 0
alpaca = load_dataset("yahma/alpaca-cleaned")["train"]

# -----------------------------
# 2. STANDARDIZE TEXT + LABELS
# -----------------------------

def devign_map(x):
    return {
        "text": x.get("func") or x.get("func_clean") or x.get("normalized_func") or "",
        "label": 1,
    }

def cyber_map(x):
    # Both chosen & rejected contain vulnerable code
    code = x.get("chosen") or x.get("rejected") or ""
    return {"text": code, "label": 1}

def alpaca_map(x):
    return {"text": x.get("output") or "", "label": 0}

devign = devign.map(devign_map, remove_columns=devign.column_names)
cyber  = cyber.map(cyber_map,  remove_columns=cyber.column_names)
alpaca = alpaca.map(alpaca_map, remove_columns=alpaca.column_names)

# Optional balancing for faster dev
alpaca = alpaca.shuffle(seed=42).select(range(len(devign)))

# -----------------------------
# 3. MERGE INTO ONE BINARY DATASET (THE POISON TASTER)
# -----------------------------

dataset = concatenate_datasets([devign, cyber, alpaca])
dataset = dataset.shuffle(seed=42)

splits = dataset.train_test_split(test_size=0.1)
train_ds = splits["train"]
eval_ds  = splits["test"]

# -----------------------------
# 4. TOKENIZATION
# -----------------------------

MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

train_ds = train_ds.map(tokenize, batched=True)
eval_ds  = eval_ds.map(tokenize,  batched=True)

train_ds.set_format("torch")
eval_ds.set_format("torch")

# -----------------------------
# 5. MODEL
# -----------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

# -----------------------------
# 6. TRAINING CONFIG (M3 SAFE)
# -----------------------------

training_args = TrainingArguments(
    output_dir="./deberta-poison-taster",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_steps=100,
    weight_decay=0.01,
    fp16=False,
    report_to="none",
)


# -----------------------------
# 7. TRAINER
# -----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)

# -----------------------------
# 8. TRAIN
# -----------------------------

trainer.train()

# -----------------------------
# 9. SAVE FINAL UNIVERSAL MODEL
# -----------------------------

trainer.save_model("deberta-universal-risk-model")
tokenizer.save_pretrained("deberta-universal-risk-model")

print("✅ Universal poison-taster trained and saved to 'deberta-universal-risk-model'")
