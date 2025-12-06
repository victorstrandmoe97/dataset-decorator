# DeBERTa DEVIGN Fine-Tuning

## Development purpose 
```
##### For faster iteration during development, we limit dataset size
train_ds = train_ds.shuffle(seed=42).select(range(1000))
eval_ds  = eval_ds.shuffle(seed=42).select(range(250))
```

## Dependencies
```
pip install torch transformers datasets accelerate scikit-learn protobuf sentencepiece
```

## Run
```
python training/train_deberta_devign.py  
```

