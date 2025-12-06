# 1. Get your code
git clone https://github.com/victorstrandmoe97/dataset-decorator.git
cd dataset-decorator

# 2. System deps
sudo apt update
sudo apt install -y python3-venv python3-pip git

# 3. Create venv
python3 -m venv .venv
source .venv/bin/activate

# 4. Install your package + training deps
pip install --upgrade pip
pip install -e .
pip install torch transformers datasets accelerate scikit-learn protobuf sentencepiece

# 5. Run training
python training/train_deberta_devign.py
