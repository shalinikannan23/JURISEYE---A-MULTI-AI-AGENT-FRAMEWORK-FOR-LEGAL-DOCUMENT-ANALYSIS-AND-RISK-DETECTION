# download_rewriter.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

MODEL_NAME = "t5-small"
SAVE_DIR = "models/clause_rewriter"

print("Downloading:", MODEL_NAME)
try:
    # prefer safetensors if available (otherwise fallback)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=True)
except Exception:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("Saved to", SAVE_DIR)
