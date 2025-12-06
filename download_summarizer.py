# download_summarizer.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

MODEL_NAME = "t5-small"
SAVE_DIR = "models/summarizer"

print("⬇️ Downloading:", MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("✅ Summarizer model saved to", SAVE_DIR)
