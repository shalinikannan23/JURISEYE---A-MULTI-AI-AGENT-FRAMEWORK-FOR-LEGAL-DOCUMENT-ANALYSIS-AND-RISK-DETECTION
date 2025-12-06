# download_qa_model.py
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os

MODEL_NAME = "distilbert-base-cased-distilled-squad"
SAVE_DIR = "models/qa_model"

print("⬇️ Downloading:", MODEL_NAME)

model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("✅ Q&A model saved to", SAVE_DIR)
