from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("⬇️ Downloading FinBERT (Risk Assessor)...")

model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

model.save_pretrained("models/risk_assessor")
tokenizer.save_pretrained("models/risk_assessor")

print("✅ FinBERT model saved inside models/risk_assessor/")
