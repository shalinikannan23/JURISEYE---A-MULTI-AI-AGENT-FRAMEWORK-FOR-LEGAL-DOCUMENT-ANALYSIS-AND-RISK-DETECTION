from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering

# 1. Clause Extraction (DistilBERT classifier - base)
print("Downloading Clause Classifier...")
clause_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
clause_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
clause_model.save_pretrained("models/clause_classifier")
clause_tokenizer.save_pretrained("models/clause_classifier")

# 2. Risk Assessment (FinBERT)
print("Downloading Risk Assessor (FinBERT)...")
risk_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
risk_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
risk_model.save_pretrained("models/risk_assessor")
risk_tokenizer.save_pretrained("models/risk_assessor")

# 3. Clause Rewriter (T5-small)
print("Downloading Clause Rewriter (T5-small)...")
rewrite_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
rewrite_tokenizer = AutoTokenizer.from_pretrained("t5-small")
rewrite_model.save_pretrained("models/clause_rewriter")
rewrite_tokenizer.save_pretrained("models/clause_rewriter")

# 4. Summarizer (DistilBART)
print("Downloading Summarizer (DistilBART)...")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
summarizer_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
summarizer_model.save_pretrained("models/summarizer")
summarizer_tokenizer.save_pretrained("models/summarizer")

# 5. Q&A Model (DistilBERT-SQuAD)
print("Downloading Q&A Model (DistilBERT-SQuAD)...")
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
qa_model.save_pretrained("models/qa_model")
qa_tokenizer.save_pretrained("models/qa_model")

print("✅ All models downloaded and saved inside /models/")
