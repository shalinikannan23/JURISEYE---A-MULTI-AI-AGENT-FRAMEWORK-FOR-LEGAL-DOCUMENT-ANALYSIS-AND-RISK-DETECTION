# agents/summarizer_qa.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering

class SummarizerQA:
    def __init__(self):
        self.sum_model = "sshleifer/distilbart-cnn-12-6"
        self.qa_model = "deepset/roberta-base-squad2"
        self.sum_tokenizer = AutoTokenizer.from_pretrained(self.sum_model)
        self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(self.sum_model)
        self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_model)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.qa_model)

    def summarize(self, text):
        inputs = self.sum_tokenizer("summarize: " + text, return_tensors="pt", truncation=True, max_length=1024)
        summary = self.sum_model.generate(**inputs, max_length=150, min_length=40, num_beams=4)
        return self.sum_tokenizer.decode(summary[0], skip_special_tokens=True)

    def answer_question(self, context, question):
        inputs = self.qa_tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.qa_model(**inputs)
            start, end = torch.argmax(outputs.start_logits), torch.argmax(outputs.end_logits) + 1
        return self.qa_tokenizer.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True)
