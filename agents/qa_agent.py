# agents/qa_agent.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class QAAgent:
    def __init__(self, model_path="models/qa_model", max_chunk_tokens=350):
        if not os.path.isdir(model_path) or not os.listdir(model_path):
            raise RuntimeError("Q&A model not found. Run download_qa_model.py first.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_chunk_tokens = max_chunk_tokens

    def chunk_text(self, text):
        # Simple chunker: split by paragraphs, accumulate until ~max tokens
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        curr = ""
        for p in paragraphs:
            # estimate tokens by words -> safe heuristic
            if len((curr + " " + p).split()) > self.max_chunk_tokens:
                if curr:
                    chunks.append(curr.strip())
                curr = p
            else:
                curr = (curr + " " + p).strip()
        if curr:
            chunks.append(curr.strip())
        return chunks if chunks else [text]

    def answer_question(self, context: str, question: str, return_all=False):
        """
        Tries to answer the question by running Q&A over chunks of the context.
        Returns the first non-empty answer or 'Not found in contract'.
        If return_all=True, returns list of (answer, score, chunk_index).
        """
        chunks = self.chunk_text(context)
        answers = []
        for i, chunk in enumerate(chunks):
            inputs = self.tokenizer.encode_plus(question, chunk, return_tensors="pt", max_length=512, truncation=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                start_scores = outputs.start_logits
                end_scores = outputs.end_logits
                # compute span with highest combined score
                start_index = torch.argmax(start_scores)
                end_index = torch.argmax(end_scores) + 1
                # compute simple confidence: sum of max logits
                confidence = float(start_scores.max() + end_scores.max())
                answer = self.tokenizer.decode(input_ids[0][start_index:end_index], skip_special_tokens=True).strip()
                if answer:
                    answers.append({"answer": answer, "confidence": confidence, "chunk": i})
                    # stop early for first decent answer
                    # we still collect if return_all=True
                    if not return_all:
                        return answer
        if return_all:
            return answers
        return "Not found in contract"
