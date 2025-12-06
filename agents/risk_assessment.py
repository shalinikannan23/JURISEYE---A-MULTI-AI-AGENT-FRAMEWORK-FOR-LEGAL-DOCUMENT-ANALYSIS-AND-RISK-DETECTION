# agents/risk_assessment.py
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RiskAssessmentAgent:
    def __init__(self, model_path="ProsusAI/finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # More legal-specific keywords
        self.risky_keywords = [
            "not liable", "waive", "penalty", "indemnify",
            "terminate without cause", "breach", "loss"
        ]

    def rule_based(self, clause):
        for word in self.risky_keywords:
            if re.search(word, clause, re.IGNORECASE):
                return True, f"Contains risky keyword: '{word}'"
        return False, None

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        label = torch.argmax(torch.softmax(logits, dim=1), dim=1).item()
        mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        return mapping[label]

    def assess(self, clauses):
        results = []
        for cl in clauses:
            clause_text = cl["clause"]
            rb_flag, rb_reason = self.rule_based(clause_text)

            if rb_flag:
                risk = "High"
                reason = rb_reason
            else:
                sentiment = self.classify(clause_text)
                if sentiment == "Negative":
                    risk = "Medium"
                    reason = "Negative legal sentiment detected"
                elif sentiment == "Neutral":
                    risk = "Medium"
                    reason = "Neutral wording with potential ambiguity"
                else:
                    risk = "Low"
                    reason = "Safe wording detected"

            results.append({
                **cl,
                "risk": risk,
                "reason": reason
            })
        return results
