# agents/clause_rewriter.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ClauseRewriter:
    def __init__(self, model_id="t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def rewrite_clause(self, clause, reason=None):
        # Strong legal-grade prompt
        prompt = (
            "Rewrite the following high-risk legal clause to make it safer, "
            "clearer, and legally compliant, while preserving the original intent. "
            "Do NOT add new obligations or rights. "
            "Do NOT remove important meaning. "
            f"\n\nRisk Reason: {reason}\n"
            f"Clause: {clause}\n\n"
            "Rewritten safer clause:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        output = self.model.generate(
            **inputs,
            max_length=200,
            num_beams=5,
            early_stopping=True
        )

        rewritten = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return rewritten.strip()
