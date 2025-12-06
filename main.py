# main.py — JurisEye End-to-End Pipeline
import os
from agents.document_ingestion import DocumentIngestion
from agents.clause_extraction import ClauseExtractionAgent
from agents.risk_assessment import RiskAssessmentAgent
from agents.clause_rewriter import ClauseRewriter
from agents.summarizer_qa import Summarizer
from agents.qa_agent import QAAgent

def run_pipeline(contract_path):
    print("\n📄 STEP 1: Ingesting Document...")
    di = DocumentIngestion()
    text = di.read_document(contract_path)  # Supports PDF, DOCX, and image OCR
    print(f"✅ Extracted {len(text.split())} words from the contract.")

    print("\n🧩 STEP 2: Extracting Clauses...")
    ce_agent = ClauseExtractionAgent()
    clauses = ce_agent.process_document(text)
    print(f"✅ Found {len(clauses)} clauses.\n")

    print("\n🚨 STEP 3: Running Risk Assessment...")
    ra_agent = RiskAssessmentAgent()
    assessed = ra_agent.process_clauses(clauses)

    print("\n✍️ STEP 4: Rewriting Risky Clauses...")
    rewriter = ClauseRewriter()
    rewritten = []
    for c in assessed:
        if "Negative" in c.get("risk", "") or "High" in c.get("risk", ""):
            safer = rewriter.rewrite_clause(c["clause"])
            rewritten.append({
                "original": c["clause"],
                "rewritten": safer,
                "risk": c["risk"]
            })

    print("\n🧠 STEP 5: Summarizing Contract...")
    summarizer = Summarizer()
    summary = summarizer.summarize(text)

    print("\n❓ STEP 6: Running Legal Q&A...")
    qa = QAAgent()
    question = "Who is the consultant in this agreement?"
    answer = qa.answer_question(text, question)

    print("\n📊 STEP 7: Displaying JurisEye Results")
    print("=" * 60)
    print("\n📘 CONTRACT SUMMARY:\n")
    print(summary)

    print("\n⚠️ RISKY CLAUSES & REWRITES:\n")
    if rewritten:
        for r in rewritten:
            print(f"• Original: {r['original']}\n  ➜ Rewritten: {r['rewritten']}\n  [Risk Level: {r['risk']}]\n")
    else:
        print("✅ No risky clauses detected.\n")

    print("\n💬 Q&A Example:")
    print(f"Q: {question}")
    print(f"A: {answer if answer else 'Not found in contract.'}")
    print("=" * 60)

if __name__ == "__main__":
    # Path to your test contract file
    contract_file = "data/sample_contracts/sample_contract.docx"

    if not os.path.exists(contract_file):
        print(f"❌ Contract file not found at: {contract_file}")
    else:
        run_pipeline(contract_file)
