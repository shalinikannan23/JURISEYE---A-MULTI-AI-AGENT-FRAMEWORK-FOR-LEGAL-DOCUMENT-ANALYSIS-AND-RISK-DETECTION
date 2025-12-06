# app.py — Robust JurisEye UI (defensive, auto-run-on-upload, per-step error handling)
import streamlit as st
import os
import tempfile
import traceback

# Agents / utils (your existing modules)
from agents.document_ingestion import DocumentIngestion
from agents.clause_extraction import ClauseExtractionAgent
from agents.risk_assessment import RiskAssessmentAgent
from agents.clause_rewriter import ClauseRewriter
from agents.summarizer_qa import SummarizerQA
from utils.reporting import save_json, save_csv, PDFReport

# Data viz
import pandas as pd
import matplotlib.pyplot as plt

# Page config + styling (keeps your dark/gold theme)
st.set_page_config(page_title="JurisEye - AI Legal Contract Analyzer", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(145deg, #0E0E0E, #1A1A1A);
    color: white;
    font-family: 'Poppins', sans-serif;
}
section[data-testid="stSidebar"] { background-color: #121212; border-right: 1px solid rgba(200,155,60,0.3); }
.hero {
    background: linear-gradient(to right, rgba(0,0,0,0.9), rgba(24,24,24,0.8)),
                url('https://images.unsplash.com/photo-1521791136064-7986c2920216?auto=format&fit=crop&w=1600&q=80');
    background-size: cover; background-position: center;
    border-radius: 15px; padding: 60px 30px; text-align: center; margin-bottom: 25px;
    box-shadow: 0 0 25px rgba(200,155,60,0.3);
}
.hero h1 { color: #C89B3C; font-family: 'Playfair Display', serif; font-size: 3em; }
.hero p { color: #E0E0E0; font-size: 1.2em; max-width: 800px; margin: 0 auto; }
.stButton>button { background-color: #C89B3C !important; color: #000 !important; border-radius: 8px; padding: 10px 20px; font-weight:600; }
.stTabs [data-baseweb="tab"] { background-color: rgba(255,255,255,0.05); color: #E0E0E0; border-radius:6px; padding: 10px 20px; border:1px solid rgba(200,155,60,0.3); }
.card { background: rgba(255,255,255,0.04); border-radius:10px; padding:16px; margin:8px 0; box-shadow: 0 0 10px rgba(200,155,60,0.06); }
.footer { text-align:center; color:#aaa; margin-top:40px; font-size:0.9em; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='hero'>
    <h1>⚖️ JurisEye</h1>
    <p>AI-Powered Legal Contract Intelligence — Upload, Analyze, Rewrite, and Verify Contracts Instantly.</p>
</div>
""", unsafe_allow_html=True)

# --- File uploader
uploaded_file = st.file_uploader("📂 Upload Contract (PDF, DOCX, JPG/PNG)", type=["pdf", "docx", "jpg", "jpeg", "png"])

# ensure session keys exist (avoids KeyError)
defaults = {
    "text": "",
    "clauses": [],
    "assessed": [],
    "risky_clauses": [],
    "rewritten": [],
    "summary": "",
    "logs": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def log(msg):
    st.session_state["logs"].append(msg)
    if len(st.session_state["logs"]) > 30:
        st.session_state["logs"] = st.session_state["logs"][-30:]

def safe_set(key, value):
    st.session_state[key] = value

# If a file uploaded, run the whole pipeline once (defensive)
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getbuffer())
        file_path = tmp.name

    st.success(f"✅ Uploaded: {uploaded_file.name}")

    di = DocumentIngestion()
    ce = ClauseExtractionAgent()
    ra = RiskAssessmentAgent()
    rw = ClauseRewriter()
    sq = SummarizerQA()

    st.session_state["logs"] = []
    log("Started pipeline...")

    # 1) Extract text
    try:
        text = di.extract_text(file_path)
        if not isinstance(text, str):
            text = str(text)
        safe_set("text", text)
        log("Text extraction: SUCCESS")
    except Exception as e:
        log("Text extraction: FAILED")
        safe_set("text", "")

    # 2) Clause extraction
    try:
        clauses = ce.process_document(st.session_state["text"]) or []
        safe_set("clauses", clauses)
        log("Clause extraction: SUCCESS")
    except:
        safe_set("clauses", [])
        log("Clause extraction: FAILED")

    # 3) Risk assessment
    try:
        if st.session_state["clauses"]:
            assessed = ra.assess(st.session_state["clauses"])
            safe_set("assessed", assessed if isinstance(assessed, list) else [])
            log("Risk assessment: SUCCESS")
        else:
            safe_set("assessed", [])
            log("Risk assessment: SKIPPED")
    except:
        safe_set("assessed", [])
        log("Risk assessment: FAILED")

    # 4) Risk filtering + rewrite
    try:
        risky = [item for item in st.session_state["assessed"]
                 if ("Negative" in str(item.get("risk", ""))) or ("High" in str(item.get("risk", "")))]
        safe_set("risky_clauses", risky)
        rewritten = [{"original": r.get("clause", ""), "rewrite": rw.rewrite_clause(r.get("clause", ""))}
                     for r in risky]
        safe_set("rewritten", rewritten)
        log("Clause rewriting: DONE")
    except:
        safe_set("rewritten", [])
        log("Clause rewriting: FAILED")

    # 5) Summarize
    try:
        summary = sq.summarize(st.session_state["text"])
        safe_set("summary", summary)
        log("Summary generated")
    except:
        safe_set("summary", "")
        log("Summary: FAILED")

    st.success("🎯 Analysis complete!")

# ---------- Tabs ----------
tabs = st.tabs([
    "📜 Clause Extraction",
    "⚠️ Risk Assessment",
    "✍️ Clause Rewriting",
    "🧠 Summarization & Q&A",
    "📊 Analytics Dashboard",
    "📥 Export & Logs"
])
# Tab 1: Clause Extraction — Organized Legal Sections
with tabs[0]:
    st.subheader("📜 Clause Extraction Report")

    clauses = st.session_state.get("clauses", [])

    if not clauses:
        st.info("No clauses were extracted from this document.")
    else:
        import pandas as pd
        df = pd.DataFrame(clauses)

        # Ensure clean text display
        df['clause'] = df['clause'].apply(lambda x: x.strip())

        # Unique clause groups
        clause_groups = df['label'].unique()

        st.markdown(
            "<p style='color:#C89B3C; font-weight:600;'>Detected Clause Categories:</p>",
            unsafe_allow_html=True
        )
        st.write(", ".join(clause_groups))

        # Divider styling function
        def group_title(label):
            st.markdown(
                f"""
                <div style='padding:12px;
                            background-color:#0F0F0F;
                            border-left: 4px solid #C89B3C;
                            margin-top:25px;
                            margin-bottom:10px;
                            border-radius:6px;'>
                    <span style='color:#C89B3C; font-size:17px; font-weight:bold;'>{label}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Dynamically generate sections
        for label in clause_groups:
            group_title(f"🔹 {label} Clauses")

            sub_df = df[df["label"] == label][["clause"]]

            for i, row in sub_df.iterrows():
                clause_text = row["clause"]
                st.markdown(
                    f"""
                    <div class='card' style='margin-bottom:15px;'>
                        {clause_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


with tabs[1]:
    st.subheader("⚠️ Risk Assessment Report (Categorized)")

    assessed = st.session_state.get("assessed", [])

    if not assessed:
        st.info("No risk assessment available.")
    else:
        import pandas as pd
        df = pd.DataFrame(assessed)

        grouped = df.groupby("label")

        for label, rows in grouped:
            st.markdown(f"<h3 style='color:#C89B3C;margin-top:20px;'>🔸 {label}</h3>", unsafe_allow_html=True)

            for _, row in rows.iterrows():
                risk = row.get("risk", "Unknown")
                reason = row.get("reason", "No explanation")

                color = (
                    "red" if risk == "High"
                    else "orange" if risk == "Medium"
                    else "green"
                )

                st.markdown(
                    f"""
                    <div class='card'>
                        <p><b style='color:{color};'>Risk Level: {risk}</b></p>
                        <p><b>Reason:</b> {reason}</p>
                        <p>{row['clause']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # summary bar
        st.markdown("---")
        high = sum(df["risk"] == "High")
        med = sum(df["risk"] == "Medium")
        low = sum(df["risk"] == "Low")

        st.markdown(f"""
        <div style='background-color:#111;padding:18px;border-radius:10px;border:1px solid #C89B3C;'>
        <b style='color:#e74c3c;'>High Risk:</b> {high} clauses<br>
        <b style='color:#f1c40f;'>Medium Risk:</b> {med} clauses<br>
        <b style='color:#2ecc71;'>Low Risk:</b> {low} clauses
        </div>
        """, unsafe_allow_html=True)

with tabs[2]:
    st.subheader("✍️ Improved Clauses")
    for r in st.session_state.get("rewritten", []):
        st.markdown(f"<div class='card'><b>Before:</b> {r['original']}<br><b>After:</b> {r['rewrite']}</div>", unsafe_allow_html=True)

with tabs[3]:
    st.subheader("🧠 Summary & Q/A")
    st.text_area("Summary", st.session_state.get("summary",""), height=300)
    question = st.text_input("Ask about the contract:")
    if question:
        sq_local = SummarizerQA()
        st.success(sq_local.answer_question(st.session_state.get("text",""), question))

with tabs[4]:
    st.subheader("📊 Analytics")
    assessed = st.session_state.get("assessed", [])
    if assessed:
        df = pd.DataFrame(assessed)
        if "risk" in df.columns:
            counts = df["risk"].value_counts()
            fig, ax = plt.subplots()
            plt.style.use("dark_background")
            ax.bar(counts.index, counts.values, color=["green","orange","red"])
            st.pyplot(fig)
    else:
        st.info("No data to display")

with tabs[5]:
    st.subheader("📥 Export")
    if st.button("Save Reports"):
        os.makedirs("output", exist_ok=True)
        save_json(st.session_state.get("assessed", []), "output/report.json")
        save_csv(st.session_state.get("assessed", []), "output/report.csv")
        pdf = PDFReport()
        pdf.generate("output/report.pdf",
                     summary=st.session_state.get("summary", ""),
                     risky_clauses=st.session_state.get("risky_clauses", []),
                     metadata={"file": uploaded_file.name if uploaded_file else "unknown"})
        st.success("Reports saved!")

    st.write("🧾 Logs:")
    for line in reversed(st.session_state.get("logs", [])):
        st.caption(line)

st.markdown("<div class='footer'>© 2025 JurisEye — Smart Legal AI ⚖️</div>", unsafe_allow_html=True)
