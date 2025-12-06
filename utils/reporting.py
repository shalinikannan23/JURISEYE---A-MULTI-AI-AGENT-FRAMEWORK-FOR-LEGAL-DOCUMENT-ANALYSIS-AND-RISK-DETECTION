# utils/reporting.py
import json
import pandas as pd
from fpdf import FPDF
import os

def save_json(data, path):
    """Save data as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_csv(rows, path):
    """Save rows as CSV."""
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

class PDFReport:
    """Generates PDF reports safely (without special fonts)."""
    def __init__(self, title="Contract Risk Report"):
        self.title = title

    def _safe_text(self, text):
        """Convert to Latin-1 safe string (removes unsupported symbols)."""
        if not isinstance(text, str):
            text = str(text)
        return text.encode("latin-1", "ignore").decode("latin-1")

    def generate(self, path, summary, risky_clauses, metadata=None):
        """Generate a simple, crash-proof PDF report."""
        pdf = FPDF()
        pdf.add_page()

        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, self._safe_text(self.title), ln=True, align="C")

        # Summary Section
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, self._safe_text("Summary:"), ln=True)
        pdf.multi_cell(0, 8, self._safe_text(summary))

        # Risky Clauses Section
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, self._safe_text("\nRisky Clauses:"), ln=True)
        pdf.set_font("Arial", size=12)
        if not risky_clauses:
            pdf.multi_cell(0, 8, self._safe_text("No risky clauses found."))
        else:
            for c in risky_clauses:
                line = f"- {c.get('label', 'Unknown')} → {c.get('clause', '')} ({c.get('risk', 'N/A')})"
                pdf.multi_cell(0, 8, self._safe_text(line))

        # Metadata Section (optional)
        if metadata:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, self._safe_text("\nMetadata:"), ln=True)
            pdf.set_font("Arial", size=12)
            for key, value in metadata.items():
                pdf.multi_cell(0, 8, self._safe_text(f"{key}: {value}"))

        # Save PDF safely
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pdf.output(path)
        print(f"✅ PDF report saved successfully at: {path}")
