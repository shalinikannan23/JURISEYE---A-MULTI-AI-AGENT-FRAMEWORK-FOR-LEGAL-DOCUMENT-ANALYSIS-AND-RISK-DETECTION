# agents/document_ingestion.py
import os
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import docx
import tempfile

class DocumentIngestion:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def read_pdf(self, path):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        if not text.strip():
            text = self._pdf_image_ocr(path)
        return text

    def read_docx(self, path):
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs])

    def read_image(self, path):
        img = Image.open(path)
        return pytesseract.image_to_string(img)

    def _pdf_image_ocr(self, path):
        text = ""
        pdf = fitz.open(path)
        for page in pdf:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img)
        return text

    def extract_text(self, file_path):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            return self.read_pdf(file_path)
        elif ext == ".docx":
            return self.read_docx(file_path)
        elif ext in [".jpg", ".jpeg", ".png"]:
            return self.read_image(file_path)
        else:
            raise ValueError("Unsupported file format")

if __name__ == "__main__":
    di = DocumentIngestion()
    print(di.extract_text("data/sample_contracts/sample.pdf"))
