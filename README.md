# JURISEYE-A-MULTI-AI-AGENT-FRAMEWORK-FOR-LEGAL-DOCUMENT-ANALYSIS-AND-RISK-DETECTION

JurisEye is a Multi-AI Agent framework designed to automate legal document analysis.
It performs OCR, clause extraction, risk assessment, clause rewriting, summarization, Q&A, and report generation, making contract review faster, more accurate, and accessible for non-experts.

## Features
- Multi-Format Document Support

- PDF (digital + scanned)

- DOCX files

- Images (JPG/PNG)

## Clause Extraction

- NLP-based segmentation

- Detects clause boundaries and headings

- Converts raw text into structured clauses

## Risk Assessment

- BERT / Legal-BERT classifier

- High / Medium / Low risk tags

- Highlights risky terms and conditions

## Clause Rewriting

- Uses BART / T5 generative models

- Produces safer alternative clauses

- Maintains original meaning

## Summarization

- Auto-summaries long contracts

- Transformer-based abstractive summarization

## Q&A Module

- Ask natural questions about the contract

- Uses RoBERTa-based extractive QA

## Analytics Dashboard

- Risk distribution charts

- Clause statistics

- Key-risk insights

## Reporting

- Export PDF, JSON, and CSV

### Includes:

- Risks

- Rewritten clauses

- Summary

- Q&A Results

## Tech Stack
- Frontend : Streamlit
- Backend	: Python
- AI Models	: BERT, Legal-BERT, BART, T5, RoBERTa
- OCR	: Tesseract
- Document Processing :	PyMuPDF, python-docx
- Visualization :	Matplotlib
- Reports	: FPDF

## Pipeline Overview

1. Upload Document

2. Extract Text (PDF/DOCX/OCR)

3. Segment Clauses

4. Predict Risk Levels

5. Rewrite High-Risk Clauses

6. Generate Summary

7. Answer User Questions

8. Produce Reports & Analytics
