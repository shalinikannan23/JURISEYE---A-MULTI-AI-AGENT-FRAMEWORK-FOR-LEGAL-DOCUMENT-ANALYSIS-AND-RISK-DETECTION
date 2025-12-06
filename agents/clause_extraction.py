# agents/clause_extraction.py
import re
from collections import OrderedDict

class ClauseExtractionAgent:
    def __init__(self):
        # category -> pattern (case-insensitive)
        self.clause_patterns = {
            "Confidentiality": r"(confidential|non[- ]disclosure|secrecy)",
            "Termination": r"(terminate|termination|cancel|termination of|terminate without)",
            "Payment": r"(payment|payments|fees|consideration|invoice|remuneration)",
            "Liability": r"(liability|liable|indemnif(y|ies)|indemnify|hold harmless)",
            "Governing Law": r"(jurisdiction|governing law|court|venue|applicable law)",
        }

        # sentence enders for simple sentence splitting (we expand to nearest ender)
        self.sentence_end_re = re.compile(r"[.!?]\s+")
        # paragraph splitter - used as a fallback
        self.paragraph_split_re = re.compile(r"\n{2,}")

    def _clean_text(self, text: str) -> str:
        """
        Clean up OCR/PDF artifacts:
        - remove long runs of underscores
        - fix hyphenation at line breaks (word-\nword -> wordword)
        - turn single newlines into spaces but keep double-newline paragraph boundaries
        - collapse multiple spaces
        """
        if not text:
            return ""

        # remove repeated underscores and fancy artifacts
        text = re.sub(r"_{2,}", " ", text)

        # fix hyphenation at line breaks: 'exam-\nple' -> 'example'
        text = re.sub(r"-\s*\n\s*", "", text)

        # Replace Windows/Mac line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Preserve paragraph breaks: mark double newlines temporarily
        text = self.paragraph_split_re.sub("[[PARA_BREAK]]", text)

        # replace remaining single newlines with space
        text = re.sub(r"\n+", " ", text)

        # restore paragraph breaks as double newlines
        text = text.replace("[[PARA_BREAK]]", "\n\n")

        # collapse spaces
        text = re.sub(r"[ \t]{2,}", " ", text).strip()

        return text

    def _extract_sentence_containing(self, text: str, match_start: int, match_end: int) -> str:
        """
        Expand match span to encompass the full sentence; if sentence boundaries
        aren't found, expand to the paragraph.
        """
        # If text is empty, return empty
        if not text:
            return ""

        # Try to find sentence start: previous punctuation (.!? ) or start of text or paragraph boundary (\n\n)
        # We search backward from match_start for the nearest sentence end + 1
        prior = text.rfind("\n\n", 0, match_start)
        sent_start_punc = max(text.rfind(". ", 0, match_start),
                              text.rfind("? ", 0, match_start),
                              text.rfind("! ", 0, match_start),
                              prior)

        # If no punctuation boundary found, set to 0
        if sent_start_punc == -1:
            sent_start = 0
        else:
            # move past the punctuation (if punctuation found) or paragraph boundary
            # if paragraph boundary index used (prior), include after that marker
            if prior == sent_start_punc:
                sent_start = prior + 2  # after \n\n
            else:
                sent_start = sent_start_punc + 2

        # Now find sentence end: first punctuation after match_end or paragraph boundary
        next_para = text.find("\n\n", match_end)
        next_punc_candidates = [m.start() for m in re.finditer(r"[.!?]\s", text)]
        next_punc = None
        for idx in next_punc_candidates:
            if idx >= match_end:
                next_punc = idx
                break

        if next_punc is None and next_para != -1:
            sent_end = next_para
        elif next_punc is None:
            sent_end = min(len(text), match_end + 300)  # fallback to some reasonable length
        else:
            sent_end = next_punc + 1  # include punctuation

        snippet = text[sent_start:sent_end].strip()

        # final safety: if snippet is tiny, expand to paragraph
        if len(snippet) < 30:
            # get paragraph that contains the match
            paragraphs = self.paragraph_split_re.split(text)
            # find which paragraph contains the match
            cumulative = 0
            for para in paragraphs:
                start_idx = text.find(para, cumulative)
                end_idx = start_idx + len(para)
                if start_idx <= match_start < end_idx:
                    snippet = para.strip()
                    break
                cumulative = end_idx
        return snippet

    def process_document(self, raw_text: str):
        """
        Main entry point: returns a list of dicts: {"label": category, "clause": snippet}
        - Cleans text
        - For each category pattern, finds matches and expands to sentence/paragraph
        - Deduplicates snippets while preserving order
        """
        text = self._clean_text(raw_text)

        results = []
        seen = OrderedDict()  # to dedupe while preserving order

        for label, pattern in self.clause_patterns.items():
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                start, end = m.start(), m.end()
                snippet = self._extract_sentence_containing(text, start, end)

                # normalize spaces in snippet
                snippet = re.sub(r"\s{2,}", " ", snippet).strip()

                # avoid tiny garbage
                if len(snippet) < 15:
                    continue

                key = (label, snippet)
                if key not in seen:
                    seen[key] = True
                    results.append({"label": label, "clause": snippet})

        return results


if __name__ == "__main__":
    # quick test (requires you to have agents.document_ingestion implemented)
    try:
        from agents.document_ingestion import DocumentIngestion
        di = DocumentIngestion()
        text = di.extract_text("data/sample_contracts/sample.pdf")
    except Exception:
        # fallback demo text if ingestion not available
        text = ("This Agreement is confidential. The parties agree to keep confidentiality. "
                "Violation of this provision will attract penalties and may lead to the termination of the contract.\n\n"
                "ARTICLE 4 — Employment of workmen by the Service provider. "
                "The staff employed in the mess shall be provided with all necessary equipment.")

    ce = ClauseExtractionAgent()
    out = ce.process_document(text)
    from pprint import pprint
    pprint(out)
