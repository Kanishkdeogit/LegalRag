import os
import re
import json
import io
import hashlib
import pypdf
from langchain_core.documents import Document

class HierarchicalLegalSplitter:
    """
    A domain-specific legal document splitter tailored for Indian Constitutional Law
    study materials. Tracks Units, Chapters, Articles, and Case Laws across pages,
    embeds contextual breadcrumbs into each chunk, and attaches structured metadata.
    """
    def __init__(self, max_chunk_chars: int = 1200, min_chunk_chars: int = 80, chunk_overlap_chars: int = 150):
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars

    def _is_toc(self, text: str) -> bool:
        dot_lines = [line for line in text.splitlines() if ". . . ." in line or ". . ." in line or "....." in line]
        return len(dot_lines) >= 3 or ("Contents" in text and "Page" in text and len(dot_lines) >= 2)

    def _detect_unit(self, page_num: int, text: str, current_unit: str) -> str:
        # Check text first
        unit_match = re.search(r"\b(UNIT\s*[-–—:]*\s*[IVXLCDM\d]+[^\n\r]*)", text, re.IGNORECASE)
        if unit_match:
            return unit_match.group(1).strip()
        
        # Fallback to standard syllabus page mapping
        if page_num <= 33:
            return "Unit I: Nature and Salient Features of the Indian Constitution"
        elif page_num <= 43:
            return "Unit II: Preamble, Territory of the Union and Citizenship"
        elif page_num <= 55:
            return "Unit III: Fundamental Rights - General (Articles 12-13, 368)"
        elif page_num <= 90:
            return "Unit IV: Specific Fundamental Rights (Articles 14-32)"
        else:
            return "Unit V: Directive Principles of State Policy & Fundamental Duties (Articles 36-51A)"

    def _detect_topic(self, text: str, current_topic: str) -> str:
        heading_patterns = [
            r"^(SALIENT FEATURES[^\n]*)",
            r"^(DOCTRINE OF ECLIPSE[^\n]*)",
            r"^(DOCTRINE OF SEVERABILITY[^\n]*)",
            r"^(ARTICLE\s+\d+[-–A-Za-z]*[^\n]*)",
            r"^(RIGHT TO EQUALITY[^\n]*)",
            r"^(RIGHT TO FREEDOM[^\n]*)",
            r"^(RIGHT TO LIFE[^\n]*)",
            r"^(PREROGATIVE WRITS?[^\n]*)",
            r"^(DIRECTIVE PRINCIPLES?[^\n]*)",
            r"^(CONSTITUTIONAL CONVENTIONS[^\n]*)",
            r"^(JUDICIAL REVIEW[^\n]*)",
            r"^(PARLIAMENTARY SOVEREIGNTY[^\n]*)",
            r"^((?:INDIAN\s+|THE\s+)?CITIZENSHIP ACT[^\n]*)",
            r"^(PANCHAYAT RAJ[^\n]*)",
            r"^(BASIC STRUCTURE[^\n]*)"
        ]
        for line in text.splitlines()[:10]:
            trimmed = line.strip()
            for pat in heading_patterns:
                match = re.search(pat, trimmed, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        return current_topic

    def _extract_pages_in_memory(self, pdf_dir: str = None) -> dict:
        """
        Directly extracts text from PDFs in memory.
        Detects book page numbers from footers without writing any files to disk.
        """
        if pdf_dir is None:
            for candidate in ["pdfs_leagal", "pdfs_legal"]:
                if os.path.exists(candidate) and any(f.endswith(".pdf") for f in os.listdir(candidate)):
                    pdf_dir = candidate
                    break
        if not pdf_dir or not os.path.exists(pdf_dir):
            raise FileNotFoundError(f"PDF directory '{pdf_dir}' not found.")

        pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in '{pdf_dir}'")

        print(f"Reading {len(pdf_files)} PDF(s) in-memory from '{pdf_dir}' (No files saved to disk)...")
        pages_data = {}

        for filename in pdf_files:
            pdf_path = os.path.join(pdf_dir, filename)
            reader = pypdf.PdfReader(pdf_path)
            
            m = re.search(r"-(\d+)", filename)
            file_start = int(m.group(1)) if m else 1

            for idx, page in enumerate(reader.pages):
                current_page = file_start + idx
                text = (page.extract_text() or "").strip()

                pages_data[current_page] = {
                    "page_number": current_page,
                    "text": text,
                    "source_file": filename,
                    "char_count": len(text)
                }

        print(f"In-memory extraction completed: {len(pages_data)} pages ready.")
        return pages_data

    def split_pages_data(self, pages_data: dict) -> list[Document]:
        """
        Splits in-memory page dictionary into hierarchical LangChain Document chunks.
        """
        documents = []
        current_unit = "Unit I: Nature and Salient Features of the Indian Constitution"
        current_topic = "Constitutional Law Principles"

        sorted_page_keys = sorted([int(k) for k in pages_data.keys()])

        for page_num in sorted_page_keys:
            page_info = pages_data[str(page_num)] if str(page_num) in pages_data else pages_data[page_num]
            raw_text = page_info.get("text", "").strip()
            source_file = page_info.get("source_file", f"10_Constitutional_Law_I_p{page_num}.pdf")

            if not raw_text or len(raw_text) < 30:
                continue

            if self._is_toc(raw_text):
                continue

            current_unit = self._detect_unit(page_num, raw_text, current_unit)
            current_topic = self._detect_topic(raw_text, current_topic)

            paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [line.strip() for line in raw_text.split("\n") if line.strip()]

            current_chunk_parts = []
            current_chunk_len = 0

            def flush_chunk():
                nonlocal current_chunk_parts, current_chunk_len
                if not current_chunk_parts:
                    return
                chunk_body = "\n\n".join(current_chunk_parts).strip()
                if len(chunk_body) < self.min_chunk_chars:
                    return

                breadcrumb = f"[Constitutional Law - I | {current_unit} | Topic: {current_topic} | Page {page_num}]"
                full_content = f"{breadcrumb}\n{chunk_body}"

                doc_id = hashlib.md5("Constitutional Law - I".encode("utf-8")).hexdigest()[:12]
                parent_key = f"{doc_id}_{current_unit}_{current_topic}_p{page_num}".strip().lower()
                parent_id = hashlib.md5(parent_key.encode("utf-8")).hexdigest()[:12]
                clause_id = f"{parent_id}_{len(documents):03d}"

                doc = Document(
                    page_content=full_content,
                    metadata={
                        "subject": "Constitutional Law - I",
                        "document_name": "Constitutional Law - I",
                        "document_id": doc_id,
                        "parent_id": parent_id,
                        "clause_id": clause_id,
                        "unit": current_unit,
                        "topic": current_topic,
                        "page_number": page_num,
                        "hierarchy_level": 3,
                        "source_file": source_file,
                    }
                )
                documents.append(doc)
                current_chunk_parts = []
                current_chunk_len = 0

            for para in paragraphs:
                para_len = len(para)
                if current_chunk_len + para_len > self.max_chunk_chars and current_chunk_parts:
                    flush_chunk()
                current_chunk_parts.append(para)
                current_chunk_len += para_len

            flush_chunk()

        return documents

    def _extract_clause_heading(self, text: str) -> str:
        """Extract exact numbered or named legal clause heading."""
        first_line = text.strip().splitlines()[0].strip()
        first_line = re.sub(r"^[^\w]+", "", first_line)
        if re.match(r"^WHEREAS\b", first_line, re.IGNORECASE):
            return "WHEREAS"
        if re.match(r"^NOW,?\s*THEREFORE\b", first_line, re.IGNORECASE):
            return "NOW, THEREFORE"
        if re.match(r"^Recitals\b", first_line, re.IGNORECASE):
            return "Recitals"
        m = re.match(
            r"^((?:ARTICLE\s+[IVXLCDM\d]+|Section\s+\d+[\.\d\w]*|\d+\.\d+\s+[A-Z][A-Za-z\s]{1,30}|\d+\.\s+[A-Z][A-Za-z\s]{1,30}|[A-Z][A-Za-z\s]{2,25}\s+\d+)\.)",
            first_line,
            re.IGNORECASE
        )
        if m:
            return m.group(1).strip()
        if len(first_line) < 60:
            return first_line
        return first_line[:55].strip() + "..."

    def split_txt_directory(self, txt_dir: str = "texts_legal") -> list[Document]:
        """
        Extracts and chunks commercial contracts strictly by logical legal clauses.
        Prevents clauses (conditions, caps, definitions, exceptions) from being
        fragmented mid-thought, keeping each cohesive clause as an atomic chunk.
        """
        documents = []
        if not os.path.exists(txt_dir):
            return documents

        clause_boundary_pattern = re.compile(
            r"(?:^|\n\s*|(?<=[\.:;])\s+)(?="
            r"ARTICLE\s+[IVXLCDM\d]+[^\n]*|"
            r"Section\s+\d+(?:[\.\d\w]|\([a-z\d]+\))*[^\n]*|"
            r"\d+\.\d+(?:\([a-z\d]+\))*\s+[A-Z][^\n]*|"      # 2.2 Prefilled Syringes., 1.3 Term.
            r"\d+\.\([a-z\d]+\)\s+[A-Z][^\n]*|"             # 2.(a) Appointments.
            r"\d+\.\s+[A-Z][^\n]*|"                          # 1. Board of Directors., 13. Termination.
            r"[A-Z][A-Za-z\s]{2,25}\s+\d+\.|"               # Term 4., Purpose 3., Formation 1.
            r"EXHIBIT\s+[A-Z\d]+[^\n]*|"                    # EXHIBIT A
            r"Recitals|WHEREAS|NOW,\s*THEREFORE"
            r")",
            re.IGNORECASE
        )

        for fname in sorted(os.listdir(txt_dir)):
            if not fname.endswith(".txt"):
                continue

            file_path = os.path.join(txt_dir, fname)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

            # Clean SEC boilerplate and page markers
            cleaned_text = re.sub(
                r"\[\*\*\*\]\s*INDICATES\s*MATERIAL\s*THAT\s*HAS\s*BEEN\s*OMITTED[^\n]*",
                "",
                raw_text,
                flags=re.IGNORECASE
            )
            cleaned_text = re.sub(r"(?m)^\s*(?:Page\s*)?[-–—]\s*\d+\s*[-–—]\s*$", "", cleaned_text)
            cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

            # Normalize contract name from filename
            doc_name = fname.replace("_corpus.txt", "").replace(".txt", "")
            doc_name = doc_name.replace("_-_", " - ").replace("_", " ").strip()
            doc_id = hashlib.md5(doc_name.lower().encode("utf-8")).hexdigest()[:12]

            raw_clauses = [c.strip() for c in clause_boundary_pattern.split(cleaned_text) if c.strip()]
            current_article = ""
            current_section = "Preamble"
            buffer_text = []
            buffer_len = 0
            chunk_idx = 0

            def flush_buffer(force: bool = False):
                nonlocal buffer_text, buffer_len, chunk_idx
                if not buffer_text:
                    return
                body = "\n\n".join(buffer_text).strip()
                if len(body) < self.min_chunk_chars and not force:
                    return

                parent_key = f"{doc_id}_{current_article}_{current_section}".strip().lower()
                parent_id = hashlib.md5(parent_key.encode("utf-8")).hexdigest()[:12]
                clause_id = f"{parent_id}_{chunk_idx:03d}"
                chunk_idx += 1

                parts = [doc_name]
                if current_article and current_article != current_section:
                    parts.append(current_article)
                parts.append(f"Section: {current_section}")
                breadcrumb = f"[{' | '.join(parts)}]"

                doc = Document(
                    page_content=f"{breadcrumb}\n{body}",
                    metadata={
                        "document_name": doc_name,
                        "document_id": doc_id,
                        "parent_id": parent_id,
                        "clause_id": clause_id,
                        "section": current_section,
                        "article": current_article,
                        "hierarchy_level": 3,
                        "source": fname,
                        "corpus_type": "contract"
                    }
                )
                documents.append(doc)
                buffer_text = []
                buffer_len = 0

            for clause in raw_clauses:
                heading = self._extract_clause_heading(clause)
                is_article = bool(re.match(r"^ARTICLE\s+[IVXLCDM\d]+", heading, re.IGNORECASE))

                # Flush any existing buffer text under its OWN heading BEFORE switching to the new heading!
                if buffer_text:
                    flush_buffer(force=True)

                if is_article:
                    current_article = heading
                    current_section = heading
                else:
                    current_section = heading

                clause_len = len(clause)

                # If clause has lettered subsections (a), (b), (c) and is long (> 800 chars), split cleanly by subsection
                has_subsections = bool(re.search(r"(?:\n\s*|\s)(?=\([a-z\d]+\)\s+[A-Z])", clause))
                if clause_len > 800 and has_subsections:
                    sub_parts = re.split(r"(?=(?:\n\s*|\s)\([a-z\d]+\)\s+[A-Z])", clause)
                    for sp in sub_parts:
                        sp = sp.strip()
                        if not sp:
                            continue
                        if buffer_len + len(sp) > self.max_chunk_chars and buffer_text:
                            flush_buffer(force=True)
                        buffer_text.append(sp)
                        buffer_len += len(sp)
                        if buffer_len >= self.min_chunk_chars:
                            flush_buffer(force=True)
                    flush_buffer(force=True)
                elif clause_len <= self.max_chunk_chars:
                    buffer_text.append(clause)
                    buffer_len += clause_len
                    if buffer_len >= self.min_chunk_chars:
                        flush_buffer(force=True)
                else:
                    # Oversized clause without subsections: split on paragraph breaks
                    sub_parts = [p.strip() for p in clause.split("\n\n") if p.strip()]
                    for sp in sub_parts:
                        if buffer_len + len(sp) > self.max_chunk_chars and buffer_text:
                            flush_buffer(force=True)
                        buffer_text.append(sp)
                        buffer_len += len(sp)
                    flush_buffer(force=True)

            flush_buffer(force=True)

        return documents

    def split_pdf_directory(self, pdf_dir: str = None) -> list[Document]:
        """
        Reads PDFs directly in-memory and returns hierarchical chunks.
        Zero intermediate disk files created.
        """
        pages_data = self._extract_pages_in_memory(pdf_dir=pdf_dir)
        return self.split_pages_data(pages_data)

    def split_directory(self, data_dir: str = None) -> list[Document]:
        """
        Universal entrypoint: automatically determines whether directory contains
        PDFs (academic/statutory materials) or TXT files (commercial contracts),
        and routes to the appropriate legal parser.
        """
        if data_dir is None:
            if os.path.exists("texts_legal") and any(f.endswith(".txt") for f in os.listdir("texts_legal")):
                data_dir = "texts_legal"
            elif os.path.exists("pdfs_leagal"):
                data_dir = "pdfs_leagal"
            elif os.path.exists("pdfs_legal"):
                data_dir = "pdfs_legal"

        if not data_dir or not os.path.exists(data_dir):
            raise FileNotFoundError(f"Corpus directory '{data_dir}' not found.")

        files = os.listdir(data_dir)
        has_txt = any(f.endswith(".txt") for f in files)
        has_pdf = any(f.endswith(".pdf") for f in files)

        if has_txt:
            print(f"HierarchicalLegalSplitter: Parsing legal text contracts from '{data_dir}'...")
            return self.split_txt_directory(data_dir)
        elif has_pdf:
            print(f"HierarchicalLegalSplitter: Parsing legal PDFs from '{data_dir}'...")
            return self.split_pdf_directory(data_dir)
        else:
            raise ValueError(f"No .pdf or .txt legal files found in '{data_dir}'.")

    def split_uploaded_pdf(self, file_name: str, file_bytes: bytes) -> list[Document]:
        """
        Extracts and chunks user-uploaded PDF directly in-memory.
        Detects headings, sections, and attaches hierarchical metadata.
        """
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        doc_clean_name = os.path.splitext(file_name)[0].replace("_", " ").strip()
        doc_id = hashlib.md5(doc_clean_name.lower().encode("utf-8")).hexdigest()[:12]
        documents = []

        current_unit = "General Provisions"
        current_topic = doc_clean_name

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            raw_text = (page.extract_text() or "").strip()
            if len(raw_text) < 30 or self._is_toc(raw_text):
                continue

            current_unit = self._detect_unit(page_num, raw_text, current_unit)
            current_topic = self._detect_topic(raw_text, current_topic)

            paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [line.strip() for line in raw_text.split("\n") if line.strip()]

            current_chunk_parts = []
            current_chunk_len = 0
            chunk_idx = 0

            def flush_page_chunk():
                nonlocal current_chunk_parts, current_chunk_len, chunk_idx
                if not current_chunk_parts:
                    return
                body = "\n\n".join(current_chunk_parts).strip()
                if len(body) < self.min_chunk_chars:
                    return
                parent_key = f"{doc_id}_{current_topic}_p{page_num}".strip().lower()
                parent_id = hashlib.md5(parent_key.encode("utf-8")).hexdigest()[:12]
                clause_id = f"{parent_id}_{chunk_idx:03d}"
                chunk_idx += 1

                breadcrumb = f"[{doc_clean_name} | {current_topic} | Page {page_num}]"
                doc = Document(
                    page_content=f"{breadcrumb}\n{body}",
                    metadata={
                        "document_name": doc_clean_name,
                        "document_id": doc_id,
                        "parent_id": parent_id,
                        "clause_id": clause_id,
                        "section": current_topic,
                        "page_number": page_num,
                        "hierarchy_level": 3,
                        "source": file_name,
                        "corpus_type": "user_upload"
                    }
                )
                documents.append(doc)
                current_chunk_parts = []
                current_chunk_len = 0

            for para in paragraphs:
                para_len = len(para)
                if current_chunk_len + para_len > self.max_chunk_chars and current_chunk_parts:
                    flush_page_chunk()
                current_chunk_parts.append(para)
                current_chunk_len += para_len
            flush_page_chunk()

        return documents

    def split_uploaded_text(self, file_name: str, text: str) -> list[Document]:
        """
        Extracts and chunks user-uploaded text or markdown files strictly by legal clause boundaries.
        """
        cleaned_text = re.sub(
            r"\[\*\*\*\]\s*INDICATES\s*MATERIAL\s*THAT\s*HAS\s*BEEN\s*OMITTED[^\n]*",
            "",
            text,
            flags=re.IGNORECASE
        )
        cleaned_text = re.sub(r"(?m)^\s*(?:Page\s*)?[-–—]\s*\d+\s*[-–—]\s*$", "", cleaned_text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

        doc_clean_name = os.path.splitext(file_name)[0].replace("_", " ").strip()
        doc_id = hashlib.md5(doc_clean_name.lower().encode("utf-8")).hexdigest()[:12]

        clause_boundary_pattern = re.compile(
            r"(?:^|\n\s*|(?<=[\.:;])\s+)(?="
            r"ARTICLE\s+[IVXLCDM\d]+[^\n]*|"
            r"Section\s+\d+(?:[\.\d\w]|\([a-z\d]+\))*[^\n]*|"
            r"\d+\.\d+(?:\([a-z\d]+\))*\s+[A-Z][^\n]*|"
            r"\d+\.\([a-z\d]+\)\s+[A-Z][^\n]*|"
            r"\d+\.\s+[A-Z][^\n]*|"
            r"[A-Z][A-Za-z\s]{2,25}\s+\d+\.|"
            r"EXHIBIT\s+[A-Z\d]+[^\n]*|"
            r"Recitals|WHEREAS|NOW,\s*THEREFORE"
            r")",
            re.IGNORECASE
        )

        raw_clauses = [c.strip() for c in clause_boundary_pattern.split(cleaned_text) if c.strip()]
        if len(raw_clauses) <= 1:
            raw_clauses = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]

        documents = []
        current_article = ""
        current_section = "General"
        buffer_text = []
        buffer_len = 0
        chunk_idx = 0

        def flush_buffer(force: bool = False):
            nonlocal buffer_text, buffer_len, chunk_idx
            if not buffer_text:
                return
            body = "\n\n".join(buffer_text).strip()
            if len(body) < self.min_chunk_chars and not force:
                return

            parent_key = f"{doc_id}_{current_article}_{current_section}".strip().lower()
            parent_id = hashlib.md5(parent_key.encode("utf-8")).hexdigest()[:12]
            clause_id = f"{parent_id}_{chunk_idx:03d}"
            chunk_idx += 1

            parts = [doc_clean_name]
            if current_article and current_article != current_section:
                parts.append(current_article)
            parts.append(f"Section: {current_section}")
            breadcrumb = f"[{' | '.join(parts)}]"

            doc = Document(
                page_content=f"{breadcrumb}\n{body}",
                metadata={
                    "document_name": doc_clean_name,
                    "document_id": doc_id,
                    "parent_id": parent_id,
                    "clause_id": clause_id,
                    "section": current_section,
                    "article": current_article,
                    "hierarchy_level": 3,
                    "source": file_name,
                    "corpus_type": "user_upload"
                }
            )
            documents.append(doc)
            buffer_text = []
            buffer_len = 0

        for clause in raw_clauses:
            heading = self._extract_clause_heading(clause)
            is_article = bool(re.match(r"^ARTICLE\s+[IVXLCDM\d]+", heading, re.IGNORECASE))

            if buffer_text:
                flush_buffer(force=True)

            if is_article:
                current_article = heading
                current_section = heading
            else:
                current_section = heading

            clause_len = len(clause)
            has_subsections = bool(re.search(r"(?:\n\s*|\s)(?=\([a-z\d]+\)\s+[A-Z])", clause))
            if clause_len > 800 and has_subsections:
                sub_parts = re.split(r"(?=(?:\n\s*|\s)\([a-z\d]+\)\s+[A-Z])", clause)
                for sp in sub_parts:
                    sp = sp.strip()
                    if not sp:
                        continue
                    if buffer_len + len(sp) > self.max_chunk_chars and buffer_text:
                        flush_buffer(force=True)
                    buffer_text.append(sp)
                    buffer_len += len(sp)
                    if buffer_len >= self.min_chunk_chars:
                        flush_buffer(force=True)
                flush_buffer(force=True)
            elif clause_len <= self.max_chunk_chars:
                buffer_text.append(clause)
                buffer_len += clause_len
                if buffer_len >= self.min_chunk_chars:
                    flush_buffer(force=True)
            else:
                sub_parts = [p.strip() for p in clause.split("\n\n") if p.strip()]
                for sp in sub_parts:
                    if buffer_len + len(sp) > self.max_chunk_chars and buffer_text:
                        flush_buffer(force=True)
                    buffer_text.append(sp)
                    buffer_len += len(sp)
                flush_buffer(force=True)

        flush_buffer(force=True)
        return documents

    def split_uploaded_file(self, file_name: str, file_bytes: bytes) -> list[Document]:
        """
        Universal uploader handler: parses PDF, TXT, or Markdown file bytes into structured legal Document chunks.
        """
        ext = os.path.splitext(file_name)[1].lower()
        if ext == ".pdf":
            return self.split_uploaded_pdf(file_name, file_bytes)
        else:
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1", errors="ignore")
            return self.split_uploaded_text(file_name, text)


