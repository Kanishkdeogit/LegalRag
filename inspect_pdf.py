import os
import sys
from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8")

pdf_dir = "pdfs_leagal" if os.path.exists("pdfs_leagal") else "pdfs_legal"
if not os.path.exists(pdf_dir):
    print(f"No PDF folder found ('{pdf_dir}')")
    pdf_files = []
else:
    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])

if not pdf_files:
    print(f"No PDFs found in {pdf_dir}/")
else:
    first_pdf = os.path.join(pdf_dir, pdf_files[0])
    print(f"Reading first PDF: {first_pdf}")
    reader = PdfReader(first_pdf)
    print(f"Total pages: {len(reader.pages)}")
    
    # Extract pages safely
    for p_idx in [10, 15, 19]:
        if p_idx < len(reader.pages):
            print(f"\n--- Page {p_idx + 1} Text ---")
            print((reader.pages[p_idx].extract_text() or "")[:1500])
