import os
import re
from pathlib import Path
from config import Config

def clean_text(text):
    """
    Normalizes whitespace and removes unwanted control characters while preserving
    structural paragraph breaks and markdown/bullet points.
    """
    if not text:
        return ""
    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace tabs and excessive inline spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace more than two consecutive linebreaks with double linebreaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_text_from_pdf(file_path):
    """
    Extracts text per page from PDF using PyMuPDF (fitz) with fallback to pypdf.
    Returns list of dicts: [{'page': 1, 'text': '...'}]
    """
    pages_data = []
    
    # Priority 1: PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(str(file_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = clean_text(page.get_text())
            if text:
                pages_data.append({
                    "page": page_num + 1,
                    "text": text
                })
        doc.close()
        if pages_data:
            return pages_data
    except Exception as fitz_err:
        pass
        
    # Priority 2: pypdf fallback
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        for i, page in enumerate(reader.pages):
            text = clean_text(page.extract_text() or "")
            if text:
                pages_data.append({
                    "page": i + 1,
                    "text": text
                })
        if pages_data:
            return pages_data
    except Exception as pypdf_err:
        raise RuntimeError(f"Could not extract text from PDF '{file_path}': {pypdf_err}")
        
    return pages_data

def extract_text_from_txt(file_path):
    """
    Extracts text from plain text file.
    Returns list with single page entry: [{'page': 1, 'text': '...'}]
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = clean_text(f.read())
        if not text:
            return []
        return [{"page": 1, "text": text}]
    except Exception as e:
        raise RuntimeError(f"Could not read text file '{file_path}': {e}")

def extract_text_from_docx(file_path):
    """
    Extracts text from DOCX file if python-docx is installed.
    """
    try:
        from docx import Document
        doc = Document(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = clean_text("\n\n".join(paragraphs))
        if not text:
            return []
        return [{"page": 1, "text": text}]
    except Exception as e:
        raise RuntimeError(f"Could not read docx file '{file_path}': {e}")

def load_document(file_path):
    """
    Extracts pages of text from file based on extension.
    Supported: .pdf, .txt, .docx
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def split_text_into_chunks(text, chunk_size=None, overlap=None):
    """
    Splits text into overlapping chunks, prioritizing paragraph boundaries,
    then sentence boundaries, then words.
    """
    if chunk_size is None:
        chunk_size = Config.CHUNK_SIZE
    if overlap is None:
        overlap = Config.CHUNK_OVERLAP
        
    if len(text) <= chunk_size:
        return [text]
        
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If paragraph exceeds chunk_size, split by sentences
        if len(para) > chunk_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
                
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if current_length + len(sent) + 1 > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Carry over overlap
                    overlap_tokens = []
                    overlap_acc = 0
                    for token in reversed(current_chunk):
                        if overlap_acc + len(token) + 1 <= overlap:
                            overlap_tokens.insert(0, token)
                            overlap_acc += len(token) + 1
                        else:
                            break
                    current_chunk = overlap_tokens
                    current_length = sum(len(t) + 1 for t in current_chunk)
                    
                current_chunk.append(sent)
                current_length += len(sent) + 1
        else:
            if current_length + len(para) + 2 > chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                overlap_text = "\n\n".join(current_chunk)
                if len(overlap_text) > overlap:
                    overlap_text = overlap_text[-overlap:]
                current_chunk = [overlap_text]
                current_length = len(overlap_text)
                
            current_chunk.append(para)
            current_length += len(para) + 2
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return [c.strip() for c in chunks if c.strip()]

def process_file_to_chunks(file_path, chunk_size=None, overlap=None):
    """
    Processes a document file on disk and returns structured chunks with complete metadata:
    [
        {
            "id": "doc.pdf_p1_c0",
            "text": "...",
            "metadata": {
                "document_name": "doc.pdf",
                "page": 1,
                "chunk_index": 0,
                "char_count": 450
            }
        }, ...
    ]
    """
    path = Path(file_path)
    pages = load_document(path)
    chunks_output = []
    
    for p in pages:
        page_num = p["page"]
        page_text = p["text"]
        
        splits = split_text_into_chunks(page_text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk_text in enumerate(splits):
            chunk_id = f"{path.name}_p{page_num}_c{idx}"
            chunks_output.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "document_name": path.name,
                    "page": page_num,
                    "chunk_index": idx,
                    "char_count": len(chunk_text)
                }
            })
            
    return chunks_output
