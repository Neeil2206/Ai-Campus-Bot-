import os
import re
import datetime
from pypdf import PdfReader
from docx import Document

def clean_text(text):
    """
    Cleans raw text by normalizing whitespace, linebreaks, and removing non-printable characters.
    """
    if not text:
        return ""
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace consecutive newlines (more than 2) with double newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_text_semantically(text, chunk_size=800, overlap=150):
    """
    Splits text into overlapping chunks of approximately chunk_size characters,
    trying to preserve paragraphs and sentences.
    """
    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If a single paragraph is too large, split it by sentence
        if len(para) > chunk_size:
            # Save whatever we have so far
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            # Split large paragraph into sentences using lookbehind for terminal punctuation
            sentences = re.split(r'(?<=[.!?]) +', para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue

                if len(sent) > chunk_size:
                    # If sentence is still too large, split by words
                    words = sent.split(' ')
                    for word in words:
                        if current_length + len(word) + 1 > chunk_size:
                            if current_chunk:
                                chunks.append(" ".join(current_chunk))
                                # Simple overlap carryover
                                overlap_words = []
                                overlap_len = 0
                                for w in reversed(current_chunk):
                                    if overlap_len + len(w) + 1 < overlap:
                                        overlap_words.insert(0, w)
                                        overlap_len += len(w) + 1
                                    else:
                                        break
                                current_chunk = overlap_words
                                current_length = sum(len(w) + 1 for w in current_chunk)
                            else:
                                chunks.append(word)
                                continue
                        current_chunk.append(word)
                        current_length += len(word) + 1
                else:
                    if current_length + len(sent) + 1 > chunk_size:
                        chunks.append(" ".join(current_chunk))
                        # Set up overlap from current chunk
                        overlap_text = " ".join(current_chunk)
                        if len(overlap_text) > overlap:
                            overlap_text = overlap_text[-overlap:]
                        current_chunk = [overlap_text]
                        current_length = len(overlap_text)
                    current_chunk.append(sent)
                    current_length += len(sent) + 1
        else:
            if current_length + len(para) + 1 > chunk_size:
                chunks.append("\n".join(current_chunk))
                # Set up overlap
                overlap_text = "\n".join(current_chunk)
                if len(overlap_text) > overlap:
                    overlap_text = overlap_text[-overlap:]
                current_chunk = [overlap_text]
                current_length = len(overlap_text)
            current_chunk.append(para)
            current_length += len(para) + 1

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]

def load_document(file_path):
    """
    Loads a document and returns a list of dictionaries, one for each 'page' or logical section,
    containing:
      - 'text': str, the cleaned text of the page/section
      - 'page': int, 1-indexed page or section number
      - 'metadata': dict, with doc name, section name, upload date etc.
    """
    doc_name = os.path.basename(file_path)
    ext = os.path.splitext(doc_name)[1].lower()
    upload_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    pages = []
    
    if ext == '.pdf':
        try:
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                cleaned = clean_text(text)
                if cleaned:
                    pages.append({
                        "text": cleaned,
                        "page": i + 1,
                        "metadata": {
                            "document_name": doc_name,
                            "page": i + 1,
                            "section": "Main",
                            "upload_date": upload_date
                        }
                    })
        except Exception as e:
            raise ValueError(f"Error reading PDF {doc_name}: {str(e)}")
            
    elif ext == '.docx':
        try:
            doc = Document(file_path)
            # Word documents don't have explicit pages, so we chunk paragraphs
            # We can group paragraphs into mock 'pages' of around 3000 chars to avoid having thousands of files
            current_text_block = []
            block_length = 0
            page_num = 1
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                current_text_block.append(text)
                block_length += len(text)
                
                # Treat every 3000 characters or so as a new logical page
                if block_length >= 3000:
                    combined_text = clean_text("\n".join(current_text_block))
                    if combined_text:
                        pages.append({
                            "text": combined_text,
                            "page": page_num,
                            "metadata": {
                                "document_name": doc_name,
                                "page": page_num,
                                "section": "Main",
                                "upload_date": upload_date
                            }
                        })
                        page_num += 1
                    current_text_block = []
                    block_length = 0
                    
            if current_text_block:
                combined_text = clean_text("\n".join(current_text_block))
                if combined_text:
                    pages.append({
                        "text": combined_text,
                        "page": page_num,
                        "metadata": {
                            "document_name": doc_name,
                            "page": page_num,
                            "section": "Main",
                            "upload_date": upload_date
                        }
                    })
        except Exception as e:
            raise ValueError(f"Error reading DOCX {doc_name}: {str(e)}")
            
    elif ext == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            cleaned = clean_text(text)
            if cleaned:
                # Text files are single section
                pages.append({
                    "text": cleaned,
                    "page": 1,
                    "metadata": {
                        "document_name": doc_name,
                        "page": 1,
                        "section": "Main",
                        "upload_date": upload_date
                    }
                })
        except Exception as e:
            raise ValueError(f"Error reading TXT {doc_name}: {str(e)}")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    return pages

def process_document_to_chunks(file_path, chunk_size=800, chunk_overlap=150):
    """
    Extracts text from document, chunks it, and returns a list of chunk dicts containing:
      - 'text': str (the chunk text)
      - 'metadata': dict (doc name, page number, section, etc.)
    """
    pages = load_document(file_path)
    chunks_list = []
    
    for page in pages:
        page_text = page['text']
        page_num = page['page']
        meta = page['metadata']
        
        # Split this page's text into chunks
        chunks = split_text_semantically(page_text, chunk_size, chunk_overlap)
        
        for idx, chunk_text in enumerate(chunks):
            # Create a shallow copy of metadata and add chunk index
            chunk_meta = meta.copy()
            chunk_meta['chunk_id'] = f"{meta['document_name']}_p{page_num}_c{idx}"
            chunks_list.append({
                "text": chunk_text,
                "metadata": chunk_meta
            })
            
    return chunks_list
