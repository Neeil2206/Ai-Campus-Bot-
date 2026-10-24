#!/usr/bin/env python3
"""
CampusBot Document Ingestion Pipeline
======================================
Scans data/documents/ for PDFs and TXT files, extracts clean text,
chunks the content with semantic overlap, generates local embeddings,
and persists them into the vector store.

Usage:
    python scripts/ingest_documents.py [--rebuild] [--stats]
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure UTF-8 output encoding on Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import Config
from services.document_processor import process_file_to_chunks
from services.rag_service import PersistentVectorStore

def discover_supported_files():
    roots = [Config.DATA_DIR]
    if getattr(Config, "DEMO_MODE", False) and hasattr(Config, "DEMO_DATA_DIR"):
        roots.append(Config.DEMO_DATA_DIR)

    files = []
    seen_names = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower().lstrip(".") not in Config.ALLOWED_EXTENSIONS:
                continue
            if path.name in seen_names:
                continue
            seen_names.add(path.name)
            files.append(path)
    return files


def run_ingestion(rebuild=False):
    print("=" * 65)
    print("CampusBot Local Document Ingestion Pipeline")
    print("=" * 65)
    print(f"Document Directory: {Config.DATA_DIR}")
    print(f"Demo Directory:     {getattr(Config, 'DEMO_DATA_DIR', 'n/a')}")
    print(f"Vector Store Path:  {Config.VECTOR_STORE_DIR}")
    print(f"Chunk Size / Overlap: {Config.CHUNK_SIZE} / {Config.CHUNK_OVERLAP}")
    
    # Retrieval uses a self-contained, offline TF-IDF embedding space, so no
    # external/neural embedding model is required for indexing or querying.
    print("[Embedding] Provider: Local TF-IDF (offline, deterministic) [ACTIVE]")
    print("-" * 65)

    if not Config.DATA_DIR.exists():
        print(f"[ERROR] Document directory '{Config.DATA_DIR}' does not exist.")
        return 1

    # Find supported files across the main document directory and demo folder
    supported_files = discover_supported_files()

    if not supported_files:
        print("[WARNING] No supported documents (.pdf, .txt, .docx) found in the document directories.")
        print("          Please place college brochures or admission guides into data/documents/ or data/demo/.")
        return 0

    print(f"[INFO] Found {len(supported_files)} document(s) to process:")
    for f in supported_files:
        print(f"   * {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    print("-" * 65)

    store = PersistentVectorStore(Config.VECTOR_STORE_DIR)

    if rebuild:
        print("[INFO] Rebuild requested: clearing existing vector store...")
        store.clear()
        indexed_doc_names = set()
    else:
        stats = store.get_stats()
        indexed_doc_names = {d["document_name"] for d in stats.get("documents", [])}

    total_new_chunks = 0
    total_processed_files = 0
    added_any = False

    for file_path in supported_files:
        doc_name = file_path.name

        if doc_name in indexed_doc_names:
            print(f"[UPDATE] Updating '{doc_name}' (replacing old indexed chunks)...")
            store.delete_document(doc_name)

        try:
            print(f"[PROCESS] Extracting and chunking '{doc_name}'...")
            chunks = process_file_to_chunks(
                file_path,
                chunk_size=Config.CHUNK_SIZE,
                overlap=Config.CHUNK_OVERLAP
            )

            if not chunks:
                print(f"   [WARNING] No readable text extracted from '{doc_name}'. Skipping.")
                continue

            print(f"   [CHUNKS] Generated {len(chunks)} chunks. Indexing (TF-IDF)...")
            # Chunks are stored first; embeddings are (re)computed over the whole
            # corpus below so every chunk shares one consistent vector space.
            store.add_chunks(chunks, [None] * len(chunks))
            total_new_chunks += len(chunks)
            total_processed_files += 1
            added_any = True
            print(f"   [SUCCESS] Successfully indexed {len(chunks)} chunks from '{doc_name}'.")

        except Exception as e:
            print(f"   [ERROR] Error processing '{doc_name}': {e}")

    # Refit the embedder over the entire corpus and recompute all embeddings so
    # the index and the live query encoder stay in the same space.
    if added_any or not store.is_signature_current():
        print("[INFO] Fitting TF-IDF embedder over the full corpus...")
        store.reindex_embeddings()

    print("=" * 65)
    final_stats = store.get_stats()
    print("Ingestion Pipeline Completed Successfully!")
    print(f" * Total Indexed Documents: {final_stats['total_documents']}")
    print(f" * Total Vector Chunks:     {final_stats['total_chunks']}")
    print("=" * 65)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CampusBot Document Ingestion Pipeline")
    parser.add_argument("--rebuild", action="store_true", help="Clear and re-index all documents from scratch")
    parser.add_argument("--stats", action="store_true", help="Show current vector index statistics")
    args = parser.parse_args()

    if args.stats:
        store = PersistentVectorStore(Config.VECTOR_STORE_DIR)
        stats = store.get_stats()
        print(f"Vector Store Statistics:")
        print(f"Documents: {stats['total_documents']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        for d in stats['documents']:
            print(f" - {d['document_name']}: {d['chunks']} chunks")
        sys.exit(0)

    exit_code = run_ingestion(rebuild=args.rebuild)
    sys.exit(exit_code)
