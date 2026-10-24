import os
import sys

# Locate files and project directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.document_loader import process_document_to_chunks
from utils.embeddings import get_embeddings_batch
from utils.vector_store import NumpyVectorStore

def index_brochure():
    print("==================================================")
    print(" CAMPUSBOT BROCHURE INDEXATION UTILITY")
    print("==================================================")
    
    file_path = "data/documents/sample_brochure.txt"
    filename = "sample_brochure.txt"
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return
        
    print(f"1. Reading and chunking '{filename}'...")
    chunks = process_document_to_chunks(file_path)
    print(f"-> Created {len(chunks)} chunks.")
    
    if not chunks:
        print("Error: No chunks extracted.")
        return
        
    print("2. Generating embeddings...")
    texts_to_embed = [c["text"] for c in chunks]
    embeddings = get_embeddings_batch(texts_to_embed)
    print(f"-> Generated {len(embeddings)} embedding vectors.")
    
    print("3. Storing chunks in NumpyVectorStore...")
    vector_store = NumpyVectorStore(storage_dir="vectorstore")
    
    # Remove existing index entries to prevent duplication
    removed_count = vector_store.delete_document(filename)
    if removed_count > 0:
        print(f"-> Removed {removed_count} existing chunks from vector store.")
        
    vector_store.add_chunks(chunks, embeddings)
    print("-> Successfully added chunks to database.")
    
    # Verify stats
    stats = vector_store.get_document_stats()
    print("\n==================================================")
    print(" VECTOR STORE STATISTICS:")
    for stat in stats:
        print(f"- Document: {stat['document_name']}")
        print(f"  Chunks count: {stat['chunks']}")
        print(f"  Last updated: {stat['upload_date']}")
    print("==================================================")

if __name__ == "__main__":
    index_brochure()
