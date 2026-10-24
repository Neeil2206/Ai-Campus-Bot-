import os
import sys
from dotenv import load_dotenv

# Locate and load environment variables relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH, override=True)

# Verify project imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.document_loader import process_document_to_chunks
from utils.vector_store import NumpyVectorStore
from utils.embeddings import get_embedding
from utils.retriever import retrieve_context
from utils.rag_pipeline import rewrite_query, generate_grounded_response
from utils.local_llm import get_ollama_status, get_ollama_models

def run_tests():
    print("==================================================")
    print(" CAMPUSBUG / CAMPUSBOT PIPELINE VERIFICATION")
    print("==================================================")
    
    # ----------------------------------------------------
    # TEST 1: Document Parsing and Chunking
    # ----------------------------------------------------
    print("\n[TEST 1] Testing Document Loader & Chunking...")
    sample_file = "sample_brochure.txt"
    if not os.path.exists(sample_file):
        print(f"[FAIL] Error: Sample brochure file '{sample_file}' not found.")
        sys.exit(1)
        
    try:
        chunks = process_document_to_chunks(sample_file, chunk_size=800, chunk_overlap=150)
        print(f"[OK] Success: Parsed '{sample_file}' into {len(chunks)} chunks.")
        
        if len(chunks) > 0:
            print(f"-> Sample Chunk 1 Text (First 150 chars):\n   \"{chunks[0]['text'][:150]}...\"")
            print(f"-> Sample Chunk 1 Metadata: {chunks[0]['metadata']}")
        else:
            print("[FAIL] Error: Chunks list is empty.")
            sys.exit(1)
    except Exception as e:
        print(f"[FAIL] Exception in Document Loader: {e}")
        sys.exit(1)

    # ----------------------------------------------------
    # TEST 2: NumPy Vector Store Serialization & Similarity Search
    # ----------------------------------------------------
    print("\n[TEST 2] Testing NumPy Vector Store Indexing...")
    try:
        test_store_dir = "vectorstore_test"
        if os.path.exists(test_store_dir):
            import shutil
            shutil.rmtree(test_store_dir)
            
        store = NumpyVectorStore(storage_dir=test_store_dir)
        print("[OK] Success: Initialized vector store.")
        
        # Insert test chunks with mock 4-dimensional embeddings
        test_chunks = [
            {"text": "B.Tech Computer Science eligibility is 60% in Physics, Chemistry, Math.", "metadata": {"document_name": "test.txt", "page": 1, "upload_date": "2026-08-12"}},
            {"text": "Hostel fee for double occupancy room is $1,200 per year including food.", "metadata": {"document_name": "test.txt", "page": 2, "upload_date": "2026-08-12"}}
        ]
        # Unit normalized mock embeddings
        test_embeddings = [
            [0.8, 0.6, 0.0, 0.0],
            [0.0, 0.0, 0.6, 0.8]
        ]
        
        store.add_chunks(test_chunks, test_embeddings)
        print(f"[OK] Success: Added {len(test_chunks)} mock chunks.")
        
        # Load from disk again to test persistence
        loaded_store = NumpyVectorStore(storage_dir=test_store_dir)
        print(f"[OK] Success: Saved and reloaded store. Core chunks found: {len(loaded_store.chunks)}")
        
        if len(loaded_store.chunks) != 2:
            print("[FAIL] Error: Chunks count mismatch after reloading.")
            sys.exit(1)
            
        # Search query matching B.Tech embedding (close to [1.0, 0.0, 0.0, 0.0])
        query_vec = [1.0, 0.0, 0.0, 0.0]
        results = loaded_store.search(query_vec, top_k=1, min_similarity=0.1)
        print(f"[OK] Success: Search performed.")
        if results:
            match_chunk, score = results[0]
            print(f"-> Match Score: {score:.4f}")
            print(f"-> Match Text: \"{match_chunk['text']}\"")
            if "B.Tech" not in match_chunk["text"]:
                print("[FAIL] Error: Search returned incorrect chunk.")
                sys.exit(1)
        else:
            print("[FAIL] Error: Search returned zero results.")
            sys.exit(1)
            
        # Test document deletion
        deleted_count = loaded_store.delete_document("test.txt")
        print(f"[OK] Success: Deleted document 'test.txt'. Removed chunks: {deleted_count}.")
        if len(loaded_store.chunks) != 0:
            print("[FAIL] Error: Document deletion failed to clear items.")
            sys.exit(1)
            
        # Cleanup test store directory
        if os.path.exists(test_store_dir):
            import shutil
            shutil.rmtree(test_store_dir)
            
    except Exception as e:
        print(f"[FAIL] Exception in Vector Store: {e}")
        sys.exit(1)

    # ----------------------------------------------------
    # TEST 3: Local AI Connectivity & RAG Pipeline
    # ----------------------------------------------------
    print("\n[TEST 3] Testing Local AI Connectivity & RAG Response...")
    try:
        # Embed check
        print("Testing local embeddings generation (SentenceTransformers)...")
        test_emb = get_embedding("admission query")
        print(f"[OK] Success: Generated vector embedding (Dimensions: {len(test_emb)}).")
        
        ollama_online = get_ollama_status()
        if not ollama_online:
            print("[WARN] Warning: Local Ollama service is not running. Skipping local LLM tests.")
            print("   To test the complete local RAG pipeline, please start Ollama and run again.")
        else:
            print("Connecting to local Ollama service...")
            models = get_ollama_models()
            print(f"[OK] Success: Found local models: {models}")
            
            # Query rewrite check
            print("Testing Query Rewriter...")
            history = [
                {"role": "user", "content": "What is the fee for B.Tech?"},
                {"role": "assistant", "content": "The fee is $5,000 per year."}
            ]
            rewritten = rewrite_query("What about hostel costs?", history)
            print(f"-> Original follow-up: \"What about hostel costs?\"")
            print(f"-> Rewritten query: \"{rewritten}\"")
            
            # Grounded response check
            print("Testing Grounded local LLM Response...")
            retrieved = [{
                "text": "Merit-Based Scholarship provides a 50% tuition fee waiver for students scoring above 95% aggregate in 12th grade.",
                "document_name": "sample_brochure.txt",
                "page": 4,
                "similarity": 0.82
            }]
            response_text = generate_grounded_response(
                query="What scholarships are available for high scorers?", 
                retrieved_chunks=retrieved, 
                chat_history=[]
            )
            print(f"-> Local LLM Response:\n{response_text}")
            
    except Exception as e:
        print(f"[FAIL] Exception in Local AI test: {e}")
        sys.exit(1)

    print("\n==================================================")
    print(" VERIFICATION COMPLETE - CORE MODULES LOOK SOLID!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
