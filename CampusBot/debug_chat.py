"""Debug the chat pipeline to find the exact error."""
import sys
import traceback

try:
    from services.rag_service import generate_chat_response, vector_store
    
    # Check vector store
    stats = vector_store.get_stats()
    print(f"Vector store: {stats['total_chunks']} chunks, {stats['total_documents']} docs", flush=True)
    
    if vector_store.embeddings:
        print(f"Stored embedding dimension: {len(vector_store.embeddings[0])}", flush=True)
    
    # Test embedding
    from services.embedding_service import get_embedding
    query_vec = get_embedding("What are admission requirements?")
    print(f"Query embedding dimension: {len(query_vec)}", flush=True)
    
    # Test search
    from services.rag_service import retrieve_relevant_passages
    passages = retrieve_relevant_passages("What are admission requirements?")
    print(f"Retrieved {len(passages)} passages", flush=True)
    
    # Test full pipeline
    print("\nTesting full chat pipeline...", flush=True)
    result = generate_chat_response("What are the admission requirements?", conversation_id="debug-test-1")
    print(f"Success: {result['success']}", flush=True)
    print(f"Mode: {result['mode']}", flush=True)
    print(f"Answer preview: {result['answer'][:200]}", flush=True)
    
except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
