from utils.embeddings import get_embedding

def retrieve_context(query, vector_store, top_k=4, min_similarity=0.25):
    """
    Takes a query, converts it to an embedding, and retrieves the most relevant
    text chunks from the NumpyVectorStore.
    If the embedding API fails (due to quota limits), falls back to a term-frequency
    based keyword matching search.
    """
    if not query.strip():
        return []
        
    # Gracefully handle empty knowledge base without making API calls
    if not vector_store.chunks:
        return []
        
    # 1. Attempt Semantic search using OpenAI Embeddings
    try:
        # Convert query to vector embedding
        query_embedding = get_embedding(query)
        
        # Search the vector store
        search_results = vector_store.search(
            query_embedding, 
            top_k=top_k, 
            min_similarity=min_similarity
        )
        
        retrieved_chunks = []
        for chunk, score in search_results:
            retrieved_chunks.append({
                "text": chunk["text"],
                "document_name": chunk["metadata"].get("document_name", "Unknown Document"),
                "page": chunk["metadata"].get("page", 1),
                "similarity": score
            })
        
        # If we got semantic matches that are valid, return them
        if retrieved_chunks:
            return retrieved_chunks
    except Exception as e:
        print(f"Semantic embedding search failed: {e}. Falling back to keyword search.")
        
    # 2. Keyword-based search fallback (useful for testing with depleted OpenAI credits)
    import re
    stop_words = {
        "what", "is", "the", "a", "an", "and", "or", "but", "if", "then", "of", 
        "to", "for", "in", "on", "at", "by", "with", "about", "are", "be", 
        "you", "your", "who", "whom", "whose", "does", "do", "how"
    }
    query_words = {word for word in re.findall(r'\w+', query.lower()) if word not in stop_words}
    if not query_words:
        # Fallback to unfiltered if query only contains stop words
        query_words = set(re.findall(r'\w+', query.lower()))
        if not query_words:
            return []
        
    keyword_results = []
    for chunk in vector_store.chunks:
        chunk_text_lower = chunk["text"].lower()
        # Compute word overlap score
        matches = sum(1 for word in query_words if word in chunk_text_lower)
        score = matches / len(query_words)
        
        # Filter chunks that have at least some keyword matches
        if score > 0.05:
            keyword_results.append((chunk, score))
            
    # Sort by overlap score descending
    keyword_results.sort(key=lambda x: x[1], reverse=True)
    
    retrieved_chunks = []
    for chunk, score in keyword_results[:top_k]:
        retrieved_chunks.append({
            "text": chunk["text"],
            "document_name": chunk["metadata"].get("document_name", "Unknown Document"),
            "page": chunk["metadata"].get("page", 1),
            "similarity": score
        })
        
    return retrieved_chunks
