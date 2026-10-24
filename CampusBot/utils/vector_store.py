import os
import pickle

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    import math

class NumpyVectorStore:
    def __init__(self, storage_dir="vectorstore"):
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, "index.pkl")
        self.chunks = []      # List of dicts: {"text": str, "metadata": dict}
        self.embeddings = []  # List of lists of floats: [[float], ...]
        self._load()

    def _load(self):
        """
        Loads the index from file if it exists.
        """
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.chunks = data.get("chunks", [])
                    self.embeddings = data.get("embeddings", [])
            except Exception as e:
                print(f"Error loading vector index: {e}")
                self.chunks = []
                self.embeddings = []

    def save(self):
        """
        Saves the current index to the pickle file.
        """
        os.makedirs(self.storage_dir, exist_ok=True)
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump({
                    "chunks": self.chunks,
                    "embeddings": self.embeddings
                }, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save vector index to file: {str(e)}")

    def add_chunks(self, chunks, embeddings):
        """
        Appends chunks and their corresponding embeddings to the vector store.
        chunks: list of dicts: {"text": str, "metadata": dict}
        embeddings: list of list of floats
        """
        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks and embeddings must match.")
        
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
        self.save()

    def delete_document(self, document_name):
        """
        Deletes all chunks and embeddings associated with a given document name.
        """
        new_chunks = []
        new_embeddings = []
        deleted_count = 0
        
        for chunk, embedding in zip(self.chunks, self.embeddings):
            if chunk["metadata"].get("document_name") == document_name:
                deleted_count += 1
            else:
                new_chunks.append(chunk)
                new_embeddings.append(embedding)
                
        self.chunks = new_chunks
        self.embeddings = new_embeddings
        self.save()
        return deleted_count

    def search(self, query_embedding, top_k=4, min_similarity=0.25):
        """
        Performs Cosine Similarity search.
        Uses optimized NumPy operations if available, or falls back to pure-Python
        math operations to guarantee compatibility with any Python environment.
        Returns a list of tuples: (chunk_dict, similarity_score) sorted descending by similarity.
        """
        if not self.chunks or not self.embeddings:
            return []
            
        try:
            results = []
            if HAS_NUMPY:
                # Optimized NumPy vector search
                embeddings_matrix = np.array(self.embeddings)  # Shape: (N, D)
                query_vec = np.array(query_embedding)          # Shape: (D,)
                
                # Calculate dot products
                dot_product = np.dot(embeddings_matrix, query_vec)
                
                # Calculate norms
                norms_embeddings = np.linalg.norm(embeddings_matrix, axis=1)
                norm_query = np.linalg.norm(query_vec)
                
                # Divide to get cosine similarity
                similarities = dot_product / (norms_embeddings * norm_query + 1e-9)
                
                # Get indices sorted descending by similarity score
                top_indices = np.argsort(similarities)[::-1]
                
                for idx in top_indices:
                    score = float(similarities[idx])
                    if score >= min_similarity:
                        results.append((self.chunks[idx], score))
                    if len(results) >= top_k:
                        break
            else:
                # Resilient pure-Python Cosine Similarity fallback
                def magnitude(v):
                    return math.sqrt(sum(x*x for x in v))
                
                def dot_product(v1, v2):
                    return sum(x*y for x, y in zip(v1, v2))
                
                norm_query = magnitude(query_embedding)
                scores = []
                
                for idx, emb in enumerate(self.embeddings):
                    norm_emb = magnitude(emb)
                    if norm_emb == 0 or norm_query == 0:
                        score = 0.0
                    else:
                        score = dot_product(emb, query_embedding) / (norm_emb * norm_query)
                    scores.append((idx, score))
                
                # Sort indices by score descending
                scores.sort(key=lambda x: x[1], reverse=True)
                
                for idx, score in scores:
                    if score >= min_similarity:
                        results.append((self.chunks[idx], score))
                    if len(results) >= top_k:
                        break
                        
            return results
        except Exception as e:
            print(f"Error during vector store search: {e}")
            return []

    def get_document_stats(self):
        """
        Returns statistical summaries of processed documents in the database.
        Returns list of dicts: {"document_name": str, "chunks": int, "upload_date": str}
        """
        doc_stats = {}
        for chunk in self.chunks:
            meta = chunk["metadata"]
            doc_name = meta.get("document_name")
            if not doc_name:
                continue
            if doc_name not in doc_stats:
                doc_stats[doc_name] = {
                    "document_name": doc_name,
                    "chunks": 0,
                    "upload_date": meta.get("upload_date", "Unknown")
                }
            doc_stats[doc_name]["chunks"] += 1
            
        return list(doc_stats.values())
