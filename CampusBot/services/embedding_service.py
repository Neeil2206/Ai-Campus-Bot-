import math
import re
import requests
import numpy as np
from config import Config
from services.ollama_service import check_ollama_reachable

# Signature identifying the deterministic offline embedding space. It is stored
# alongside the vector index so that if the embedding strategy ever changes we
# can detect a stale index and rebuild it instead of comparing incompatible
# vectors (which would silently return zero retrieval results).
EMBEDDING_SIGNATURE = "tfidf-v1"

# Common English stopwords removed before vectorizing so that frequent function
# words do not dominate the vectors and drown out meaningful domain terms.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "to",
    "of", "in", "on", "at", "by", "with", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "am", "do", "does", "did", "doing", "have",
    "has", "had", "having", "i", "you", "he", "she", "it", "we", "they", "me",
    "him", "her", "them", "my", "your", "his", "its", "our", "their", "this",
    "that", "these", "those", "what", "which", "who", "whom", "when", "where",
    "why", "how", "can", "could", "should", "would", "will", "shall", "may",
    "might", "must", "about", "into", "over", "under", "again", "any", "some",
    "there", "here", "so", "than", "too", "very", "just", "also", "tell", "me",
    "please", "give", "show", "know", "want", "many", "much", "get", "got",
}


def _tokenize(text):
    """Lowercase word tokens with stopwords and 1-char noise removed."""
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _normalize_vector(vec):
    """Normalize vector to unit length for accurate cosine similarity."""
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm > 1e-9:
        v = v / norm
    return v.tolist()


class TfidfEmbedder:
    """
    A small, fully-offline TF-IDF vectorizer used as CampusBot's canonical
    embedding space.

    Why TF-IDF instead of random hashing:
    - No hash collisions, so unrelated queries (e.g. "weather on Mars") share
      almost no vocabulary with admission documents and correctly score near
      zero, while genuine questions about fees, eligibility, or scholarships
      share strong, IDF-weighted terms with the right chunks.
    - It is deterministic, instant, and needs no network or embedding model, so
      the knowledge base works even when Ollama's embedding model is absent.
    - The SAME fitted vocabulary/IDF is used for both indexing documents and
      encoding live queries, guaranteeing both live in one comparable space.

    The fitted state (vocabulary + IDF weights) is small and is persisted with
    the vector index so queries always use the exact model the index was built
    with.
    """

    def __init__(self):
        self.vocab = {}      # term -> column index
        self.idf = []        # IDF weight aligned with vocab columns
        self.doc_count = 0

    def fit(self, texts):
        """Learn vocabulary and IDF weights from a list of documents."""
        doc_freq = {}
        docs = list(texts or [])
        for text in docs:
            for term in set(_tokenize(text)):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        self.doc_count = max(len(docs), 1)
        # Deterministic vocabulary ordering for reproducible indexes.
        ordered_terms = sorted(doc_freq.keys())
        self.vocab = {term: i for i, term in enumerate(ordered_terms)}
        n = self.doc_count
        self.idf = [math.log((n + 1) / (doc_freq[term] + 1)) + 1.0 for term in ordered_terms]
        return self

    def embed(self, text):
        """
        Return an L2-normalized TF-IDF vector for a text, restricted to the
        fitted vocabulary. Out-of-vocabulary terms contribute nothing, which is
        what lets unrelated questions be rejected instead of spuriously matching.
        """
        dim = len(self.vocab)
        if dim == 0:
            return []
        vec = np.zeros(dim, dtype=np.float32)
        tf = {}
        for tok in _tokenize(text):
            tf[tok] = tf.get(tok, 0) + 1
        for term, count in tf.items():
            idx = self.vocab.get(term)
            if idx is None:
                continue
            # Sub-linear TF scaling dampens very frequent terms within a chunk.
            vec[idx] += (1.0 + math.log(count)) * self.idf[idx]
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec.tolist()

    def to_state(self):
        return {"vocab": self.vocab, "idf": self.idf, "doc_count": self.doc_count}

    def load_state(self, state):
        state = state or {}
        self.vocab = dict(state.get("vocab", {}))
        self.idf = list(state.get("idf", []))
        self.doc_count = state.get("doc_count", 0)
        return self


# ---------------------------------------------------------------------------
# Module-level default embedder + backwards-compatible helper functions.
# The persistent vector store owns and persists its own fitted embedder and is
# the authoritative source for query embedding; these helpers exist so that
# standalone scripts and simple callers continue to work.
# ---------------------------------------------------------------------------
_default_embedder = TfidfEmbedder()


def fit_embedder(texts):
    """Fit the module-level default embedder on the given corpus."""
    _default_embedder.fit(texts)
    return _default_embedder


def get_embedder_state():
    return _default_embedder.to_state()


def load_embedder_state(state):
    _default_embedder.load_state(state)


def get_embedding(text):
    """
    Generate an embedding for a single text using the module-level default
    (corpus-fitted) TF-IDF embedder. If the embedder has not been fitted yet, it
    is fit on the single text so the call still returns a valid, non-empty vector.
    """
    if not text or not text.strip():
        dim = len(_default_embedder.vocab)
        return [0.0] * (dim if dim else 1)
    if not _default_embedder.vocab:
        _default_embedder.fit([text])
    return _default_embedder.embed(text)


def get_embeddings_batch(texts):
    """
    Generates embeddings for a list of texts in batch.
    """
    if not texts:
        return []
    return [get_embedding(t) for t in texts]
