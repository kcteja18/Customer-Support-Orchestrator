import os
from typing import List

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class HuggingFaceEmbeddings:
    """Minimal wrapper around sentence-transformers to provide
    `embed_documents` and `embed_query` for compatibility with LangChain/Chroma.

    Model name can be set via `HUGGINGFACE_EMBEDDING_MODEL` env var or passed.
    """

    def __init__(self, model_name: str | None = None):
        model_name = model_name or os.getenv(
            "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers not installed. Add `sentence-transformers` to requirements.txt"
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # returns list of embedding vectors
        embs = self.model.encode(texts, show_progress_bar=False)
        return [list(map(float, e)) for e in embs]

    def embed_query(self, text: str) -> List[float]:
        emb = self.model.encode([text], show_progress_bar=False)[0]
        return list(map(float, emb))
