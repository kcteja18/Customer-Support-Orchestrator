from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleDoc:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class LocalRetriever:
    """A very small retriever using TF-IDF over local files (no external API).

    Usage:
        retriever = LocalRetriever(data_dir='examples/data')
        docs = retriever.get_relevant_documents('reset password')
    """

    def __init__(self, data_dir: str = "examples/data", k: int = 4):
        self.data_dir = Path(data_dir)
        self.k = k
        self._docs: List[SimpleDoc] = []
        self._vectorizer = None
        self._matrix = None
        self._fit()

    def _fit(self):
        texts = []
        for p in sorted(self.data_dir.rglob("*.md")):
            txt = p.read_text(encoding="utf-8")
            texts.append(txt)
            self._docs.append(SimpleDoc(page_content=txt, metadata={"source": str(p)}))

        if texts:
            self._vectorizer = TfidfVectorizer(stop_words="english").fit(texts)
            self._matrix = self._vectorizer.transform(texts)
        else:
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._matrix = np.zeros((0, 0))

    def get_relevant_documents(self, query: str) -> List[SimpleDoc]:
        if self._matrix.size == 0:
            return []
        qv = self._vectorizer.transform([query])
        scores = cosine_similarity(qv, self._matrix)[0]
        idxs = np.argsort(scores)[::-1][: self.k]
        return [self._docs[i] for i in idxs if scores[i] > 0]

    # compatibility helper used by run_demo when it calls `retriever.get_relevant_documents`
