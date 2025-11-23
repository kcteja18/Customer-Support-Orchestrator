import os
from typing import List

from .embeddings import HuggingFaceEmbeddings

try:
    from langchain.vectorstores import Chroma
    _LANGCHAIN_CHROMA = True
except Exception:
    Chroma = None
    _LANGCHAIN_CHROMA = False

try:
    import chromadb
    _CHROMADB = True
except Exception:
    chromadb = None
    _CHROMADB = False


class SimpleDoc:
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


def _make_chromadb_retriever(collection, embeddings, k: int):
    def get_relevant_documents(query: str) -> List[SimpleDoc]:
        q_emb = embeddings.embed_query(query)
        # chromadb collection.query expects list of embeddings
        resp = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
        docs = []
        items = resp.get("documents", [[]])[0]
        metas = resp.get("metadatas", [[]])[0]
        for d, m in zip(items, metas):
            docs.append(SimpleDoc(page_content=d, metadata=m))
        return docs

    return get_relevant_documents


def get_retriever(persist_directory: str = "./.chroma", k: int = 4):
    """Load the Chroma vectorstore and return a retriever-like object.

    If `langchain`'s `Chroma` is available, returns its retriever. Otherwise
    falls back to a small chromadb-based retriever compatible with the
    `get_relevant_documents(query)` shape used in the demo.
    """
    hf_model = os.getenv("HUGGINGFACE_EMBEDDING_MODEL")
    embeddings = HuggingFaceEmbeddings(model_name=hf_model)

    if _LANGCHAIN_CHROMA and Chroma is not None:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        return retriever

    if not _CHROMADB:
        raise RuntimeError("No Chroma/Chromadb available. Install requirements and try again.")

    # Use new PersistentClient API (legacy Client(Settings) is deprecated)
    try:
        client = chromadb.PersistentClient(path=persist_directory)
    except Exception:
        # Fallback: in-memory client if persistent path fails
        client = chromadb.Client()
    try:
        collection = client.get_collection("orchestrator")
    except Exception:
        collection = client.create_collection("orchestrator")
    # return a function-like object with get_relevant_documents
    class RetrieverWrapper:
        def get_relevant_documents(self, query: str):
            return _make_chromadb_retriever(collection, embeddings, k)(query)

    return RetrieverWrapper()
