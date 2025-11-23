from pathlib import Path
from typing import List
import os

try:
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.schema import Document
    from langchain.vectorstores import Chroma
    _LANGCHAIN_AVAILABLE = True
except Exception:
    CharacterTextSplitter = None
    Document = None
    Chroma = None
    _LANGCHAIN_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMADB_AVAILABLE = True
except Exception:
    chromadb = None
    Settings = None
    _CHROMADB_AVAILABLE = False

from .embeddings import HuggingFaceEmbeddings


def _simple_text_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - chunk_overlap
    return chunks


def ingest_from_directory(
    data_dir: str = "examples/data",
    persist_directory: str = "./.chroma",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """Ingest markdown/text files under `data_dir` into a Chroma vectorstore.

    This function will prefer LangChain's `Chroma` wrapper when available. If
    LangChain isn't installed, it will fall back to using `chromadb` directly
    with `sentence-transformers` embeddings (no OpenAI key required).
    """
    data_path = Path(data_dir)
    texts = []
    sources = []
    for p in sorted(data_path.rglob("*.md")):
        txt = p.read_text(encoding="utf-8")
        texts.append(txt)
        sources.append(str(p))

    hf_model = os.getenv("HUGGINGFACE_EMBEDDING_MODEL")
    embeddings = HuggingFaceEmbeddings(model_name=hf_model)

    # If LangChain is available, use its Chroma wrapper (compatibility)
    if _LANGCHAIN_AVAILABLE and Chroma is not None:
        # build langchain Documents and use Chroma.from_documents
        docs = []
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for t in texts:
            docs.extend(splitter.split_documents([Document(page_content=t)]))

        vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
        vectordb.persist()
        return vectordb

    # Fallback: use chromadb directly
    if not _CHROMADB_AVAILABLE:
        raise RuntimeError(
            "Neither langchain nor chromadb are available. Please install requirements.txt (pip install -r requirements.txt)."
        )

    # New Chroma PersistentClient API replaces legacy Client(Settings(...)) usage
    try:
        client = chromadb.PersistentClient(path=persist_directory)
    except Exception:
        client = chromadb.Client()  # in-memory fallback
    try:
        collection = client.get_collection("orchestrator")
    except Exception:
        collection = client.create_collection("orchestrator")

    documents = []
    metadatas = []
    ids = []
    for idx, txt in enumerate(texts):
        chunks = _simple_text_split(txt, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for cidx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": sources[idx], "chunk_index": cidx})
            ids.append(f"doc-{idx}-{cidx}")

    # compute embeddings
    vectors = embeddings.embed_documents(documents)

    # upsert into chroma collection
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors)
    # PersistentClient auto-persists, no explicit persist() needed
    return collection


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    persist = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
    print("Ingesting example data to", persist)
    ingest_from_directory(persist_directory=persist)
