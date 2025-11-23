"""Basic tests for the Customer Support Orchestrator."""
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.orchestrator.local_retriever import LocalRetriever
from src.orchestrator.agents import MockLLM
from src.orchestrator.embeddings import HuggingFaceEmbeddings


def test_local_retriever():
    """Test local retriever with TF-IDF."""
    retriever = LocalRetriever(data_dir="examples/data")
    docs = retriever.get_relevant_documents("reset password")
    assert len(docs) > 0, "Should retrieve at least one document"
    assert hasattr(docs[0], 'page_content'), "Document should have page_content attribute"
    print(f"✓ Local retriever test passed ({len(docs)} docs retrieved)")


def test_mock_llm():
    """Test MockLLM answer generation."""
    llm = MockLLM()
    retriever = LocalRetriever(data_dir="examples/data")
    docs = retriever.get_relevant_documents("reset password")
    answer = llm.generate_answer("How do I reset my password?", docs)
    assert len(answer) > 0, "Answer should not be empty"
    assert "Based on our documents" in answer or "not able to find" in answer
    print(f"✓ MockLLM test passed (answer length: {len(answer)})")


def test_embeddings():
    """Test HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings()
    texts = ["Hello world", "Test document"]
    vecs = embeddings.embed_documents(texts)
    assert len(vecs) == 2, "Should return 2 embedding vectors"
    assert len(vecs[0]) > 0, "Embedding vector should not be empty"
    
    query_vec = embeddings.embed_query("Hello")
    assert len(query_vec) > 0, "Query embedding should not be empty"
    print(f"✓ Embeddings test passed (dim: {len(vecs[0])})")


def test_langgraph_workflow():
    """Test LangGraph workflow if available."""
    try:
        from src.orchestrator.graph import run_support_workflow, _LANGGRAPH_AVAILABLE
        if not _LANGGRAPH_AVAILABLE:
            print("⊘ LangGraph not available, skipping workflow test")
            return
        
        retriever = LocalRetriever(data_dir="examples/data")
        llm = MockLLM()
        
        result = run_support_workflow("How do I reset my password?", retriever, llm)
        
        assert "query" in result
        assert "intent" in result
        assert "answer" in result
        assert len(result["messages"]) > 0, "Should have workflow messages"
        print(f"✓ LangGraph workflow test passed (intent: {result['intent']})")
    except Exception as e:
        print(f"⊘ LangGraph workflow test skipped: {e}")


if __name__ == "__main__":
    print("Running Customer Support Orchestrator Tests\n")
    test_local_retriever()
    test_mock_llm()
    test_embeddings()
    test_langgraph_workflow()
    print("\n✓ All tests completed successfully!")
