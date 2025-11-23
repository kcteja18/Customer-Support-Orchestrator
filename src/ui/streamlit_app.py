"""Streamlit UI for Customer Support Orchestrator.

Run with: streamlit run src/ui/streamlit_app.py
"""
import streamlit as st
import sys
import os

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

from src.orchestrator.local_retriever import LocalRetriever
from src.orchestrator.agents import MockLLM

try:
    from src.orchestrator.graph import run_support_workflow, _LANGGRAPH_AVAILABLE
except Exception:
    run_support_workflow = None
    _LANGGRAPH_AVAILABLE = False


st.set_page_config(page_title="Customer Support Orchestrator", page_icon="🤖", layout="wide")

st.title("🤖 Customer Support Orchestrator")
st.markdown("**RAG + LangChain + LangGraph Demo**")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox("Mode", ["local", "hf"], help="Local uses TF-IDF retriever, HF uses Chroma+HF")
    data_dir = st.text_input("Data Directory", "examples/data", help="Path to knowledge base markdown files")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This demo showcases a multi-agent RAG system using:
    - **LangChain** for retrieval chains
    - **LangGraph** for workflow orchestration
    - **sentence-transformers** for local embeddings
    - **Chroma** for vector storage
    """)

# Main interface
query = st.text_input("Enter your support query:", placeholder="How do I reset my password?")

if st.button("Submit Query", type="primary"):
    if not query:
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Processing your query through the workflow..."):
            try:
                # Initialize retriever and LLM
                if mode == "local":
                    retriever = LocalRetriever(data_dir=data_dir)
                else:
                    from src.orchestrator.retriever import get_retriever
                    retriever = get_retriever(persist_directory=".chroma")
                
                llm = MockLLM()
                
                # Run workflow
                if _LANGGRAPH_AVAILABLE and run_support_workflow:
                    result = run_support_workflow(query, retriever, llm)
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Answer")
                        st.write(result["answer"])
                        
                        st.subheader("Workflow Steps")
                        for msg in result["messages"]:
                            st.text(f"• {msg}")
                    
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.0%}")
                        st.metric("Intent", result["intent"])
                        
                        if result.get("escalate"):
                            st.warning("⚠️ Escalation Required")
                            if result.get("ticket"):
                                with st.expander("Ticket Details"):
                                    st.json(result["ticket"])
                        else:
                            st.success("✅ Query Resolved")
                    
                    # Show retrieved documents
                    if result.get("retrieved_docs"):
                        with st.expander(f"📄 Retrieved Documents ({len(result['retrieved_docs'])})"):
                            for idx, doc in enumerate(result["retrieved_docs"][:3]):
                                st.markdown(f"**Document {idx+1}**")
                                content = getattr(doc, 'page_content', str(doc))
                                st.text(content[:300] + "..." if len(content) > 300 else content)
                                st.markdown("---")
                else:
                    st.error("LangGraph not available. Install with: pip install langgraph")
                    st.info("Falling back to simple retrieval...")
                    docs = retriever.get_relevant_documents(query)
                    answer = llm.generate_answer(query, docs)
                    st.write(answer)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("*Built with LangChain + LangGraph + Streamlit*")
