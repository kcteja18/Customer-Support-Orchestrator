"""Enhanced Streamlit UI for Customer Support Orchestrator.

Provides an interactive chat interface with history, metrics, and document visualization.
"""
import streamlit as st
import requests
from datetime import datetime
import time
from typing import List, Dict, Any

# Page config
st.set_page_config(
    page_title="Customer Support AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #1a1a2e;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #16213e;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        color: #eaeaea;
    }
    
    /* User messages */
    .stChatMessage[data-testid="user-message"] {
        background-color: #0f3460;
    }
    
    /* Assistant messages */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #16213e;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #16213e;
        border: 1px solid #2d4059;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #eaeaea;
    }
    
    /* Document cards */
    .doc-card {
        background-color: #0f3460;
        border-left: 4px solid #4CAF50;
        border-radius: 4px;
        padding: 12px;
        margin: 8px 0;
        color: #eaeaea;
    }
    
    /* Text color */
    .stMarkdown, p, span, div {
        color: #eaeaea;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #16213e;
    }
    
    section[data-testid="stSidebar"] * {
        color: #eaeaea;
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        background-color: #0f3460;
        color: #eaeaea;
        border: 1px solid #2d4059;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #0f3460;
        color: #eaeaea;
        border: 1px solid #2d4059;
    }
    
    .stButton button:hover {
        background-color: #16213e;
        border-color: #4CAF50;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #16213e;
        color: #eaeaea;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #4CAF50;
    }
    
    [data-testid="stMetricLabel"] {
        color: #eaeaea;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> Dict[str, Any]:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def query_api(query: str, session_id: str, use_workflow: bool = False, top_k: int = 3) -> Dict[str, Any]:
    """Send query to the API with session support."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": query,
                "session_id": session_id,
                "use_workflow": use_workflow,
                "top_k": top_k
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}


def submit_feedback(query: str, answer: str, rating: int, session_id: str, comment: str = "") -> Dict[str, Any]:
    """Submit feedback for a response."""
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "query": query,
                "answer": answer,
                "rating": rating,
                "comment": comment,
                "session_id": session_id
            },
            timeout=5
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        return {"error": str(e)}


def ingest_documents() -> Dict[str, Any]:
    """Trigger document ingestion."""
    try:
        response = requests.post(f"{API_URL}/ingest", json={}, timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        return {"error": str(e)}


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "total_response_time" not in st.session_state:
        st.session_state.total_response_time = 0.0
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()


def display_message(role: str, content: str, metadata: Dict = None, message_id: str = None):
    """Display a chat message with optional metadata and feedback."""
    with st.chat_message(role):
        st.markdown(content)
        
        if metadata and role == "assistant":
            # Show cache indicator
            if metadata.get('cached'):
                st.success("⚡ Instant response from cache")
            
            # Always show documents first if available
            if metadata.get('documents'):
                st.markdown("---")
                st.markdown("### 📚 Referenced Documents")
                for i, doc in enumerate(metadata['documents'], 1):
                    source_name = doc['source'].replace('.md', '').replace('_', ' ').title()
                    with st.expander(f"📄 {source_name}", expanded=(i == 1)):
                        st.markdown(doc['content'])
            
            # Feedback buttons
            if message_id and message_id not in st.session_state.feedback_given:
                st.markdown("---")
                st.markdown("**Was this helpful?**")
                col1, col2, col3, col4 = st.columns([1, 1, 1, 5])
                
                with col1:
                    if st.button("👍", key=f"up_{message_id}"):
                        feedback = submit_feedback(
                            metadata.get('query', ''),
                            content,
                            5,
                            st.session_state.session_id,
                            "Helpful"
                        )
                        if feedback and 'error' not in feedback:
                            st.success("Thanks for the feedback!")
                            st.session_state.feedback_given.add(message_id)
                            st.rerun()
                
                with col2:
                    if st.button("👎", key=f"down_{message_id}"):
                        feedback = submit_feedback(
                            metadata.get('query', ''),
                            content,
                            2,
                            st.session_state.session_id,
                            "Not helpful"
                        )
                        if feedback and 'error' not in feedback:
                            st.info("Thanks! We'll work on improving this.")
                            st.session_state.feedback_given.add(message_id)
                            st.rerun()
            
            elif message_id in st.session_state.feedback_given:
                st.markdown("*✓ Feedback submitted*")
            
            # Show metrics in expander
            with st.expander("📊 Query Details", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Confidence", f"{metadata.get('confidence', 0):.2%}")
                
                with col2:
                    st.metric("Response Time", f"{metadata.get('total_time', 0):.2f}s")
                
                with col3:
                    st.metric("Documents", metadata.get('num_documents', 0))
                
                with col4:
                    turns = metadata.get('conversation_turns', 0)
                    st.metric("Conv. Turns", turns)
                
                if metadata.get('should_escalate'):
                    st.warning("⚠️ This query may require human escalation")


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.title("🤖 Customer Support AI Assistant")
    st.markdown("*Powered by RAG, LangChain, and LangGraph*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API health check
        health = check_api_health()
        if health:
            status = health.get("status", "unknown")
            if status == "healthy":
                st.success("✅ API Connected")
            else:
                st.warning("⚠️ API Degraded")
            
            st.info(f"**Mode:** {health.get('mode', 'unknown')}")
        else:
            st.error("❌ API Offline")
            st.markdown("""
            **Start the API:**
            ```bash
            python backend/main.py
            ```
            """)
            st.stop()
        
        st.divider()
        
        # Query settings
        st.subheader("Query Options")
        use_workflow = st.checkbox(
            "Use LangGraph Workflow",
            value=True,
            help="Enable multi-agent workflow with intent classification, retrieval routing, and escalation logic"
        )
        
        top_k = st.slider(
            "Documents to Retrieve",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of relevant documents to retrieve"
        )
        
        st.divider()
        
        # Management actions
        st.subheader("🔧 Management")
        
        if st.button("📥 Ingest Documents", use_container_width=True):
            with st.spinner("Starting ingestion..."):
                result = ingest_documents()
                if result and "error" not in result:
                    st.success(result.get("message", "Ingestion started"))
                else:
                    st.error(result.get("error", "Ingestion failed"))
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.session_state.total_response_time = 0.0
            st.session_state.feedback_given = set()
            st.rerun()
        
        st.divider()
        
        # Statistics
        st.subheader("📈 Session Stats")
        st.metric("Queries", st.session_state.query_count)
        if st.session_state.query_count > 0:
            avg_time = st.session_state.total_response_time / st.session_state.query_count
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
        
        # Show cache stats if available
        if health and health.get('cache_stats'):
            with st.expander("⚡ Cache Performance"):
                cache_stats = health['cache_stats']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Hit Rate", f"{cache_stats.get('hit_rate_percent', 0):.1f}%")
                    st.metric("Cache Size", cache_stats.get('size', 0))
                with col2:
                    st.metric("Total Hits", cache_stats.get('hits', 0))
                    st.metric("Misses", cache_stats.get('misses', 0))
    
    # Main chat interface
    st.markdown("### 💬 Chat")
    st.caption(f"Session ID: `{st.session_state.session_id}`")
    
    # Display chat history
    for idx, msg in enumerate(st.session_state.messages):
        message_id = f"msg_{idx}"
        display_message(
            msg["role"],
            msg["content"],
            msg.get("metadata"),
            message_id if msg["role"] == "assistant" else None
        )
    
    # Chat input
    if query := st.chat_input("Ask me anything about customer support..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now()
        })
        display_message("user", query)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                response = query_api(
                    query,
                    st.session_state.session_id,
                    use_workflow,
                    top_k
                )
                elapsed_time = time.time() - start_time
                
                if "error" in response:
                    st.error(f"❌ {response['error']}")
                    answer = f"Sorry, I encountered an error: {response['error']}"
                    metadata = None
                else:
                    answer = response.get("answer", "No answer generated")
                    metrics = response.get("metrics", {})
                    metadata = {
                        "query": query,
                        "confidence": response.get("confidence", 0),
                        "should_escalate": response.get("should_escalate", False),
                        "documents": response.get("documents", []),
                        "cached": response.get("cached", False),
                        "total_time": metrics.get("total_time", elapsed_time),
                        "num_documents": len(response.get("documents", [])),
                        "conversation_turns": metrics.get("conversation_turns", 0)
                    }
                    
                    # Update session stats
                    st.session_state.query_count += 1
                    st.session_state.total_response_time += metadata["total_time"]
                
                st.markdown(answer)
                
                # Show cache indicator
                if metadata and metadata.get('cached'):
                    st.success("⚡ Instant response from cache")
                
                # Show documents first if available
                if metadata and metadata.get('documents'):
                    st.markdown("---")
                    st.markdown("### 📚 Referenced Documents")
                    for i, doc in enumerate(metadata['documents'], 1):
                        source_name = doc['source'].replace('.md', '').replace('_', ' ').title()
                        with st.expander(f"📄 {source_name}", expanded=(i == 1)):
                            st.markdown(doc['content'])
                
                # Show metadata in collapsed expander
                if metadata:
                    with st.expander("📊 Query Details", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Confidence", f"{metadata['confidence']:.2%}")
                        
                        with col2:
                            st.metric("Response Time", f"{metadata['total_time']:.2f}s")
                        
                        with col3:
                            st.metric("Documents", metadata['num_documents'])
                        
                        with col4:
                            turns = metadata.get('conversation_turns', 0)
                            st.metric("Conv. Turns", turns)
                        
                        if metadata['should_escalate']:
                            st.warning("⚠️ This query may require human escalation")
        
        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "metadata": metadata,
            "timestamp": datetime.now()
        })
        
        st.rerun()
    
    # Sample queries
    with st.expander("💡 Sample Queries"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - How do I reset my password?
            - What are your business hours?
            - I need to update my billing information
            """)
        
        with col2:
            st.markdown("""
            - My account is locked, help!
            - How do I cancel my subscription?
            - I want to speak to a manager
            """)


if __name__ == "__main__":
    main()
