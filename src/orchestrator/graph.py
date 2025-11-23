"""LangGraph workflow orchestrator for multi-agent customer support routing.

This module defines a graph-based workflow using LangGraph that:
1. Classifies incoming queries (answer vs. escalate)
2. Routes to appropriate agents (retrieval + answer, or escalation)
3. Returns structured responses with confidence and context
"""
from typing import TypedDict, Annotated, Literal
import operator

try:
    from langgraph.graph import StateGraph, END
    _LANGGRAPH_AVAILABLE = True
except Exception:
    StateGraph = None
    END = None
    _LANGGRAPH_AVAILABLE = False


class WorkflowState(TypedDict):
    """State passed through the LangGraph workflow."""
    query: str
    intent: str  # "answer" or "escalate"
    retrieved_docs: list
    answer: str
    confidence: float
    escalate: bool
    ticket: dict
    messages: Annotated[list, operator.add]  # log of workflow steps


def classify_intent_node(state: WorkflowState) -> WorkflowState:
    """Classify the query intent: simple heuristic for demo.
    
    In production, this would use a trained classifier or LLM prompt.
    """
    query_lower = state["query"].lower()
    # Simple keyword heuristic
    if any(kw in query_lower for kw in ["urgent", "escalate", "speak to human", "manager"]):
        state["intent"] = "escalate"
        state["messages"].append("Intent: escalate (urgent keywords detected)")
    else:
        state["intent"] = "answer"
        state["messages"].append("Intent: answer (standard query)")
    return state


def retrieve_docs_node(state: WorkflowState, retriever) -> WorkflowState:
    """Retrieve relevant documents using the retriever."""
    docs = retriever.get_relevant_documents(state["query"])
    state["retrieved_docs"] = docs
    state["messages"].append(f"Retrieved {len(docs)} documents")
    return state


def answer_node(state: WorkflowState, llm) -> WorkflowState:
    """Generate answer using retrieved docs + LLM."""
    if not state["retrieved_docs"]:
        state["answer"] = "I don't have enough information to answer this query."
        state["confidence"] = 0.1
        state["escalate"] = True
        state["messages"].append("Answer: low confidence (no docs retrieved)")
    else:
        # Simple concatenation for demo; in production use RetrievalQA or custom prompt
        context = "\n\n".join(getattr(d, 'page_content', str(d)) for d in state["retrieved_docs"][:3])
        answer_text = llm.generate_answer(state["query"], state["retrieved_docs"])
        state["answer"] = answer_text
        # Heuristic confidence based on length and uncertainty phrases
        if any(p in answer_text.lower() for p in ["i don't know", "not sure", "unclear"]):
            state["confidence"] = 0.3
            state["escalate"] = True
        else:
            state["confidence"] = 0.8
            state["escalate"] = False
        state["messages"].append(f"Answer generated (confidence: {state['confidence']})")
    return state


def escalate_node(state: WorkflowState) -> WorkflowState:
    """Create escalation ticket with context."""
    context = "\n\n".join(getattr(d, 'page_content', str(d))[:500] for d in state["retrieved_docs"][:3])
    state["ticket"] = {
        "subject": f"Escalation: {state['query'][:100]}",
        "query": state["query"],
        "context": context,
        "reason": "User request or low confidence"
    }
    state["messages"].append("Escalation ticket created")
    return state


def route_after_classify(state: WorkflowState) -> Literal["retrieve", "escalate_direct"]:
    """Routing logic after intent classification."""
    if state["intent"] == "escalate":
        return "escalate_direct"
    return "retrieve"


def route_after_answer(state: WorkflowState) -> Literal["escalate_post_answer", "end"]:
    """Routing logic after answer generation."""
    if state.get("escalate", False):
        return "escalate_post_answer"
    return "end"


def build_support_graph(retriever, llm):
    """Build and return the LangGraph workflow for customer support.
    
    Args:
        retriever: Retriever object with `get_relevant_documents(query)` method
        llm: LLM object with `generate_answer(query, docs)` method
    
    Returns:
        Compiled LangGraph workflow
    """
    if not _LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph not available. Install it: pip install langgraph")
    
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("classify", classify_intent_node)
    workflow.add_node("retrieve", lambda s: retrieve_docs_node(s, retriever))
    workflow.add_node("answer", lambda s: answer_node(s, llm))
    workflow.add_node("escalate_direct", escalate_node)
    workflow.add_node("escalate_post_answer", escalate_node)
    
    # Define edges
    workflow.set_entry_point("classify")
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "retrieve": "retrieve",
            "escalate_direct": "escalate_direct"
        }
    )
    workflow.add_edge("retrieve", "answer")
    workflow.add_conditional_edges(
        "answer",
        route_after_answer,
        {
            "escalate_post_answer": "escalate_post_answer",
            "end": END
        }
    )
    workflow.add_edge("escalate_direct", END)
    workflow.add_edge("escalate_post_answer", END)
    
    return workflow.compile()


def run_support_workflow(query: str, retriever, llm) -> dict:
    """Execute the support workflow for a given query.
    
    Args:
        query: User query string
        retriever: Retriever instance
        llm: LLM instance
    
    Returns:
        Final workflow state as dict
    """
    graph = build_support_graph(retriever, llm)
    
    initial_state: WorkflowState = {
        "query": query,
        "intent": "",
        "retrieved_docs": [],
        "answer": "",
        "confidence": 0.0,
        "escalate": False,
        "ticket": {},
        "messages": []
    }
    
    result = graph.invoke(initial_state)
    return result
