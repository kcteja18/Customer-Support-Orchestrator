"""FastAPI backend for Customer Support Orchestrator.

Provides REST API endpoints for querying, ingesting documents, and health checks.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import timedelta
import time
from contextlib import asynccontextmanager

from src.config import config
from src.logging_config import setup_logging, log_query_metrics
from src.orchestrator.retriever import get_retriever
from src.orchestrator.agents import SupportOrchestrator
from src.orchestrator.ingest import ingest_from_directory
from src.orchestrator.memory import ConversationMemory
from src.orchestrator.cache import QueryCache
from src.orchestrator.feedback import FeedbackCollector

# Initialize logging
logger = setup_logging(config.log_level, config.log_file)

# Global state
orchestrator = None
retriever = None
query_cache = None
feedback_collector = None
sessions = {}  # Store conversation memories by session_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global orchestrator, retriever, query_cache, feedback_collector
    
    logger.info("Starting Customer Support Orchestrator API...")
    
    # Initialize cache
    query_cache = QueryCache(ttl_minutes=60, max_size=500)
    logger.info("Query cache initialized")
    
    # Initialize feedback collector
    feedback_collector = FeedbackCollector()
    logger.info("Feedback collector initialized")
    
    # Validate configuration
    issues = config.validate()
    if issues:
        logger.warning(f"Configuration issues: {', '.join(issues)}")
    
    # Initialize retriever
    try:
        retriever = get_retriever(persist_directory=config.vector_store.persist_directory)
        logger.info("Retriever initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        retriever = None
    
    # Initialize orchestrator
    try:
        # Set environment variables for HF mode
        if config.model.mode == "hf":
            if config.model.hf_token:
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = config.model.hf_token
            if config.model.hf_model:
                os.environ["HUGGINGFACE_MODEL"] = config.model.hf_model
        
        orchestrator = SupportOrchestrator(
            retriever=retriever,
            mode=config.model.mode
        )
        logger.info(f"Orchestrator initialized in {config.model.mode} mode")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        orchestrator = None
    
    yield
    
    logger.info("Shutting down API...")


app = FastAPI(
    title="Customer Support Orchestrator",
    description="RAG-powered multi-agent customer support system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., min_length=1, description="User query text")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of documents to retrieve")
    use_workflow: Optional[bool] = Field(False, description="Use LangGraph workflow")


class Document(BaseModel):
    """Retrieved document model."""
    content: str
    source: str
    chunk_index: int


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    confidence: float
    should_escalate: bool
    documents: List[Document]
    session_id: str
    cached: bool
    metrics: Dict[str, Any]


class IngestRequest(BaseModel):
    """Ingestion request model."""
    data_directory: Optional[str] = Field(None, description="Directory containing documents")


class IngestResponse(BaseModel):
    """Ingestion response model."""
    status: str
    message: str
    documents_processed: int


class FeedbackRequest(BaseModel):
    """Feedback submission model."""
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comment: Optional[str] = Field("", description="Optional comment")
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    mode: str
    retriever_ready: bool
    orchestrator_ready: bool
    cache_stats: Optional[Dict[str, Any]] = None


# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Support Orchestrator API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    cache_stats = query_cache.get_stats() if query_cache else None
    
    return HealthResponse(
        status="healthy" if (orchestrator and retriever) else "degraded",
        mode=config.model.mode,
        retriever_ready=retriever is not None,
        orchestrator_ready=orchestrator is not None,
        cache_stats=cache_stats
    )


@app.post("/query", response_model=QueryResponse)
async def query_support(request: QueryRequest):
    """Query the support system with caching and conversation memory.
    
    Args:
        request: Query request containing user question
    
    Returns:
        QueryResponse with answer, confidence, and supporting documents
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    logger.info(f"Received query: {request.query[:100]}...")
    
    # Get or create session
    session_id = request.session_id or f"session_{int(time.time()*1000)}"
    if session_id not in sessions:
        sessions[session_id] = ConversationMemory(session_id=session_id)
    
    memory = sessions[session_id]
    
    # Check cache first
    cached_response = query_cache.get(request.query) if query_cache else None
    if cached_response:
        logger.info("Cache hit! Returning cached response")
        # Add to conversation memory
        memory.add_message("user", request.query)
        memory.add_message("assistant", cached_response['answer'], 
                          {'cached': True, 'confidence': cached_response['confidence']})
        
        return QueryResponse(
            **cached_response,
            session_id=session_id,
            cached=True
        )
    
    start_time = time.time()
    
    try:
        # Add user query to memory
        memory.add_message("user", request.query)
        
        # Check if follow-up question - add context
        query_with_context = request.query
        if memory.has_context() and memory.is_follow_up_question(request.query):
            context = memory.get_context(num_messages=3)
            query_with_context = f"Context:\n{context}\n\nCurrent question: {request.query}"
            logger.info("Follow-up question detected, adding conversation context")
        
        # Retrieve documents
        retrieval_start = time.time()
        docs = []
        if retriever:
            try:
                raw_docs = retriever.get_relevant_documents(query_with_context)
                docs = [
                    Document(
                        content=doc.page_content,
                        source=doc.metadata.get("source", "unknown"),
                        chunk_index=doc.metadata.get("chunk_index", 0)
                    )
                    for doc in raw_docs[:request.top_k]
                ]
            except Exception as e:
                logger.warning(f"Retrieval failed: {e}")
        retrieval_time = time.time() - retrieval_start
        
        # Generate answer
        generation_start = time.time()
        if request.use_workflow:
            try:
                from src.orchestrator.graph import run_support_workflow
                result = run_support_workflow(request.query, retriever, orchestrator.llm)
                answer = result.get("answer", "Workflow execution failed")
                confidence = result.get("confidence", 0.0)
                should_escalate = result.get("escalate", False)
            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")
                answer = orchestrator.answer(request.query)
                confidence = orchestrator.classify_confidence(request.query, answer)
                should_escalate = orchestrator.should_escalate(request.query, answer)
        else:
            answer = orchestrator.answer(request.query)
            confidence = orchestrator.classify_confidence(request.query, answer)
            should_escalate = orchestrator.should_escalate(request.query, answer)
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # Add assistant response to memory
        memory.add_message("assistant", answer, {
            'confidence': confidence,
            'escalate': should_escalate,
            'num_documents': len(docs)
        })
        
        # Log metrics
        log_query_metrics(
            logger, request.query, answer, 
            retrieval_time, generation_time, 
            len(docs), confidence
        )
        
        response_data = {
            "answer": answer,
            "confidence": confidence,
            "should_escalate": should_escalate,
            "documents": docs,
            "session_id": session_id,
            "cached": False,
            "metrics": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "num_documents": len(docs),
                "workflow_used": request.use_workflow,
                "conversation_turns": len(memory.get_history())
            }
        }
        
        # Cache the response (only non-escalated, high confidence responses)
        if query_cache and confidence >= 0.6 and not should_escalate:
            query_cache.set(request.query, response_data)
            logger.info("Response cached for future queries")
        
        return QueryResponse(**response_data)
    
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for a query response.
    
    Args:
        feedback: Feedback data including rating and optional comment
        
    Returns:
        Success status
    """
    if not feedback_collector:
        raise HTTPException(status_code=503, detail="Feedback system not initialized")
    
    try:
        feedback_collector.record_feedback(
            query=feedback.query,
            answer=feedback.answer,
            rating=feedback.rating,
            comment=feedback.comment,
            session_id=feedback.session_id
        )
        
        logger.info(f"Feedback recorded: {feedback.rating}/5 stars")
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!",
            "rating": feedback.rating
        }
    
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/feedback")
async def get_feedback_stats():
    """Get feedback statistics and insights.
    
    Returns:
        Feedback statistics including ratings, suggestions, etc.
    """
    if not feedback_collector:
        raise HTTPException(status_code=503, detail="Feedback system not initialized")
    
    try:
        stats = feedback_collector.get_feedback_stats()
        suggestions = feedback_collector.get_improvement_suggestions()
        popular = query_cache.get_popular_queries(top_n=5) if query_cache else []
        
        return {
            "stats": stats,
            "suggestions": suggestions,
            "popular_queries": [
                {"query": query, "hits": hits}
                for query, hits in popular
            ]
        }
    
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/cache")
async def get_cache_stats():
    """Get cache performance statistics.
    
    Returns:
        Cache statistics including hit rate, size, etc.
    """
    if not query_cache:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    
    try:
        stats = query_cache.get_stats()
        popular = query_cache.get_popular_queries(top_n=10)
        
        return {
            "cache_stats": stats,
            "most_cached_queries": [
                {"query": query, "hits": hits}
                for query, hits in popular
            ]
        }
    
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear")
async def clear_cache():
    """Clear the query cache.
    
    Returns:
        Success status
    """
    if not query_cache:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    
    try:
        query_cache.clear()
        logger.info("Cache cleared successfully")
        
        return {
            "status": "success",
            "message": "Cache cleared successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Conversation history
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        memory = sessions[session_id]
        return {
            "session_id": session_id,
            "history": memory.get_history(),
            "metadata": memory.metadata,
            "total_messages": len(memory.get_history())
        }
    
    except Exception as e:
        logger.error(f"Failed to get session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success status
    """
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"Session {session_id} cleared")
    
    return {
        "status": "success",
        "message": f"Session {session_id} cleared"
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest documents into the vector store.
    
    Args:
        request: Ingestion request with optional directory path
        background_tasks: FastAPI background tasks
    
    Returns:
        IngestResponse with status and document count
    """
    data_dir = request.data_directory or str(config.data_dir)
    
    logger.info(f"Starting ingestion from: {data_dir}")
    
    def run_ingestion():
        try:
            ingest_from_directory(
                data_dir=data_dir,
                persist_directory=config.vector_store.persist_directory,
                chunk_size=config.vector_store.chunk_size,
                chunk_overlap=config.vector_store.chunk_overlap
            )
            logger.info("Ingestion completed successfully")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
    
    # Run ingestion in background
    background_tasks.add_task(run_ingestion)
    
    return IngestResponse(
        status="processing",
        message=f"Ingestion started for directory: {data_dir}",
        documents_processed=0
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.api_reload
    )
