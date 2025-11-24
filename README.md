# Customer Support Orchestrator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**AI-powered customer support system with RAG, conversation memory, intelligent caching, and feedback analytics**

</div>

---

## 🎯 Overview

Production-ready customer support system featuring:

-  **Conversation Memory** - Multi-turn dialogues with context awareness
-  **Query Caching** - Faster responses for repeated questions
-  **Feedback Analytics** - User ratings and improvement insights
-  **Smart Routing** - LangGraph workflow with intent classification
-  **Knowledge Base** - 90+ support topics across 4 categories
-  **RAG Pipeline** - Semantic search with Chroma vector database
-  **Local Mode** - No OpenAI API required (uses MockLLM)

**Tech Stack:** Python 3.11+, FastAPI, LangChain, LangGraph, Streamlit, Chroma, sentence-transformers

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Streamlit UI (Port 8501)                   │
│   Chat • Feedback Buttons • Cache Indicators • Session ID    │
└────────────────────┬─────────────────────────────────────────┘
                     │ HTTP/REST (with session_id)
                     ▼
┌──────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (Port 8000)                  │
│  /query • /feedback • /analytics/* • /session/* • /cache/*   │
└────────────────────┬─────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┬─────────────┐
        ▼            ▼            ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐
   │  Query  │  │Conversation│ │Feedback │  │LangGraph│
   │  Cache  │  │  Memory   │ │Collector│  │Workflow │
   │  (LRU)  │  │(Sessions) │ │(Analytics)│ │ Router  │
   └────┬────┘  └─────┬────┘  └──────────┘  └────┬────┘
        │             │                           │
        │             │         ┌─────────────────┘
        │             │         ▼
        │             │    ┌──────────┐
        │             │    │Retriever │
        │             │    │ + LLM    │
        │             │    └────┬─────┘
        │             │         │
        └─────────────┴─────────┼─────────────┐
                                ▼             ▼
                          ┌──────────┐   ┌────────┐
                          │  Chroma  │   │MockLLM/│
                          │VectorDB  │   │   HF   │
                          └──────────┘   └────────┘
```

### **Request Flow with New Features**

```
User Query (with session_id)
    ↓
[1] Check Query Cache
    ├─→ Cache HIT → Return cached response (< 0.1s)
    └─→ Cache MISS → Continue
            ↓
[2] Get/Create Conversation Session
            ↓
[3] Check if Follow-up Question
    ├─→ Yes → Inject conversation context
    └─→ No → Process normally
            ↓
[4] Relevance Check
    ├─→ Out-of-scope → Return guidance message
    └─→ In-scope
            ↓
[5] Intent Classification (Workflow)
            ↓
[6] Document Retrieval (Vector DB)
            ↓
[7] Answer Generation (LLM)
            ↓
[8] Confidence Scoring
            ↓
[9] Should Cache?
    └─→ High confidence (≥0.6) + Not escalated → Add to cache
            ↓
[10] Update Conversation Memory
            ↓
[11] Return Answer + Metadata
            ↓
[Optional] User Provides Feedback
            ↓
[12] Store in Feedback Collector
            ↓
[13] Analytics & Improvement Insights
```

---

## 🚀 Quick Start


### **1. Clone and Install**

```bash
git clone https://github.com/kcteja18/Customer-Support-Orchestrator.git
cd Customer-Support-Orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### **2. Ingest Knowledge Base**

```bash
python scripts/manage.py ingest
```

### **3. Run the Application**

**Option A: Development Mode**

```bash
# Terminal 1: Start Backend
python backend/main.py

# Terminal 2: Start UI
streamlit run src/ui/app.py
```

**Option B: Using Docker**

```bash
docker-compose up --build
```

### **4. Access the Application**

- **UI**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 📖 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with cache stats |
| `/query` | POST | Process query with caching & memory |
| `/ingest` | POST | Ingest documents (background) |
| `/feedback` | POST | Submit user feedback |
| `/analytics/feedback` | GET | Get feedback analytics |
| `/analytics/cache` | GET | Get cache performance stats |
| `/cache/clear` | POST | Clear query cache |
| `/session/{id}/history` | GET | Get conversation history |
| `/session/{id}` | DELETE | Clear session |

---

## 🧪 Testing

```bash
# Quick test
python scripts/manage.py query "How do I reset my password?"

# Run all tests
python scripts/manage.py test

# Or use pytest
pytest tests/ -v
```

**Test Queries:**
- ✅ In-scope: "How do I reset my password?", "What payment methods?"
- ❌ Out-of-scope: "What's the weather?", "Tell me a joke"

---

## 📁 Project Structure

```
Customer-Support-Orchestrator/
├── backend/
│   └── main.py                   # FastAPI app with all endpoints
├── src/
│   ├── orchestrator/
│   │   ├── agents.py             # Support orchestrator & MockLLM
│   │   ├── cache.py              # Query caching (NEW)
│   │   ├── memory.py             # Conversation memory (NEW)
│   │   ├── feedback.py           # Feedback collector (NEW)
│   │   ├── graph.py              # LangGraph workflow
│   │   ├── retriever.py          # Chroma vector store
│   │   ├── ingest.py             # Document ingestion
│   │   └── embeddings.py         # sentence-transformers
│   └── ui/
│       └── app.py                # Streamlit UI
├── scripts/
│   └── manage.py                 # Management CLI
├── examples/data/                # Knowledge base (4 .md files)
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

Create `.env` file:

```env
LLM_MODE=local                           # or "hf" for HuggingFace
HUGGINGFACE_API_TOKEN=your_token_here    # Optional
LOG_LEVEL=INFO
```

**Customize in `src/config.py`:**
- Cache TTL (default: 60 minutes)
- Max cache size (default: 1000 entries)
- Conversation memory (default: 10 messages)
- Confidence threshold (default: 0.7)

---

## 🔄 Workflow

```
User Query → Cache Check → Session Management → Follow-up Detection
                ↓
     Intent Classification → Document Retrieval → Answer Generation
                ↓
     Confidence Scoring → Cache Decision → Update Memory → Return
```

**Workflow features:**
- Intent-based routing (technical, billing, general)
- Automatic escalation for low confidence (<40%)
- Out-of-scope detection
- Source document tracking

---

## 🛠️ Management CLI

```bash
# System information
python scripts/manage.py info

# Test query with documents
python scripts/manage.py query "How do I reset password?" --show-docs

# Clear and reingest documents
python scripts/manage.py clear --confirm
python scripts/manage.py ingest

# Run tests
python scripts/manage.py test
```

---

## 🐳 Docker Deployment

```bash
# Build and start
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 📊 Features

### Conversation Memory
- Maintains context across queries
- Detects follow-up questions
- Exports conversation history

### Query Caching
- LRU cache with TTL
- Query normalization
- Performance tracking

### Feedback System
- 5-star rating system
- Analytics dashboard
- Improvement suggestions

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Cache not working | Check cache initialization and confidence threshold (≥0.6) |
| Memory not persisting | Ensure consistent session IDs across requests |
| Feedback not saving | Verify `data/` directory exists and is writable |
| Vector store errors | Run `python scripts/manage.py clear --confirm` then reingest |
| Import errors | Install all dependencies: `pip install -r requirements.txt` |

---

## 🤝 Contributing

Contributions welcome! Feel free to:
- Fork and customize
- Add new features
- Improve documentation
- Submit pull requests

---

## 📝 License

MIT License - free to use in your own projects!

---

##  Acknowledgments

Built with: [LangChain](https://langchain.com/) • [LangGraph](https://langchain-ai.github.io/langgraph/) • [FastAPI](https://fastapi.tiangolo.com/) • [Streamlit](https://streamlit.io/) • [Chroma](https://www.trychroma.com/) • [sentence-transformers](https://www.sbert.net/)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**


</div>
