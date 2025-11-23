"""Demo runner for the Customer Support Orchestrator scaffold.

Usage:
    python scripts\run_demo.py --mode local --query "How do I reset my password?"
    python scripts\run_demo.py --mode hf --ingest
    python scripts\run_demo.py --mode local --query "..." --graph  # Use LangGraph workflow
"""
import argparse
import os
import sys

# Ensure project root is on sys.path so `from src...` imports work when running script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv()

from src.orchestrator.ingest import ingest_from_directory
from src.orchestrator.retriever import get_retriever
from src.orchestrator.agents import SupportOrchestrator, EscalationAgent, MockLLM
from src.orchestrator.local_retriever import LocalRetriever

try:
    from src.orchestrator.graph import run_support_workflow, _LANGGRAPH_AVAILABLE
except Exception:
    run_support_workflow = None
    _LANGGRAPH_AVAILABLE = False



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true", help="Ingest example data into Chroma")
    parser.add_argument("--query", type=str, help="Query to run against the orchestrator")
    parser.add_argument("--mode", type=str, choices=["local", "hf"], default="local", help="Mode to run the demo in: local or hf (Hugging Face)")
    parser.add_argument("--persist", type=str, default=".chroma", help="Chroma persistence directory")
    parser.add_argument("--graph", action="store_true", help="Use LangGraph workflow (if available)")
    args = parser.parse_args()

    # Mode-specific warnings about missing API keys
    if args.mode == "hf" and not (os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")):
        print("Warning: HUGGINGFACEHUB_API_TOKEN is not set. Set it in the environment or .env to run in hf mode.")

    if args.ingest:
        if args.mode == "local":
            print("Local mode: no external embeddings will be created — nothing to persist.")
        else:
            print("Ingesting data into Chroma (this may take a moment)...")
            ingest_from_directory(data_dir="examples/data", persist_directory=args.persist)
            print("Ingest complete. Persisted to", args.persist)

    if args.query:
        if args.mode == "local":
            retriever = LocalRetriever(data_dir="examples/data")
        else:
            retriever = get_retriever(persist_directory=args.persist)

        llm = MockLLM()
        
        # Use LangGraph workflow if requested and available
        if args.graph:
            if _LANGGRAPH_AVAILABLE and run_support_workflow:
                print("Running LangGraph workflow...")
                result = run_support_workflow(args.query, retriever, llm)
                print("\n=== Workflow Result ===")
                print(f"Query: {result['query']}")
                print(f"Intent: {result['intent']}")
                print(f"Answer: {result['answer']}")
                print(f"Confidence: {result['confidence']:.0%}")
                print(f"\nWorkflow Steps:")
                for msg in result['messages']:
                    print(f"  • {msg}")
                if result.get('escalate'):
                    print("\n⚠️ Escalation Required")
                    if result.get('ticket'):
                        print(f"Ticket: {result['ticket']}")
            else:
                print("LangGraph not available. Install with: pip install langgraph")
                print("Falling back to simple orchestrator...")
                args.graph = False
        
        # Fallback to simple orchestrator
        if not args.graph:
            orchestrator = SupportOrchestrator(retriever, mode=args.mode)
            print("Query:", args.query)
            result = orchestrator.answer(args.query)
            print("Answer:\n", result["answer"])
            if result.get("escalate"):
                print("Confidence low — creating escalation ticket...")
                docs = retriever.get_relevant_documents(args.query)
                context = "\n\n---\n\n".join(getattr(d, 'page_content', str(d)) for d in docs[:3])
                ticket = EscalationAgent.create_ticket(args.query, context=context)
                print("Created ticket (demo):", ticket)


if __name__ == "__main__":
    main()
