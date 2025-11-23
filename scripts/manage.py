#!/usr/bin/env python
"""Management CLI for Customer Support Orchestrator.

Provides commands for ingestion, testing, and maintenance.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.logging_config import setup_logging
from src.orchestrator.ingest import ingest_from_directory
from src.orchestrator.retriever import get_retriever
from src.orchestrator.agents import SupportOrchestrator

logger = setup_logging(config.log_level)


def cmd_ingest(args):
    """Ingest documents into the vector store."""
    logger.info(f"Ingesting documents from: {args.data_dir}")
    
    try:
        ingest_from_directory(
            data_dir=args.data_dir,
            persist_directory=args.persist_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        logger.info("✅ Ingestion completed successfully")
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        sys.exit(1)


def cmd_query(args):
    """Test a query against the system."""
    logger.info(f"Processing query: {args.query}")
    
    try:
        retriever = get_retriever(persist_directory=args.persist_dir)
        orchestrator = SupportOrchestrator(
            retriever=retriever,
            mode=args.mode,
            hf_token=config.model.hf_token,
            hf_model=config.model.hf_model
        )
        
        # Get answer
        answer = orchestrator.answer(args.query)
        confidence = orchestrator.classify_confidence(args.query, answer)
        should_escalate = orchestrator.should_escalate(args.query, answer)
        
        # Display results
        print("\n" + "="*60)
        print("QUERY:", args.query)
        print("="*60)
        print("\nANSWER:")
        print(answer)
        print(f"\nCONFIDENCE: {confidence:.2%}")
        print(f"ESCALATE: {'Yes' if should_escalate else 'No'}")
        
        # Show retrieved documents
        if args.show_docs:
            print("\n" + "-"*60)
            print("RETRIEVED DOCUMENTS:")
            print("-"*60)
            docs = retriever.get_relevant_documents(args.query)
            for i, doc in enumerate(docs[:args.top_k], 1):
                print(f"\n[{i}] {doc.metadata.get('source', 'unknown')}")
                print(doc.page_content[:200] + "...")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        sys.exit(1)


def cmd_test(args):
    """Run system tests."""
    logger.info("Running system tests...")
    
    test_queries = [
        "How do I reset my password?",
        "What are your business hours?",
        "I need to speak to a manager",
        "How do I cancel my subscription?"
    ]
    
    try:
        retriever = get_retriever(persist_directory=args.persist_dir)
        orchestrator = SupportOrchestrator(
            retriever=retriever,
            mode=args.mode,
            hf_token=config.model.hf_token,
            hf_model=config.model.hf_model
        )
        
        results = []
        for query in test_queries:
            answer = orchestrator.answer(query)
            confidence = orchestrator.classify_confidence(query, answer)
            results.append({
                "query": query,
                "answer_length": len(answer),
                "confidence": confidence
            })
        
        # Display summary
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['query']}")
            print(f"   Answer Length: {result['answer_length']} chars")
            print(f"   Confidence: {result['confidence']:.2%}")
        
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"\nAverage Confidence: {avg_confidence:.2%}")
        print("="*60)
        
        logger.info("✅ Tests completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Tests failed: {e}")
        sys.exit(1)


def cmd_clear(args):
    """Clear the vector store."""
    import shutil
    
    persist_dir = Path(args.persist_dir)
    
    if not persist_dir.exists():
        logger.info(f"Directory {persist_dir} does not exist, nothing to clear")
        return
    
    if args.confirm or input(f"Clear {persist_dir}? (yes/no): ").lower() == "yes":
        shutil.rmtree(persist_dir)
        logger.info(f"✅ Cleared {persist_dir}")
    else:
        logger.info("Cancelled")


def cmd_info(args):
    """Display system information."""
    print("\n" + "="*60)
    print("CUSTOMER SUPPORT ORCHESTRATOR - SYSTEM INFO")
    print("="*60)
    
    print(f"\n📁 Project Root: {config.project_root}")
    print(f"📂 Data Directory: {config.data_dir}")
    print(f"💾 Vector Store: {config.vector_store.persist_directory}")
    
    print(f"\n🤖 Model Configuration:")
    print(f"   Mode: {config.model.mode}")
    print(f"   HF Model: {config.model.hf_model}")
    print(f"   Embedding Model: {config.model.embedding_model}")
    print(f"   Temperature: {config.model.temperature}")
    print(f"   Max Tokens: {config.model.max_tokens}")
    
    print(f"\n⚙️ Vector Store Configuration:")
    print(f"   Collection: {config.vector_store.collection_name}")
    print(f"   Chunk Size: {config.vector_store.chunk_size}")
    print(f"   Chunk Overlap: {config.vector_store.chunk_overlap}")
    print(f"   Top K: {config.vector_store.top_k}")
    
    print(f"\n🔧 Orchestrator Configuration:")
    print(f"   Confidence Threshold: {config.orchestrator.confidence_threshold}")
    print(f"   Max Retries: {config.orchestrator.max_retries}")
    print(f"   Timeout: {config.orchestrator.timeout_seconds}s")
    
    # Validation
    issues = config.validate()
    if issues:
        print(f"\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"\n✅ Configuration is valid")
    
    # Check data directory
    if config.data_dir.exists():
        data_files = list(config.data_dir.glob("*.md"))
        print(f"\n📄 Data Files: {len(data_files)}")
        for f in data_files:
            print(f"   - {f.name}")
    else:
        print(f"\n❌ Data directory not found")
    
    # Check vector store
    persist_path = Path(config.vector_store.persist_directory)
    if persist_path.exists():
        print(f"\n💾 Vector Store: Initialized")
    else:
        print(f"\n⚠️  Vector Store: Not initialized (run 'manage.py ingest')")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Customer Support Orchestrator Management CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--data-dir", default=str(config.data_dir), help="Data directory")
    ingest_parser.add_argument("--persist-dir", default=config.vector_store.persist_directory, help="Vector store directory")
    ingest_parser.add_argument("--chunk-size", type=int, default=config.vector_store.chunk_size, help="Chunk size")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=config.vector_store.chunk_overlap, help="Chunk overlap")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Test a query")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--mode", choices=["local", "hf"], default=config.model.mode, help="LLM mode")
    query_parser.add_argument("--persist-dir", default=config.vector_store.persist_directory, help="Vector store directory")
    query_parser.add_argument("--show-docs", action="store_true", help="Show retrieved documents")
    query_parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run system tests")
    test_parser.add_argument("--mode", choices=["local", "hf"], default=config.model.mode, help="LLM mode")
    test_parser.add_argument("--persist-dir", default=config.vector_store.persist_directory, help="Vector store directory")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear vector store")
    clear_parser.add_argument("--persist-dir", default=config.vector_store.persist_directory, help="Vector store directory")
    clear_parser.add_argument("--confirm", action="store_true", help="Skip confirmation")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Display system information")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to command handler
    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "test": cmd_test,
        "clear": cmd_clear,
        "info": cmd_info
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
