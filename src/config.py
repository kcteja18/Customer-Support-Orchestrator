"""Configuration management for the Customer Support Orchestrator."""
import os
from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """LLM and embedding model configuration."""
    mode: Literal["local", "hf"] = "local"
    hf_model: str = "google/flan-t5-base"
    hf_token: str = field(default_factory=lambda: os.getenv("HUGGINGFACE_API_TOKEN", ""))
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    persist_directory: str = ".chroma"
    collection_name: str = "support_docs"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3


@dataclass
class OrchestatorConfig:
    """Orchestrator behavior configuration."""
    confidence_threshold: float = 0.7
    escalation_keywords: list[str] = field(default_factory=lambda: [
        "speak to manager", "supervisor", "human agent", "escalate", "complaint"
    ])
    max_retries: int = 3
    timeout_seconds: int = 30


@dataclass
class AppConfig:
    """Main application configuration."""
    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(init=False)
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    orchestrator: OrchestatorConfig = field(default_factory=OrchestatorConfig)
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "app.log"
    
    def __post_init__(self):
        """Initialize derived paths."""
        self.data_dir = self.project_root / "examples" / "data"
        
        # Load from environment if available
        if mode := os.getenv("LLM_MODE"):
            self.model.mode = mode
        if hf_model := os.getenv("HF_MODEL"):
            self.model.hf_model = hf_model
        if log_level := os.getenv("LOG_LEVEL"):
            self.log_level = log_level
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if self.model.mode == "hf" and not self.model.hf_token:
            issues.append("HuggingFace mode requires HUGGINGFACE_API_TOKEN environment variable")
        
        if not self.data_dir.exists():
            issues.append(f"Data directory does not exist: {self.data_dir}")
        
        if self.orchestrator.confidence_threshold < 0 or self.orchestrator.confidence_threshold > 1:
            issues.append("Confidence threshold must be between 0 and 1")
        
        return issues


# Global config instance
config = AppConfig()
