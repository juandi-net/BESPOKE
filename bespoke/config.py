"""Central configuration for BESPOKE."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class LLMConfig:
    """LLM API configuration. Swap providers by changing base_url."""
    base_url: str = os.environ.get("CLIPROXY_BASE_URL", "http://localhost:8317/v1")
    api_key: str = os.environ.get("CLIPROXY_API_KEY", "not-needed")

    # Model assignments per stage (CLIProxy model IDs)
    model_stage_2a: str = "claude-sonnet-4-6"   # Nightly extraction
    model_stage_2b: str = "claude-opus-4-6"              # Weekly pattern mining
    model_benchmark: str = "claude-opus-4-6"              # Benchmark interview + analysis
    model_judge: str = "claude-sonnet-4-6"      # LLM-as-judge evaluation


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_path: Path = Path.home() / ".bespoke" / "models" / "embeddinggemma-300m"
    onnx_path: Path = field(default=None)
    dimension: int = 768

    def __post_init__(self):
        if self.onnx_path is None:
            self.onnx_path = self.model_path / "onnx" / "model.onnx"


@dataclass
class BaseModelConfig:
    """Base model configuration for training and inference."""
    # Training (MLX format)
    training_model_path: Path = Path.home() / ".bespoke" / "models" / "qwen3.5-4b-mlx"

    # Inference (GGUF format) — source: unsloth/Qwen3.5-4B-GGUF on HuggingFace
    inference_model_path: Path = Path.home() / ".bespoke" / "models" / "qwen3.5-4b-gguf" / "Qwen3.5-4B-Q4_K_M.gguf"

    # Serving
    llama_server_port: int = 8080
    context_size: int = 4096  # Start conservative, increase as needed
    gpu_layers: int = 99      # Offload everything to GPU


@dataclass
class TrainingConfig:
    """Default training hyperparameters."""
    # SFT phase
    sft_learning_rate: float = 2e-4
    sft_rank: int = 16
    sft_epochs: int = 2
    sft_batch_size: int = 4

    # DPO phase
    dpo_learning_rate: float = 5e-6
    dpo_rank: int = 8
    dpo_epochs: int = 1
    dpo_batch_size: int = 2

    # Adapter configuration
    use_dora: bool = True
    use_rslora: bool = True
    lora_plus_ratio: float = 10.0  # B matrix LR = A matrix LR * this

    # Target modules (attention layers for V0)
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # Training time budget (minutes)
    time_budget_minutes: int = 30


@dataclass
class PipelineConfig:
    """Pipeline scheduling and thresholds."""
    # Capture
    capture_interval_minutes: int = 10
    # No token threshold — short interactions ("ok", "do it") are accept/reject
    # signals needed for Stage 2a feedback classification. Capture everything.

    # Stage 2a
    stage_2a_schedule: str = "23:00"  # 11 PM

    # Stage 3
    stage_3_schedule: str = "00:00"   # Midnight
    min_new_examples_for_training: int = 10  # Skip training if fewer new examples

    # Stage 2b
    stage_2b_schedule: str = "weekly"  # Sunday 2 AM

    # Quality thresholds
    min_quality_for_sft: str = "medium"   # Include medium and high
    min_quality_for_dpo: str = "high"     # DPO only on high-quality pairs


@dataclass
class BespokeConfig:
    """Root configuration."""
    db_path: Path = Path.home() / ".bespoke" / "bespoke.db"
    adapters_dir: Path = Path.home() / ".bespoke" / "adapters"
    scorecards_dir: Path = Path.home() / ".bespoke" / "scorecards"
    benchmark_dir: Path = Path.home() / ".bespoke" / "benchmark"
    backups_dir: Path = Path.home() / ".bespoke" / "backups"

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    base_model: BaseModelConfig = field(default_factory=BaseModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "BespokeConfig":
        """Load config from YAML file, falling back to defaults."""
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # TODO: merge YAML data with defaults
        return cls()

    def ensure_dirs(self):
        """Create all required directories."""
        for d in [self.adapters_dir, self.scorecards_dir,
                  self.benchmark_dir, self.backups_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Global config instance
config = BespokeConfig()
config.ensure_dirs()
