"""
Centralized configuration management for the multidocqa package.

This module provides dataclasses for all configuration parameters,
eliminating hardcoded values scattered across training scripts.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training parameters."""

    # Core training settings
    learning_rate: float = 0.0001
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 500
    warmup_steps: int = 50

    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50

    # Model settings
    max_length: int = 2048
    bf16: bool = False


@dataclass
class DataConfig:
    """Configuration for dataset building and processing."""

    # Dataset building
    negative_ratio: int = 3
    split_ratio: float = 0.9
    balance_classes: bool = True

    # Data processing
    max_sequence_length: int = 2048
    shuffle_data: bool = True

    # Data paths
    combined_dataset_path: str = "data/processed/combined_train.json"


@dataclass
class ModelConfig:
    """Configuration for model parameters and paths."""

    # Model paths and names
    model_name: str = "microsoft/DialoGPT-medium"
    tokenizer_name: Optional[str] = None  # If None, uses model_name
    output_dir: str = "./checkpoints"

    # Model loading
    trust_remote_code: bool = False
    use_flash_attention: bool = False
    use_peft: bool = False

    # Generation settings
    max_new_tokens: int = 10
    temperature: float = 0.1
    do_sample: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and metadata."""

    # Experiment tracking
    experiment_name: str = "legal_reasoning"
    run_name: Optional[str] = None

    # Dataset metadata
    dataset_version: str = "coliee2025"
    legal_domain: str = "civil_law"
    jurisdiction: str = "japanese_civil_code"

    # Evaluation
    eval_batch_size: int = 4
    eval_max_samples: Optional[int] = None


@dataclass
class Config:
    """Unified configuration container for all config sections."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def load_config_from_yaml(config_path: str) -> Config:
    """
    Load unified configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Config object with all configuration sections
    """

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        logger.info(f"Loaded configuration from YAML: {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML configuration: {e}")
        raise

    # Create unified config with YAML data, falling back to defaults
    config = Config(
        training=TrainingConfig(**config_data.get("training", {})),
        data=DataConfig(**config_data.get("data", {})),
        model=ModelConfig(**config_data.get("model", {})),
        experiment=ExperimentConfig(**config_data.get("experiment", {})),
    )

    logger.info(
        f"Created configuration from YAML: model={config.model.model_name}, "
        f"lr={config.training.learning_rate}, "
        f"batch_size={config.training.batch_size}"
    )

    return config


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load unified configuration from YAML file or use defaults.

    Args:
        config_path: Optional path to YAML configuration file. If not provided,
                    uses default values from dataclasses.

    Returns:
        Config object with all configuration sections
    """
    if config_path:
        return load_config_from_yaml(config_path)

    logger.info("Using default configuration values")

    config = Config()

    logger.info(
        f"Loaded configuration: model={config.model.model_name}, "
        f"lr={config.training.learning_rate}, "
        f"batch_size={config.training.batch_size}"
    )

    return config
