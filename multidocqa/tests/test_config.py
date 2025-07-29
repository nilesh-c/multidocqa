"""
Unit tests for the configuration module.

Tests centralized configuration management and YAML loading.
"""

import tempfile
import unittest
from pathlib import Path

from multidocqa.config import (
    load_config,
    load_config_from_yaml,
)


class TestYAMLConfiguration(unittest.TestCase):
    """Test YAML configuration loading."""

    def setUp(self) -> None:
        """Set up test fixtures with temporary directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_load_config_from_yaml_success(self) -> None:
        """Test successful YAML configuration loading."""
        yaml_content = """
training:
  learning_rate: 0.001
  batch_size: 4
  max_steps: 1000

data:
  negative_ratio: 5
  balance_classes: false

model:
  model_name: "test/model"
  output_dir: "/tmp/test"

experiment:
  experiment_name: "test_experiment"
  run_name: "test_run"
"""

        yaml_file = self.config_dir / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        config = load_config_from_yaml(str(yaml_file))

        # Check that YAML values override defaults
        self.assertEqual(config.training.learning_rate, 0.001)
        self.assertEqual(config.training.batch_size, 4)
        self.assertEqual(config.training.max_steps, 1000)

        self.assertEqual(config.data.negative_ratio, 5)
        self.assertFalse(config.data.balance_classes)

        self.assertEqual(config.model.model_name, "test/model")
        self.assertEqual(config.model.output_dir, "/tmp/test")

        self.assertEqual(config.experiment.experiment_name, "test_experiment")
        self.assertEqual(config.experiment.run_name, "test_run")

    def test_load_config_from_yaml_partial(self) -> None:
        """Test YAML loading with partial configuration."""
        yaml_content = """
training:
  learning_rate: 0.002

model:
  model_name: "partial/model"
"""

        yaml_file = self.config_dir / "partial_config.yaml"
        yaml_file.write_text(yaml_content)

        config = load_config_from_yaml(str(yaml_file))

        # Check overridden values
        self.assertEqual(config.training.learning_rate, 0.002)
        self.assertEqual(config.model.model_name, "partial/model")

        # Check that defaults are preserved
        self.assertEqual(config.training.batch_size, 1)  # Default
        self.assertEqual(config.data.negative_ratio, 3)  # Default
        self.assertEqual(
            config.experiment.experiment_name, "legal_reasoning"
        )  # Default

    def test_load_config_from_yaml_empty(self) -> None:
        """Test YAML loading with empty file uses defaults."""
        yaml_file = self.config_dir / "empty_config.yaml"
        yaml_file.write_text("")

        config = load_config_from_yaml(str(yaml_file))

        # All values should be defaults
        self.assertEqual(config.training.learning_rate, 0.0001)
        self.assertEqual(config.model.model_name, "microsoft/DialoGPT-medium")
        self.assertEqual(config.data.negative_ratio, 3)
        self.assertEqual(config.experiment.experiment_name, "legal_reasoning")

    def test_load_config_from_yaml_file_not_found(self) -> None:
        """Test YAML loading with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_config_from_yaml("nonexistent.yaml")

    def test_load_config_with_yaml_path(self) -> None:
        """Test load_config with YAML path parameter."""
        yaml_content = """
training:
  learning_rate: 0.003
"""

        yaml_file = self.config_dir / "load_test.yaml"
        yaml_file.write_text(yaml_content)

        config = load_config(str(yaml_file))
        self.assertEqual(config.training.learning_rate, 0.003)


if __name__ == "__main__":
    unittest.main()
