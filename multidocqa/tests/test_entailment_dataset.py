"""
Unit tests for the EntailmentDatasetBuilder and related functions.

Tests dataset construction, class balancing, and prompt generation
for legal entailment prediction.
"""

import json
import os
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from multidocqa.entailment_dataset import EntailmentDatasetBuilder


class TestEntailmentDatasetBuilder(unittest.TestCase):
    """Test cases for EntailmentDatasetBuilder."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create sample training data
        self.sample_train_data = [
            {
                "id": "case1",
                "statement": "Can A rescind the contract?",
                "articles": [
                    {"number": "101", "content": "Content of article 101"},
                    {"number": "96", "content": "Content of article 96"},
                ],
                "label": "Y",
            },
            {
                "id": "case2",
                "statement": "Is the contract valid?",
                "articles": [{"number": "102", "content": "Content of article 102"}],
                "label": "N",
            },
            {
                "id": "case3",
                "statement": "Does B have authority?",
                "articles": [
                    {"number": "101", "content": "Content of article 101"},
                    {"number": "103", "content": "Content of article 103"},
                ],
                "label": "Y",
            },
            {
                "id": "case4",
                "statement": "Are the damages recoverable?",
                "articles": [
                    {"number": "104", "content": "Content of article 104"},
                    {"number": "105", "content": "Content of article 105"},
                    {"number": "106", "content": "Content of article 106"},
                ],
                "label": "N",
            },
        ]

        # Create sample civil code
        self.sample_civil_code = [
            {"number": "101", "content": "Content of article 101"},
            {"number": "96", "content": "Content of article 96"},
            {"number": "102", "content": "Content of article 102"},
            {"number": "103", "content": "Content of article 103"},
            {"number": "104", "content": "Content of article 104"},
            {"number": "105", "content": "Content of article 105"},
            {"number": "106", "content": "Content of article 106"},
        ]

        # Create temporary files
        self.train_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        self.civil_code_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )

        json.dump(self.sample_train_data, self.train_file)
        json.dump(self.sample_civil_code, self.civil_code_file)

        self.train_file.close()
        self.civil_code_file.close()

        # Mock the load functions to use our test data
        self.load_data_patcher = patch("multidocqa.entailment_dataset.load_data")

        self.mock_load_data = self.load_data_patcher.start()

        # Create combined dataset structure that EntailmentDatasetBuilder expects
        combined_data = {
            "entailment_pairs": self.sample_train_data,
            "articles": self.sample_civil_code,
        }
        self.mock_load_data.return_value = combined_data

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        os.unlink(self.train_file.name)
        os.unlink(self.civil_code_file.name)
        self.load_data_patcher.stop()

    def test_init(self) -> None:
        """Test EntailmentDatasetBuilder initialization."""
        builder = EntailmentDatasetBuilder(self.train_file.name, balance_classes=True)

        self.assertEqual(builder.balance_classes, True)
        self.assertEqual(len(builder.train_data), 4)
        self.assertEqual(len(builder.civil_code), 7)
        self.assertEqual(len(builder.article_lookup), 7)

        # Check class distribution
        expected_distribution = {"Y": 2, "N": 2}
        self.assertEqual(builder.class_distribution, expected_distribution)

    def test_init_no_balance(self) -> None:
        """Test EntailmentDatasetBuilder initialization without class balancing."""
        builder = EntailmentDatasetBuilder(self.train_file.name, balance_classes=False)

        self.assertEqual(builder.balance_classes, False)
        self.assertEqual(len(builder.train_data), 4)

    def test_analyze_class_distribution(self) -> None:
        """Test class distribution analysis."""
        builder = EntailmentDatasetBuilder(self.train_file.name, balance_classes=False)

        distribution = builder._analyze_class_distribution()

        expected_distribution = {"Y": 2, "N": 2}
        self.assertEqual(distribution, expected_distribution)

    def test_generate_entailment_samples(self) -> None:
        """Test entailment sample generation."""
        builder = EntailmentDatasetBuilder(self.train_file.name, balance_classes=False)

        samples = builder._generate_entailment_samples()

        # Should have 4 samples (one per training item)
        self.assertEqual(len(samples), 4)

        # Check sample structure
        for sample in samples:
            self.assertIn("statement", sample)
            self.assertIn("statement_id", sample)
            self.assertIn("articles", sample)
            self.assertIn("label", sample)
            self.assertIn("sample_type", sample)
            self.assertEqual(sample["sample_type"], "original")

        # Check specific samples
        sample_ids = {sample["statement_id"] for sample in samples}
        expected_ids = {"case1", "case2", "case3", "case4"}
        self.assertEqual(sample_ids, expected_ids)

        # Check article counts
        article_counts = [len(sample["articles"]) for sample in samples]
        expected_counts = [2, 1, 2, 3]  # case1, case2, case3, case4
        self.assertEqual(sorted(article_counts), sorted(expected_counts))

    def test_balance_classes(self) -> None:
        """Test class balancing functionality."""
        builder = EntailmentDatasetBuilder(self.train_file.name, balance_classes=True)

        # Create unbalanced samples
        unbalanced_samples = [
            {"label": "Y", "id": "1"},
            {"label": "Y", "id": "2"},
            {"label": "Y", "id": "3"},
            {"label": "N", "id": "4"},
        ]

        balanced_samples = builder._balance_classes(unbalanced_samples)

        # Should have equal numbers of Y and N samples
        y_count = sum(1 for s in balanced_samples if s["label"] == "Y")
        n_count = sum(1 for s in balanced_samples if s["label"] == "N")

        self.assertEqual(y_count, n_count)
        self.assertEqual(len(balanced_samples), 2)  # min(3, 1) * 2 = 2

    def test_balance_classes_equal_distribution(self) -> None:
        """Test class balancing with already balanced data."""
        builder = EntailmentDatasetBuilder(self.train_file.name, balance_classes=True)

        # Create balanced samples
        balanced_samples = [
            {"label": "Y", "id": "1"},
            {"label": "Y", "id": "2"},
            {"label": "N", "id": "3"},
            {"label": "N", "id": "4"},
        ]

        result = builder._balance_classes(balanced_samples)

        # Should maintain balance
        y_count = sum(1 for s in result if s["label"] == "Y")
        n_count = sum(1 for s in result if s["label"] == "N")

        self.assertEqual(y_count, n_count)
        self.assertEqual(len(result), 4)

    def test_create_entailment_prompt(self) -> None:
        """Test entailment prompt creation."""
        builder = EntailmentDatasetBuilder(self.train_file.name, balance_classes=False)

        test_sample = {
            "statement": "Can A rescind the contract?",
            "statement_id": "case1",
            "articles": [
                {"number": "101", "content": "Content of article 101"},
                {"number": "96", "content": "Content of article 96"},
            ],
            "label": "Y",
            "sample_type": "original",
        }

        prompt_data = builder._create_entailment_prompt(test_sample)

        # Check structure
        required_keys = [
            "prompt",
            "reference",
            "statement",
            "statement_id",
            "articles",
            "sample_type",
            "num_articles",
        ]
        for key in required_keys:
            self.assertIn(key, prompt_data)

        # Check content
        self.assertEqual(prompt_data["reference"], "Y")
        self.assertEqual(prompt_data["statement"], "Can A rescind the contract?")
        self.assertEqual(prompt_data["statement_id"], "case1")
        self.assertEqual(prompt_data["sample_type"], "original")
        self.assertEqual(prompt_data["num_articles"], 2)

        # Check prompt contains key elements
        prompt = prompt_data["prompt"]
        self.assertIn("Article 101", prompt)
        self.assertIn("Content of article 101", prompt)
        self.assertIn("Article 96", prompt)
        self.assertIn("Content of article 96", prompt)
        self.assertIn("Can A rescind the contract?", prompt)
        self.assertIn("Y", prompt)
        self.assertIn("N", prompt)

    def test_build_entailment_dataset_balanced(self) -> None:
        """Test complete dataset building with class balancing."""
        builder = EntailmentDatasetBuilder(self.train_file.name, balance_classes=True)

        train_dataset, val_dataset = builder.build_entailment_dataset(split_ratio=0.8)

        # Should have balanced classes (2 Y + 2 N = 4 samples)
        total_samples = len(train_dataset) + len(val_dataset)
        self.assertEqual(total_samples, 4)

        # Check split ratio approximately correct
        train_ratio = len(train_dataset) / total_samples
        self.assertAlmostEqual(train_ratio, 0.8, delta=0.2)

        # Check dataset structure
        sample = train_dataset[0]
        required_keys = [
            "prompt",
            "reference",
            "statement",
            "statement_id",
            "articles",
            "sample_type",
            "num_articles",
        ]
        for key in required_keys:
            self.assertIn(key, sample)

        # Check class balance in combined dataset
        all_samples = list(train_dataset) + list(val_dataset)
        y_count = sum(1 for s in all_samples if s["reference"] == "Y")
        n_count = sum(1 for s in all_samples if s["reference"] == "N")
        self.assertEqual(y_count, n_count)

    def test_build_entailment_dataset_unbalanced(self) -> None:
        """Test complete dataset building without class balancing."""
        builder = EntailmentDatasetBuilder(self.train_file.name, balance_classes=False)

        train_dataset, val_dataset = builder.build_entailment_dataset(split_ratio=0.8)

        # Should have all original samples (4 samples)
        total_samples = len(train_dataset) + len(val_dataset)
        self.assertEqual(total_samples, 4)

        # Check class distribution matches original
        all_samples = list(train_dataset) + list(val_dataset)
        y_count = sum(1 for s in all_samples if s["reference"] == "Y")
        n_count = sum(1 for s in all_samples if s["reference"] == "N")
        self.assertEqual(y_count, 2)
        self.assertEqual(n_count, 2)

    def test_get_dataset_statistics(self) -> None:
        """Test dataset statistics computation."""
        builder = EntailmentDatasetBuilder(self.train_file.name, balance_classes=False)

        stats = builder.get_dataset_statistics()

        # Check basic statistics
        self.assertEqual(stats["total_samples"], 4)
        self.assertEqual(stats["y_samples"], 2)
        self.assertEqual(stats["n_samples"], 2)
        self.assertEqual(stats["class_ratio"], "2:2")
        self.assertEqual(stats["balance_classes"], False)

        # Check that statistics are computed without errors
        self.assertIsInstance(stats["avg_articles_per_statement"], float)
        self.assertIsInstance(stats["min_articles_per_statement"], int)
        self.assertIsInstance(stats["max_articles_per_statement"], int)
        self.assertIsInstance(stats["unique_articles_used"], int)
        self.assertIsInstance(stats["total_civil_code_articles"], int)

    @patch("multidocqa.entailment_dataset.random.shuffle")
    def test_dataset_shuffling(self, mock_shuffle: Any) -> None:
        """Test that dataset samples are shuffled."""
        builder = EntailmentDatasetBuilder(self.train_file.name, balance_classes=False)

        builder.build_entailment_dataset()

        # Verify shuffle was called
        mock_shuffle.assert_called_once()

    def test_article_lookup_functionality(self) -> None:
        """Test article lookup functionality."""
        builder = EntailmentDatasetBuilder(self.train_file.name, balance_classes=False)

        # Check that all articles are in lookup
        for article in self.sample_civil_code:
            self.assertIn(article["number"], builder.article_lookup)
            self.assertEqual(
                builder.article_lookup[article["number"]]["content"], article["content"]
            )

    def test_empty_dataset_handling(self) -> None:
        """Test handling of empty dataset."""
        # Create empty dataset
        empty_data: dict = {"entailment_pairs": [], "articles": []}

        with patch("multidocqa.entailment_dataset.load_data") as mock_load:
            mock_load.return_value = empty_data

            builder = EntailmentDatasetBuilder("dummy_path", balance_classes=False)

            self.assertEqual(len(builder.train_data), 0)
            self.assertEqual(len(builder.civil_code), 0)
            self.assertEqual(len(builder.article_lookup), 0)

            # Test that empty dataset doesn't crash
            samples = builder._generate_entailment_samples()
            self.assertEqual(len(samples), 0)


if __name__ == "__main__":
    unittest.main()
