"""
Unit tests for the RelevanceDatasetBuilder and related functions.

Tests dataset construction, positive/negative sampling, and prompt generation
for legal relevance prediction.
"""

import json
import os
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from multidocqa.relevance_dataset import RelevanceDatasetBuilder


class TestRelevanceDatasetBuilder(unittest.TestCase):
    """Test cases for RelevanceDatasetBuilder."""

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
        ]

        # Create sample civil code
        self.sample_civil_code = [
            {"number": "101", "content": "Content of article 101"},
            {"number": "96", "content": "Content of article 96"},
            {"number": "102", "content": "Content of article 102"},
            {"number": "103", "content": "Content of article 103"},
            {"number": "104", "content": "Content of article 104"},
            {"number": "105", "content": "Content of article 105"},
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
        self.load_data_patcher = patch("multidocqa.relevance_dataset.load_data")

        self.mock_load_data = self.load_data_patcher.start()

        # Create combined dataset structure that RelevanceDatasetBuilder expects
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
        """Test RelevanceDatasetBuilder initialization."""
        builder = RelevanceDatasetBuilder(self.train_file.name, negative_ratio=3)

        self.assertEqual(builder.negative_ratio, 3)
        # Article lookup now includes paragraph variations (original + 3 variations each = 6 * 4 = 24)  # noqa: E501
        self.assertEqual(len(builder.all_articles), 24)
        self.assertEqual(len(builder.train_data), 3)

        # Check co-occurrence maps
        self.assertIn("case1", builder.statement_to_articles)
        self.assertEqual(builder.statement_to_articles["case1"], {"101", "96"})
        self.assertEqual(builder.statement_to_articles["case2"], {"102"})
        self.assertEqual(builder.statement_to_articles["case3"], {"101", "103"})

        self.assertIn("101", builder.article_to_statements)
        self.assertEqual(builder.article_to_statements["101"], {"case1", "case3"})

    def test_generate_positive_pairs(self) -> None:
        """Test positive pair generation."""
        builder = RelevanceDatasetBuilder(self.train_file.name, negative_ratio=3)

        positive_pairs = builder._generate_positive_pairs()

        # Should have 5 positive pairs total:
        # case1: (101, case1), (96, case1)
        # case2: (102, case2)
        # case3: (101, case3), (103, case3)
        self.assertEqual(len(positive_pairs), 5)

        # Check all pairs are labeled as positive
        for pair in positive_pairs:
            self.assertEqual(pair["label"], "Y")
            self.assertEqual(pair["pair_type"], "positive")
            self.assertIn("article_number", pair)
            self.assertIn("article_content", pair)
            self.assertIn("statement", pair)
            self.assertIn("statement_id", pair)

        # Check specific pairs exist
        article_statement_pairs = {
            (pair["article_number"], pair["statement_id"]) for pair in positive_pairs
        }

        expected_pairs = {
            ("101", "case1"),
            ("96", "case1"),
            ("102", "case2"),
            ("101", "case3"),
            ("103", "case3"),
        }

        self.assertEqual(article_statement_pairs, expected_pairs)

    @patch("multidocqa.relevance_dataset.random.sample")
    @patch("multidocqa.relevance_dataset.random.choices")
    def test_generate_negative_pairs(self, mock_choices: Any, mock_sample: Any) -> None:
        """Test negative pair generation with controlled randomness."""
        builder = RelevanceDatasetBuilder(self.train_file.name, negative_ratio=2)

        # Mock random.sample to return first N elements
        def sample_side_effect(population: Any, k: int) -> Any:
            return population[:k]

        # Mock random.choices to return first N elements with replacement
        def choices_side_effect(population: Any, k: int) -> Any:
            return population[:k]

        mock_sample.side_effect = sample_side_effect
        mock_choices.side_effect = choices_side_effect

        negative_pairs = builder._generate_negative_pairs(
            2
        )  # 2 negatives per statement

        # Should generate 2 negatives per statement (3 statements × 2 = 6 pairs)
        self.assertEqual(len(negative_pairs), 6)

        # Check all pairs are labeled as negative
        for pair in negative_pairs:
            self.assertEqual(pair["label"], "N")
            self.assertEqual(pair["pair_type"], "negative")
            self.assertIn("article_number", pair)
            self.assertIn("article_content", pair)
            self.assertIn("statement", pair)
            self.assertIn("statement_id", pair)

    def test_negative_pairs_are_actually_negative(self) -> None:
        """Test that negative pairs don't contain actual positive combinations."""
        builder = RelevanceDatasetBuilder(self.train_file.name, negative_ratio=2)

        # Get positive pairs for comparison
        positive_pairs = builder._generate_positive_pairs()
        positive_combinations = {
            (pair["article_number"], pair["statement_id"]) for pair in positive_pairs
        }

        # Generate negative pairs
        negative_pairs = builder._generate_negative_pairs(20)
        negative_combinations = {
            (pair["article_number"], pair["statement_id"]) for pair in negative_pairs
        }

        # Ensure no overlap
        overlap = positive_combinations.intersection(negative_combinations)
        self.assertEqual(len(overlap), 0, f"Found overlapping pairs: {overlap}")

    def test_create_relevance_prompt(self) -> None:
        """Test prompt creation for relevance prediction."""
        builder = RelevanceDatasetBuilder(self.train_file.name, negative_ratio=3)

        test_pair = {
            "article_number": "101",
            "article_content": "Content of article 101",
            "statement": "Can A rescind the contract?",
            "statement_id": "case1",
            "label": "Y",
            "pair_type": "positive",
        }

        prompt_data = builder._create_relevance_prompt(test_pair)

        # Check structure
        self.assertIn("prompt", prompt_data)
        self.assertIn("reference", prompt_data)
        self.assertIn("article_number", prompt_data)
        self.assertIn("statement", prompt_data)
        self.assertIn("statement_id", prompt_data)
        self.assertIn("pair_type", prompt_data)

        # Check content
        self.assertEqual(prompt_data["reference"], "Y")
        self.assertEqual(prompt_data["article_number"], "101")
        self.assertEqual(prompt_data["statement"], "Can A rescind the contract?")
        self.assertEqual(prompt_data["statement_id"], "case1")
        self.assertEqual(prompt_data["pair_type"], "positive")

        # Check prompt contains key elements
        prompt = prompt_data["prompt"]
        self.assertIn("Article 101", prompt)
        self.assertIn("Content of article 101", prompt)
        self.assertIn("Can A rescind the contract?", prompt)
        self.assertIn("relevant", prompt)
        self.assertIn("Y", prompt)
        self.assertIn("N", prompt)

    def test_build_relevance_dataset(self) -> None:
        """Test complete dataset building."""
        builder = RelevanceDatasetBuilder(self.train_file.name, negative_ratio=2)

        train_dataset, val_dataset = builder.build_relevance_dataset(split_ratio=0.8)

        # Should have positive pairs (5) + negative pairs (3 statements * 2 = 6) = 11 total  # noqa: E501
        total_samples = len(train_dataset) + len(val_dataset)
        self.assertEqual(total_samples, 11)

        # Check split ratio approximately correct
        train_ratio = len(train_dataset) / total_samples
        self.assertAlmostEqual(train_ratio, 0.8, delta=0.1)

        # Check dataset structure
        sample = train_dataset[0]
        required_keys = [
            "prompt",
            "reference",
            "article_number",
            "statement",
            "statement_id",
            "pair_type",
        ]
        for key in required_keys:
            self.assertIn(key, sample)

    def test_get_dataset_statistics(self) -> None:
        """Test dataset statistics computation."""
        builder = RelevanceDatasetBuilder(self.train_file.name, negative_ratio=3)

        stats = builder.get_dataset_statistics()

        # Check expected statistics
        self.assertEqual(stats["total_positive_pairs"], 5)
        self.assertEqual(
            stats["total_negative_pairs"], 9
        )  # 3 statements * 3 negatives per statement
        self.assertEqual(stats["negative_ratio"], 3)
        self.assertEqual(stats["total_civil_code_articles"], 6)

        # Check unique counts
        self.assertEqual(stats["unique_articles_in_positives"], 4)  # 101, 96, 102, 103
        self.assertEqual(stats["unique_statements"], 3)  # case1, case2, case3

        # Check most frequent article
        most_frequent_article, count = stats["most_frequent_article"]
        self.assertEqual(most_frequent_article, "101")  # appears in case1 and case3
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
