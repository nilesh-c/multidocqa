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

from multidocqa.relevance_dataset import (
    RelevanceDatasetBuilder,
    compute_relevance_reward,
    create_relevance_prompt_simple,
)


class TestRelevanceDatasetBuilder(unittest.TestCase):
    """Test cases for RelevanceDatasetBuilder."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create sample training data
        self.sample_train_data = [
            {
                "id": "case1",
                "question": "Can A rescind the contract?",
                "articles": [
                    {"number": "101", "content": "Content of article 101"},
                    {"number": "96", "content": "Content of article 96"},
                ],
                "label": "Y",
            },
            {
                "id": "case2",
                "question": "Is the contract valid?",
                "articles": [{"number": "102", "content": "Content of article 102"}],
                "label": "N",
            },
            {
                "id": "case3",
                "question": "Does B have authority?",
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
        self.load_civil_code_patcher = patch(
            "multidocqa.relevance_dataset.load_civil_code"
        )

        self.mock_load_data = self.load_data_patcher.start()
        self.mock_load_civil_code = self.load_civil_code_patcher.start()

        self.mock_load_data.return_value = self.sample_train_data
        self.mock_load_civil_code.return_value = self.sample_civil_code

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        os.unlink(self.train_file.name)
        os.unlink(self.civil_code_file.name)
        self.load_data_patcher.stop()
        self.load_civil_code_patcher.stop()

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


class TestRelevanceUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_create_relevance_prompt_simple(self) -> None:
        """Test simple relevance prompt creation."""
        article_content = "Content of article 101"
        statement = "Can A rescind the contract?"

        prompt = create_relevance_prompt_simple(article_content, statement)

        self.assertIn(article_content, prompt)
        self.assertIn(statement, prompt)
        self.assertIn("relevant", prompt)
        self.assertIn("Y", prompt)
        self.assertIn("N", prompt)

    def test_compute_relevance_reward(self) -> None:
        """Test relevance reward computation."""
        # Test correct predictions
        self.assertEqual(compute_relevance_reward("Y", "Y"), 1.0)
        self.assertEqual(compute_relevance_reward("N", "N"), 1.0)

        # Test incorrect predictions
        self.assertEqual(compute_relevance_reward("Y", "N"), 0.0)
        self.assertEqual(compute_relevance_reward("N", "Y"), 0.0)

        # Test case insensitive
        self.assertEqual(compute_relevance_reward("y", "Y"), 1.0)
        self.assertEqual(compute_relevance_reward("n", "N"), 1.0)

        # Test with whitespace
        self.assertEqual(compute_relevance_reward(" Y ", "Y"), 1.0)
        self.assertEqual(compute_relevance_reward("N\n", "N"), 1.0)


class TestRelevanceDatasetIntegration(unittest.TestCase):
    """Integration tests for the relevance dataset builder."""

    def setUp(self) -> None:
        """Set up integration test fixtures."""
        # Create a more comprehensive test dataset
        self.comprehensive_train_data = [
            {
                "id": "H27-3-U",
                "question": (
                    "In the case where B, who was granted authority of agency to buy a land as an "  # noqa: E501
                    "agent of A, concluded a contract for sale of a land 'X' with C "  # noqa: E501
                    "representing that the same is made on behalf of A by the fraud of C to "  # noqa: E501
                    "B, A may not rescind the contract for sale."
                ),
                "articles": [
                    {
                        "number": "101",
                        "content": (
                            "(1) If the validity of a manifestation of intention that an agent has "  # noqa: E501
                            "made to the other party is to be influenced by the absence of "  # noqa: E501
                            "intention; by mistake, fraud, or duress; or by the knowledge of or "  # noqa: E501
                            "negligence in not knowing of a particular circumstance; whether or "  # noqa: E501
                            "not any such fact was present is decided as it concerns the agent."  # noqa: E501
                        ),
                    },
                    {
                        "number": "96",
                        "content": (
                            "(1) A manifestation of intention based on fraud or duress is voidable. "  # noqa: E501
                            "(2) If a third party commits a fraud inducing a first party to make a "  # noqa: E501
                            "manifestation of intention to a second party, that manifestation of "  # noqa: E501
                            "intention is voidable only if the second party knew or could have "  # noqa: E501
                            "known that fact."
                        ),
                    },
                ],
                "label": "N",
            },
            {
                "id": "H27-3-E",
                "question": (
                    "In the case where B, who was granted authority of agency to buy a land as an "  # noqa: E501
                    "agent of A, was granted authority of agency to sell the land by C, became the "  # noqa: E501
                    "agent of C, and concluded a contract for sale of a land 'X' between A and C, "  # noqa: E501
                    "if B notified to both A and C in advance that B was the agent of both "  # noqa: E501
                    "parties, the contract for sale binds A and C."
                ),
                "articles": [
                    {
                        "number": "108",
                        "content": (
                            "(1) An act that a person performs as an agent of the counterparty or as "  # noqa: E501
                            "agent of both parties for the same juridical act is deemed to be an act "  # noqa: E501
                            "performed by a person without authority to represent; provided, "  # noqa: E501
                            "however, that this does not apply to the performance of an obligation "  # noqa: E501
                            "or to an act authorized by the principal in advance."
                        ),
                    }
                ],
                "label": "N",
            },
        ]

        # Create civil code with more articles
        self.comprehensive_civil_code = [
            {
                "number": "96",
                "content": (
                    "(1) A manifestation of intention based on fraud or duress is voidable..."  # noqa: E501
                ),
            },
            {
                "number": "101",
                "content": (
                    "(1) If the validity of a manifestation of intention that an agent has "  # noqa: E501
                    "made..."
                ),
            },
            {
                "number": "108",
                "content": (
                    "(1) An act that a person performs as an agent of the counterparty..."  # noqa: E501
                ),
            },
            {"number": "110", "content": "Content of article 110"},
            {"number": "120", "content": "Content of article 120"},
            {"number": "130", "content": "Content of article 130"},
        ]

        # Mock the load functions
        self.load_data_patcher = patch("multidocqa.relevance_dataset.load_data")
        self.load_civil_code_patcher = patch(
            "multidocqa.relevance_dataset.load_civil_code"
        )

        self.mock_load_data = self.load_data_patcher.start()
        self.mock_load_civil_code = self.load_civil_code_patcher.start()

        self.mock_load_data.return_value = self.comprehensive_train_data
        self.mock_load_civil_code.return_value = self.comprehensive_civil_code

    def tearDown(self) -> None:
        """Clean up integration test fixtures."""
        self.load_data_patcher.stop()
        self.load_civil_code_patcher.stop()

    def test_realistic_dataset_construction(self) -> None:
        """Test dataset construction with realistic data."""
        builder = RelevanceDatasetBuilder("dummy_train.json", negative_ratio=3)

        # Build datasets
        train_dataset, val_dataset = builder.build_relevance_dataset()

        # Check we have the expected number of positive pairs
        # H27-3-U: 2 articles (101, 96)
        # H27-3-E: 1 article (108)
        # Total: 3 positive pairs
        positive_pairs = builder._generate_positive_pairs()
        self.assertEqual(len(positive_pairs), 3)

        # Check we have the right ratio
        total_samples = len(train_dataset) + len(val_dataset)
        expected_total = 3 + (
            2 * 3
        )  # 3 positive + 6 negative (2 statements * 3 negatives per statement)
        self.assertEqual(total_samples, expected_total)

        # Verify no data leakage in negatives
        positive_pairs = builder._generate_positive_pairs()
        positive_combinations = {
            (pair["article_number"], pair["statement_id"]) for pair in positive_pairs
        }

        negative_pairs = builder._generate_negative_pairs(
            100
        )  # Generate more than needed
        negative_combinations = {
            (pair["article_number"], pair["statement_id"]) for pair in negative_pairs
        }

        overlap = positive_combinations.intersection(negative_combinations)
        self.assertEqual(len(overlap), 0)

    def test_prompt_quality(self) -> None:
        """Test that generated prompts contain appropriate content."""
        builder = RelevanceDatasetBuilder("dummy_train.json", negative_ratio=1)

        train_dataset, _ = builder.build_relevance_dataset()

        for sample in train_dataset:
            prompt = sample["prompt"]

            # Check prompt structure
            self.assertIn("Article", prompt)
            self.assertIn("Statement:", prompt)
            self.assertIn("relevant", prompt)
            self.assertIn("Y", prompt)
            self.assertIn("N", prompt)

            # Check reference is valid
            self.assertIn(sample["reference"], ["Y", "N"])

            # Check article number is referenced
            self.assertIn(sample["article_number"], prompt)


if __name__ == "__main__":
    unittest.main()
