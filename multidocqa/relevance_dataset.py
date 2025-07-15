"""
Relevance Dataset Builder for Legal Document Relevance Prediction

This module builds datasets for training models to predict whether a legal statement
is relevant to a given civil code article.

Dataset Construction:
- Positive pairs: (article, statement) pairs that appear together in entailment_pairs
- Negative pairs: (article, statement) pairs that don't appear together
- Ratio: 1:3 (positive:negative)
- Negative sampling: Random articles from civil code + other training articles

Input Format:
- Combined dataset with "articles" and "entailment_pairs" keys
- Articles: List of civil code articles with number and content
- Entailment pairs: List of statement-article pairs with labels
"""

import random
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from datasets import Dataset

from multidocqa.utils import load_data


class RelevanceDatasetBuilder:
    """Builder for legal relevance prediction datasets."""

    def __init__(self, combined_dataset_path: str, negative_ratio: int = 3):
        """
        Initialize the relevance dataset builder.

        Args:
            combined_dataset_path: Path to combined dataset with articles and
                entailment_pairs
            negative_ratio: Number of negative samples per positive sample
        """
        self.combined_dataset_path = combined_dataset_path
        self.negative_ratio = negative_ratio

        # Load combined data
        combined_data = load_data(combined_dataset_path)
        self.train_data: List[Dict] = combined_data["entailment_pairs"]  # type: ignore
        self.civil_code: List[Dict] = combined_data["articles"]  # type: ignore

        # Build article lookup that preserves full article numbers
        self.all_articles = self._build_article_lookup()

        # Track which articles appear with which statements
        self.statement_to_articles: Dict[str, Set[str]] = defaultdict(set)
        self.article_to_statements: Dict[str, Set[str]] = defaultdict(set)

        self._build_cooccurrence_maps()

    def _clean_article_number(self, article_number: str) -> str:
        """
        Clean article numbers by removing only paragraph suffixes.
        Preserves full article identifiers including sub-article parts.

        Examples:
        - '101(1)' -> '101'
        - '121-2(1)' -> '121-2'  (preserves the -2 part)
        - '398-3(2)' -> '398-3'  (preserves the -3 part)
        - '101' -> '101'         (unchanged)
        """
        return re.sub(r"\(\d+\)$", "", article_number)

    def _build_article_lookup(self) -> Dict[str, Dict]:
        """
        Build article lookup that handles both original and paragraph-suffixed numbers.
        Preserves full article numbers while supporting training data format.
        """
        lookup = {}

        # Add all civil code articles with their original numbers
        for article in self.civil_code:
            number = article["number"]
            lookup[number] = article

            # Also map common paragraph variations that appear in training data
            # This allows lookup of '121-2(1)' to find article '121-2'
            variations = [f"{number}(1)", f"{number}(2)", f"{number}(3)"]
            for variant in variations:
                if (
                    variant not in lookup
                ):  # Don't overwrite if variant exists as real article
                    lookup[variant] = article

        return lookup

    def _build_cooccurrence_maps(self) -> None:
        """Build maps of statement-article co-occurrences with clean article numbers."""
        for item in self.train_data:
            statement_id = item["id"]
            # Clean article numbers by removing paragraph suffixes only
            article_numbers = {
                self._clean_article_number(article["number"])
                for article in item["articles"]
            }

            self.statement_to_articles[statement_id] = article_numbers

            for article_num in article_numbers:
                self.article_to_statements[article_num].add(statement_id)

    def build_relevance_dataset(
        self, split_ratio: float = 0.9
    ) -> Tuple[Dataset, Dataset]:
        """
        Build train and validation datasets for relevance prediction.

        Args:
            split_ratio: Ratio for train/validation split

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Generate positive and negative pairs
        positive_pairs = self._generate_positive_pairs()
        negative_pairs = self._generate_negative_pairs(
            self.negative_ratio  # Now expects negatives per statement
        )

        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)

        # Create prompts
        dataset_items = [self._create_relevance_prompt(pair) for pair in all_pairs]

        # Split into train/val
        split_idx = int(len(dataset_items) * split_ratio)
        train_items = dataset_items[:split_idx]
        val_items = dataset_items[split_idx:]

        return Dataset.from_list(train_items), Dataset.from_list(val_items)

    def _generate_positive_pairs(self) -> List[Dict]:
        """Generate positive (article, statement) pairs from training data."""
        positive_pairs = []

        for item in self.train_data:
            statement = item["statement"]
            statement_id = item["id"]

            # Create positive pair for each article that appears with this statement
            for article in item["articles"]:
                # Clean the article number (remove paragraph suffixes only)
                clean_number = self._clean_article_number(article["number"])

                # Get content from civil code lookup (more reliable than training data)
                if clean_number in self.all_articles:
                    article_content = self.all_articles[clean_number]["content"]
                elif article["number"] in self.all_articles:
                    # Try original number if clean version not found
                    article_content = self.all_articles[article["number"]]["content"]
                    clean_number = article["number"]  # Use original if it exists
                else:
                    # Fallback to training data content if not found in civil code
                    article_content = article["content"]

                positive_pairs.append(
                    {
                        "article_number": clean_number,
                        "article_content": article_content,
                        "statement": statement,
                        "statement_id": statement_id,
                        "label": "Y",  # Relevant
                        "pair_type": "positive",
                    }
                )

        return positive_pairs

    def _generate_negative_pairs(self, num_negatives_per_statement: int) -> List[Dict]:
        """
        Generate negative (article, statement) pairs using structured
        per-statement sampling.
        """
        negative_pairs = []

        # Get all available civil code articles
        all_civil_code_articles = set(
            [article["number"] for article in self.civil_code]
        )

        # Pre-compute irrelevant articles for each statement
        statement_irrelevant_articles: Dict[str, List[str]] = {}
        for item in self.train_data:
            statement_id = item["id"]
            relevant_articles = self.statement_to_articles[statement_id]

            irrelevant_articles = all_civil_code_articles - relevant_articles
            statement_irrelevant_articles[statement_id] = list(irrelevant_articles)

        # Generate negatives for each statement
        for item in self.train_data:
            statement = item["statement"]
            statement_id = item["id"]

            # Get pre-computed irrelevant articles
            available_articles: List[str] = statement_irrelevant_articles[statement_id]

            if not available_articles:
                continue  # Skip if no irrelevant articles (shouldn't happen)

            # Sample the required number of irrelevant articles
            if len(available_articles) >= num_negatives_per_statement:
                # Sample without replacement for better diversity
                sampled_articles: List[str] = random.sample(
                    available_articles, num_negatives_per_statement
                )
            else:
                # If not enough unique articles, sample with replacement
                sampled_articles = random.choices(
                    available_articles, k=num_negatives_per_statement
                )

            # Create negative pairs
            for article_number in sampled_articles:
                article_content = self.all_articles[article_number]["content"]

                negative_pairs.append(
                    {
                        "article_number": article_number,
                        "article_content": article_content,
                        "statement": statement,
                        "statement_id": statement_id,
                        "label": "N",  # Not relevant
                        "pair_type": "negative",
                    }
                )

        return negative_pairs

    def _create_relevance_prompt(self, pair: Dict) -> Dict:
        """Create a prompt for relevance prediction."""
        article_content = pair["article_content"]
        statement = pair["statement"]

        prompt = f"""
You are a legal AI assistant. Given a civil code article and a legal statement, \
determine if the statement is relevant to the article.

A statement is relevant to an article if:
- The article's content could be used to reason about the statement
- The article provides applicable legal principles for the statement
- The article contains rules or definitions that apply to the statement's scenario

Respond strictly with 'Y' for relevant or 'N' for not relevant.

Article {pair['article_number']}: {article_content}

Statement: {statement}

Is this statement relevant to the article? Answer:
""".strip()

        return {
            "prompt": prompt,
            "reference": pair["label"],
            "article_number": pair["article_number"],
            "statement": statement,
            "statement_id": pair["statement_id"],
            "pair_type": pair["pair_type"],
        }

    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the dataset construction."""
        positive_pairs = self._generate_positive_pairs()
        negative_pairs = self._generate_negative_pairs(
            self.negative_ratio  # negatives per statement
        )

        # Count unique articles and statements
        unique_articles_in_positives = len(
            set(pair["article_number"] for pair in positive_pairs)
        )
        unique_articles_in_negatives = len(
            set(pair["article_number"] for pair in negative_pairs)
        )
        unique_statements = len(set(pair["statement_id"] for pair in positive_pairs))

        # Article usage distribution
        article_counts: Dict[str, int] = defaultdict(int)
        for pair in positive_pairs:
            article_counts[pair["article_number"]] += 1

        # Statement-article pair distribution
        statement_article_counts: Dict[int, int] = defaultdict(int)
        for articles in self.statement_to_articles.values():
            statement_article_counts[len(articles)] += 1

        return {
            "total_positive_pairs": len(positive_pairs),
            "total_negative_pairs": len(negative_pairs),
            "unique_articles_in_positives": unique_articles_in_positives,
            "unique_articles_in_negatives": unique_articles_in_negatives,
            "unique_statements": unique_statements,
            "total_civil_code_articles": len(self.civil_code),
            "negative_ratio": self.negative_ratio,
            "avg_articles_per_statement": sum(statement_article_counts.keys())
            / len(statement_article_counts),
            "most_frequent_article": max(article_counts.items(), key=lambda x: x[1]),
            "statement_article_distribution": dict(statement_article_counts),
        }


def create_relevance_prompt_simple(article_content: str, statement: str) -> str:
    """
    Create a simple relevance prediction prompt.

    Args:
        article_content: Content of the civil code article
        statement: Legal statement to check relevance for

    Returns:
        Formatted prompt for relevance prediction
    """
    return f"""
Given this civil code article and legal statement, determine if they are relevant.

Article: {article_content}

Statement: {statement}

Are they relevant? Respond Y or N:
""".strip()


def compute_relevance_reward(prediction: str, label: str) -> float:
    """
    Compute reward for relevance prediction.

    Args:
        prediction: Model prediction ('Y' or 'N')
        label: Ground truth label ('Y' or 'N')

    Returns:
        Reward score (1.0 for correct, 0.0 for incorrect)
    """
    prediction = prediction.strip().upper()
    label = label.strip().upper()
    return 1.0 if prediction == label else 0.0
