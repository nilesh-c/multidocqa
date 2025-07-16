"""
Entailment Dataset Builder for COLIEE 2025 (Task 4)

This module builds datasets for training models to predict whether a set of
civil code articles entail a legal statement (Y) or not (N).

Dataset Construction:
- Uses existing entailment pairs from training data
- Creates prompts with all relevant articles for each statement
- Balanced dataset with equal Y/N samples or configurable ratio
- Supports different prompt templates and article selection strategies

Input Format:
- Combined dataset with "articles" and "entailment_pairs" keys generated
  from COLIEE XML files (see scripts/build_dataset.py)
- Articles: List of civil code articles with number and content
- Entailment pairs: List of statement-article pairs with Y/N labels
"""

import random
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import Dataset

from multidocqa.utils import create_simple_prompt, load_data


class EntailmentDatasetBuilder:
    """Builder for legal entailment prediction datasets."""

    def __init__(self, combined_dataset_path: str, balance_classes: bool = True):
        """
        Initialize the entailment dataset builder.

        Args:
            combined_dataset_path: Path to combined dataset with articles and
                entailment_pairs
            balance_classes: Whether to balance Y/N classes in the dataset
        """
        self.combined_dataset_path = combined_dataset_path
        self.balance_classes = balance_classes

        # Load combined data
        combined_data = load_data(combined_dataset_path)
        self.train_data: List[Dict] = combined_data["entailment_pairs"]  # type: ignore  # noqa: E501
        self.civil_code: List[Dict] = combined_data["articles"]  # type: ignore

        # Build article lookup for efficient retrieval
        self.article_lookup = {
            article["number"]: article for article in self.civil_code
        }

        # Analyze class distribution
        self.class_distribution = self._analyze_class_distribution()

    def _analyze_class_distribution(self) -> Dict[str, int]:
        """Analyze the distribution of Y/N labels in the dataset."""
        distribution: Dict[str, int] = defaultdict(int)
        for item in self.train_data:
            distribution[item["label"]] += 1
        return dict(distribution)

    def build_entailment_dataset(
        self, split_ratio: float = 0.9
    ) -> Tuple[Dataset, Dataset]:
        """
        Build train and validation datasets for entailment prediction.

        Args:
            split_ratio: Ratio for train/validation split

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Generate entailment samples
        samples = self._generate_entailment_samples()

        # Balance classes if requested
        if self.balance_classes:
            samples = self._balance_classes(samples)

        # Shuffle samples
        random.shuffle(samples)

        # Create prompts
        dataset_items = [self._create_entailment_prompt(sample) for sample in samples]

        # Split into train/val
        split_idx = int(len(dataset_items) * split_ratio)
        train_items = dataset_items[:split_idx]
        val_items = dataset_items[split_idx:]

        return Dataset.from_list(train_items), Dataset.from_list(val_items)

    def _generate_entailment_samples(self) -> List[Dict]:
        """Generate entailment samples from training data."""
        samples = []

        for item in self.train_data:
            # Get all articles for this statement
            articles = item["articles"]

            # Create sample with all relevant articles
            sample = {
                "statement": item["statement"],
                "statement_id": item["id"],
                "articles": articles,
                "label": item["label"],
                "sample_type": "original",
            }
            samples.append(sample)

        return samples

    def _balance_classes(self, samples: List[Dict]) -> List[Dict]:
        """Balance Y/N classes in the dataset."""
        # Separate by class
        y_samples = [s for s in samples if s["label"] == "Y"]
        n_samples = [s for s in samples if s["label"] == "N"]

        # Find minimum count
        min_count = min(len(y_samples), len(n_samples))

        # Sample equal numbers from each class
        balanced_samples = []
        if min_count > 0:
            balanced_samples.extend(random.sample(y_samples, min_count))
            balanced_samples.extend(random.sample(n_samples, min_count))

        return balanced_samples

    def _create_entailment_prompt(self, sample: Dict) -> Dict:
        """Create a prompt for entailment prediction."""
        articles = sample["articles"]
        statement = sample["statement"]

        # Use the existing simple prompt creation function
        prompt = create_simple_prompt(articles, statement)

        return {
            "prompt": prompt,
            "reference": sample["label"],
            "statement": statement,
            "statement_id": sample["statement_id"],
            "articles": articles,
            "sample_type": sample["sample_type"],
            "num_articles": len(articles),
        }

    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the dataset construction."""
        samples = self._generate_entailment_samples()

        # Class distribution
        y_count = sum(1 for s in samples if s["label"] == "Y")
        n_count = sum(1 for s in samples if s["label"] == "N")

        # Article count distribution
        article_counts = [len(s["articles"]) for s in samples]
        avg_articles = (
            sum(article_counts) / len(article_counts) if article_counts else 0.0
        )
        min_articles = min(article_counts) if article_counts else 0
        max_articles = max(article_counts) if article_counts else 0

        # Article usage frequency
        article_usage: Dict[str, int] = defaultdict(int)
        for sample in samples:
            for article in sample["articles"]:
                article_usage[article["number"]] += 1

        # Most/least used articles
        most_used = (
            max(article_usage.items(), key=lambda x: x[1]) if article_usage else ("", 0)
        )
        least_used = (
            min(article_usage.items(), key=lambda x: x[1]) if article_usage else ("", 0)
        )

        return {
            "total_samples": len(samples),
            "y_samples": y_count,
            "n_samples": n_count,
            "class_ratio": f"{y_count}:{n_count}",
            "avg_articles_per_statement": avg_articles,
            "min_articles_per_statement": min_articles,
            "max_articles_per_statement": max_articles,
            "unique_articles_used": len(article_usage),
            "total_civil_code_articles": len(self.civil_code),
            "most_used_article": most_used,
            "least_used_article": least_used,
            "balance_classes": self.balance_classes,
        }


def create_entailment_prompt_simple(articles: List[Dict], statement: str) -> str:
    """
    Create a simple entailment prediction prompt.

    Args:
        articles: List of civil code articles
        statement: Legal statement to check entailment for

    Returns:
        Formatted prompt for entailment prediction
    """
    return create_simple_prompt(articles, statement)


def compute_entailment_reward(prediction: str, label: str) -> float:
    """
    Compute reward for entailment prediction.

    Args:
        prediction: Model prediction ('Y' or 'N')
        label: Ground truth label ('Y' or 'N')

    Returns:
        Reward score (1.0 for correct, 0.0 for incorrect)
    """
    from multidocqa.utils import compute_reward

    return compute_reward(prediction, label)
