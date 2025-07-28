"""
Unit tests for the template loader module.

Tests template loading, rendering, and validation functionality.
"""

import tempfile
import unittest
from pathlib import Path

from multidocqa.common.template_loader import (
    render_entailment_prompt,
    render_relevance_prompt,
    render_template,
)


class TestTemplateLoader(unittest.TestCase):
    """Test template loading and rendering functionality."""

    def setUp(self) -> None:
        """Set up test fixtures with temporary template directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.template_dir = Path(self.temp_dir.name)

        # Create test templates
        self.test_template = self.template_dir / "test.txt"
        self.test_template.write_text("Hello {{ name }}!")

        self.entailment_template = self.template_dir / "entailment.txt"
        self.entailment_template.write_text(
            """Articles:
            {% for article in articles %}{{ article.number }}{% endfor %}
            Statement: {{ statement }}"""
        )

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_render_template_success(self) -> None:
        """Test successful template rendering."""
        result = render_template(
            "test.txt", template_dir=self.template_dir, name="World"
        )
        self.assertEqual(result, "Hello World!")

    def test_render_template_with_complex_data(self) -> None:
        """Test rendering with complex template data."""
        articles = [
            {"number": "1", "content": "Content 1"},
            {"number": "2", "content": "Content 2"},
        ]
        result = render_template(
            "entailment.txt",
            template_dir=self.template_dir,
            articles=articles,
            statement="Test statement",
        )
        self.assertIn("Articles:", result)
        self.assertIn("12", result)  # Article numbers concatenated
        self.assertIn("Test statement", result)

    def test_render_entailment_prompt(self) -> None:
        """Test entailment prompt rendering."""
        articles = [
            {"number": "1", "content": "Article 1 content"},
            {"number": "2", "content": "Article 2 content"},
        ]
        statement = "Test statement"

        result = render_entailment_prompt(articles, statement)

        self.assertIn("Article 1: Article 1 content", result)
        self.assertIn("Article 2: Article 2 content", result)
        self.assertIn("Test statement", result)
        self.assertIn("Y' or 'N'", result)

    def test_render_relevance_prompt(self) -> None:
        """Test relevance prompt rendering."""
        result = render_relevance_prompt(
            article_content="Test article content",
            statement="Test statement",
            article_number="123",
        )

        self.assertIn("Test article content", result)
        self.assertIn("Test statement", result)
        self.assertIn("Article 123:", result)
        self.assertIn("relevant", result)


if __name__ == "__main__":
    unittest.main()
