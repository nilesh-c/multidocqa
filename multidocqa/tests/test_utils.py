"""
Simple tests for utility functions in multidocqa.utils
"""

import unittest

from multidocqa.utils import extract_articles


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_extract_articles_simple(self) -> None:
        """Test basic article extraction."""
        text = "Article 1 This is the first article.\nArticle 2 This is the second article."  # noqa: E501
        result = extract_articles(text)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["number"], "1")
        self.assertEqual(result[0]["content"], "This is the first article.")
        self.assertEqual(result[1]["number"], "2")
        self.assertEqual(result[1]["content"], "This is the second article.")

    def test_extract_articles_multiline(self) -> None:
        """Test article extraction with multiline content."""
        text = """Article 1 First line.
Second line of article 1.
Article 2 Article 2 content."""
        result = extract_articles(text)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["content"], "First line. Second line of article 1.")
        self.assertEqual(result[1]["content"], "Article 2 content.")


if __name__ == "__main__":
    unittest.main()
