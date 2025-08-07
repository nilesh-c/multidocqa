"""
Unit tests for LLM providers module.

Tests the abstract provider interface and provider-specific implementations.
"""

import unittest
from unittest.mock import Mock, patch

from pydantic import BaseModel

from multidocqa.common.llm_providers import (
    LLMProvider,
    OpenAIProvider,
    ParseMode,
)
from multidocqa.exceptions import LLMError


class LLMTestResponse(BaseModel):
    """Test response model for structured parsing."""

    answer: str
    confidence: float


class MockProvider(LLMProvider):
    """Mock provider for testing abstract base class."""

    @property
    def provider_name(self) -> str:
        return "mock"

    def get_default_model(self) -> str:
        return "mock-model"

    def requires_api_key(self) -> bool:
        return False

    def _get_api_key_from_env(self):
        return "mock-key"

    def _initialize_client(self) -> None:
        self.client = Mock()

    def _get_structured_response(self, prompt: str, response_format):
        return LLMTestResponse(answer="Y", confidence=0.9), None

    async def _get_structured_response_async(self, prompt: str, response_format):
        return LLMTestResponse(answer="Y", confidence=0.9), None

    def _get_text_response(self, prompt: str):
        return '{"answer": "Y", "confidence": 0.9}', None

    async def _get_text_response_async(self, prompt: str):
        return '{"answer": "Y", "confidence": 0.9}', None


class TestLLMProvider(unittest.TestCase):
    """Test the abstract LLM provider base class."""

    def setUp(self):
        self.provider = MockProvider(
            api_key="test-key",
            model="test-model",
            timeout=30,
            parse_mode=ParseMode.STRUCTURED,
        )

    def test_initialization(self):
        """Test provider initialization."""
        self.assertEqual(self.provider.api_key, "test-key")
        self.assertEqual(self.provider.model, "test-model")
        self.assertEqual(self.provider.timeout, 30)
        self.assertEqual(self.provider.parse_mode, ParseMode.STRUCTURED)

    def test_get_structured_response_structured_mode(self):
        """Test structured response in structured mode."""
        response, reasoning_trace = self.provider.get_structured_response(
            "Test prompt", LLMTestResponse
        )

        self.assertIsInstance(response, LLMTestResponse)
        self.assertEqual(response.answer, "Y")
        self.assertEqual(response.confidence, 0.9)
        self.assertIsNone(reasoning_trace)  # No reasoning enabled in test

    def test_get_structured_response_manual_mode(self):
        """Test structured response in manual mode."""
        self.provider.parse_mode = ParseMode.MANUAL
        response, reasoning_trace = self.provider.get_structured_response(
            "Test prompt", LLMTestResponse
        )

        self.assertIsInstance(response, LLMTestResponse)
        self.assertEqual(response.answer, "Y")
        self.assertEqual(response.confidence, 0.9)
        self.assertIsNone(reasoning_trace)  # No reasoning enabled in test

    def test_get_structured_response_async_structured_mode(self):
        """Test async structured response in structured mode."""
        import asyncio

        response, reasoning_trace = asyncio.run(
            self.provider.get_structured_response_async("Test prompt", LLMTestResponse)
        )

        self.assertIsInstance(response, LLMTestResponse)
        self.assertEqual(response.answer, "Y")
        self.assertEqual(response.confidence, 0.9)
        self.assertIsNone(reasoning_trace)  # No reasoning enabled in test

    def test_get_structured_response_async_manual_mode(self):
        """Test async structured response in manual mode."""
        import asyncio

        self.provider.parse_mode = ParseMode.MANUAL
        response, reasoning_trace = asyncio.run(
            self.provider.get_structured_response_async("Test prompt", LLMTestResponse)
        )

        self.assertIsInstance(response, LLMTestResponse)
        self.assertEqual(response.answer, "Y")
        self.assertEqual(response.confidence, 0.9)
        self.assertIsNone(reasoning_trace)  # No reasoning enabled in test

    def test_extract_json_from_response_valid_json(self):
        """Test JSON extraction from valid response."""
        response_text = '{"answer": "Y", "confidence": 0.9}'
        result = self.provider._extract_json_from_response(response_text)

        self.assertEqual(result, {"answer": "Y", "confidence": 0.9})

    def test_extract_json_from_response_json_in_markdown(self):
        """Test JSON extraction from markdown code block."""
        response_text = """Here's the response:
```json
{"answer": "Y", "confidence": 0.9}
```
That's the result."""
        result = self.provider._extract_json_from_response(response_text)

        self.assertEqual(result, {"answer": "Y", "confidence": 0.9})

    def test_extract_json_from_response_invalid_json(self):
        """Test JSON extraction with invalid JSON."""
        response_text = "This is not valid JSON"

        with self.assertRaises(LLMError):
            self.provider._extract_json_from_response(response_text)

    def test_validate_response_structure_valid(self):
        """Test response structure validation with valid data."""
        data = {"answer": "Y", "confidence": 0.9}
        result = self.provider._validate_response_structure(data, LLMTestResponse)

        self.assertIsInstance(result, LLMTestResponse)
        self.assertEqual(result.answer, "Y")

    def test_validate_response_structure_invalid(self):
        """Test response structure validation with invalid data."""
        data = {"wrong_field": "value"}

        with self.assertRaises(LLMError):
            self.provider._validate_response_structure(data, LLMTestResponse)


class TestOpenAIProvider(unittest.TestCase):
    """Test OpenAI provider implementation."""

    def setUp(self):
        self.provider = OpenAIProvider(api_key="test-key", model="gpt-4", timeout=30)

    @patch("openai.OpenAI")
    @patch("openai.AsyncOpenAI")
    def test_initialization(self, mock_async_openai, mock_openai):
        """Test OpenAI provider initialization."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-4")

        self.assertEqual(provider.api_key, "test-key")
        self.assertEqual(provider.model, "gpt-4")
        self.assertEqual(provider.provider_name, "openai")
        mock_openai.assert_called_once()
        mock_async_openai.assert_called_once()

    def test_get_structured_response(self):
        """Test OpenAI structured response implementation."""
        # Mock OpenAI client response
        mock_choice = Mock()
        mock_choice.message = Mock()
        mock_choice.message.parsed = LLMTestResponse(answer="Y", confidence=0.9)
        mock_choice.message.reasoning = None  # No reasoning

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        # Create provider and mock its client
        provider = OpenAIProvider(api_key="test-key", model="gpt-4")
        provider.client = Mock()
        provider.client.chat.completions.parse.return_value = mock_response

        result, reasoning_trace = provider._get_structured_response(
            "Test prompt", LLMTestResponse
        )

        self.assertIsInstance(result, LLMTestResponse)
        self.assertEqual(result.answer, "Y")
        self.assertIsNone(reasoning_trace)  # No reasoning in this mock

    def test_get_text_response(self):
        """Test OpenAI text response."""
        # Mock OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.reasoning = None

        # Create provider and mock its client
        provider = OpenAIProvider(api_key="test-key", model="gpt-4")
        provider.client = Mock()
        provider.client.chat.completions.create.return_value = mock_response

        result, reasoning_trace = provider._get_text_response("Test prompt")

        self.assertEqual(result, "Test response")
        self.assertIsNone(reasoning_trace)


if __name__ == "__main__":
    unittest.main()
