"""
Abstract interface and implementations for different LLM providers.

This module provides a clean architecture for structured extraction
that supports multiple LLM providers (OpenAI, Anthropic, Mistral, vLLM)
with reasoning/thinking support and flexible token limits.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from pydantic import BaseModel

from ..config.defaults import (
    DEFAULT_BASE_URL,
    LLM_GENERATION_TEMPERATURE,
    TIMEOUT_MULTIPLIER_MS,
)
from ..exceptions import ConfigurationError, LLMError

# Generic type for response models
T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class ParseMode(Enum):
    """Enumeration for parsing modes."""

    STRUCTURED = "structured"  # Use provider's structured output API
    MANUAL = "manual"  # Parse JSON manually from text response


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        parse_mode: ParseMode = ParseMode.STRUCTURED,
        enable_reasoning: bool = False,
        reasoning_params: Optional[Dict[str, Any]] = None,
        max_tokens: int = 64000,
    ) -> None:
        self.api_key = api_key or self._get_api_key_from_env()
        self.model = model or self.get_default_model()
        self.timeout = timeout
        self.parse_mode = parse_mode
        self.enable_reasoning = enable_reasoning
        self.reasoning_params = reasoning_params or {}
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        if self.requires_api_key() and not self.api_key:
            raise ConfigurationError(f"{self.provider_name} API key is required")

        self._initialize_client()

    def get_structured_response(
        self, prompt: str, response_format: Type[T]
    ) -> Tuple[T, Optional[str]]:
        """Get structured response from the LLM provider."""
        if self.parse_mode == ParseMode.STRUCTURED:
            return self._get_structured_response(prompt, response_format)
        else:
            text_response, reasoning_trace = self._get_text_response(prompt)
            response_data = self._extract_json_from_response(text_response)
            parsed_response = self._validate_response_structure(
                response_data, response_format
            )
            return parsed_response, reasoning_trace

    async def get_structured_response_async(
        self, prompt: str, response_format: Type[T]
    ) -> Tuple[T, Optional[str]]:
        """Get structured response from the LLM provider (async)."""
        if self.parse_mode == ParseMode.STRUCTURED:
            return await self._get_structured_response_async(prompt, response_format)
        else:
            text_response, reasoning_trace = await self._get_text_response_async(prompt)
            response_data = self._extract_json_from_response(text_response)
            parsed_response = self._validate_response_structure(
                response_data, response_format
            )
            return parsed_response, reasoning_trace

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""

    @abstractmethod
    def get_default_model(self) -> str:
        """Return the default model name for this provider."""

    @abstractmethod
    def requires_api_key(self) -> bool:
        """Return whether this provider requires an API key."""

    @abstractmethod
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable."""

    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize provider-specific client(s)."""

    @abstractmethod
    def _get_structured_response(
        self, prompt: str, response_format: Type[T]
    ) -> Tuple[T, Optional[str]]:
        """Get structured response implementation."""

    @abstractmethod
    async def _get_structured_response_async(
        self, prompt: str, response_format: Type[T]
    ) -> Tuple[T, Optional[str]]:
        """Get structured response implementation (async)."""

    @abstractmethod
    def _get_text_response(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Get text response with optional reasoning trace."""

    @abstractmethod
    async def _get_text_response_async(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Get text response with optional reasoning trace (async)."""

    def _validate_response_structure(
        self, response_data: Dict[str, Any], response_format: Type[T]
    ) -> T:
        """Validate and parse response data into specified format."""
        try:
            return response_format(**response_data)  # type: ignore
        except Exception as e:
            raise LLMError(f"Invalid response structure: {e}") from e

    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from provider response with robust parsing."""
        content = content.strip()

        # Handle markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        try:
            result: Dict[str, Any] = json.loads(content)
            return result
        except json.JSONDecodeError as e:
            # Try to find JSON object in the text
            start = content.find("{")
            if start != -1:
                brace_count = 0
                for i, char in enumerate(content[start:], start):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                result = json.loads(content[start : i + 1])
                                return result
                            except json.JSONDecodeError:
                                pass

            raise LLMError(f"Failed to extract valid JSON from response: {e}")

    def _create_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Create standard message format."""
        return [{"role": "user", "content": prompt}]


class MistralProvider(LLMProvider):
    """Mistral AI provider implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        parse_mode: ParseMode = ParseMode.STRUCTURED,
        enable_reasoning: bool = False,
        reasoning_params: Optional[Dict[str, Any]] = None,
        max_tokens: int = 64000,
    ) -> None:
        super().__init__(
            api_key,
            model,
            timeout,
            parse_mode,
            enable_reasoning,
            reasoning_params,
            max_tokens,
        )

    @property
    def provider_name(self) -> str:
        return "mistral"

    def get_default_model(self) -> str:
        return "mistral-large-latest"

    def requires_api_key(self) -> bool:
        return True

    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv("MISTRAL_API_KEY")

    def _initialize_client(self) -> None:
        try:
            from mistralai import Mistral

            self.client = Mistral(api_key=self.api_key)
        except ImportError:
            raise ConfigurationError(
                "mistralai package is required for Mistral provider"
            )

    def _make_api_call(
        self,
        prompt: str,
        response_format: Optional[Type[T]] = None,
        is_async: bool = False,
    ) -> Any:
        """Unified API call method for Mistral."""
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self._create_messages(prompt),
        }

        if self.timeout:
            call_kwargs["timeout_ms"] = self.timeout * TIMEOUT_MULTIPLIER_MS

        # Add response format if provided
        if response_format:
            call_kwargs["response_format"] = response_format
            method_name = "parse_async" if is_async else "parse"
        else:
            call_kwargs["max_tokens"] = self.max_tokens
            call_kwargs["temperature"] = LLM_GENERATION_TEMPERATURE
            method_name = "complete_async" if is_async else "complete"

        api_method = getattr(self.client.chat, method_name)

        if is_async:
            return api_method(**call_kwargs)
        else:
            response = api_method(**call_kwargs)
            return self._extract_response_content(response, response_format is not None)

    async def _make_api_call_async(
        self, prompt: str, response_format: Optional[Type[T]] = None
    ) -> Union[T, Tuple[str, Optional[str]]]:
        """Async unified API call method for Mistral."""
        response = await cast(
            Awaitable[Any], self._make_api_call(prompt, response_format, is_async=True)
        )
        return self._extract_response_content(response, response_format is not None)

    def _extract_response_content(
        self, response: Any, is_structured: bool
    ) -> Union[Any, Tuple[str, Optional[str]]]:
        """Extract content from Mistral response."""
        if is_structured:
            parsed_response = response.choices[0].message.parsed
            if parsed_response is None:
                raise LLMError(
                    f"Failed to parse response from {self.provider_name} API"
                )
            return parsed_response, None
        else:
            if not response.choices or not response.choices[0].message.content:
                raise LLMError("Empty response from Mistral API")
            return response.choices[0].message.content, None

    def _get_structured_response(
        self, prompt: str, response_format: Type[T]
    ) -> Tuple[T, Optional[str]]:
        """Get structured response from Mistral API."""
        result = self._make_api_call(prompt, response_format)
        return cast(Tuple[T, Optional[str]], result)

    async def _get_structured_response_async(
        self, prompt: str, response_format: Type[T]
    ) -> Tuple[T, Optional[str]]:
        """Get structured response from Mistral API (async)."""
        result = await self._make_api_call_async(prompt, response_format)
        return cast(Tuple[T, Optional[str]], result)

    def _get_text_response(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Get text response from Mistral API."""
        result = self._make_api_call(prompt)
        return cast(Tuple[str, Optional[str]], result)

    async def _get_text_response_async(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Get text response from Mistral API (async)."""
        result = await self._make_api_call_async(prompt)
        return result


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation with reasoning support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        parse_mode: ParseMode = ParseMode.STRUCTURED,
        enable_reasoning: bool = False,
        reasoning_params: Optional[Dict[str, Any]] = None,
        max_tokens: int = 64000,
    ) -> None:
        super().__init__(
            api_key,
            model,
            timeout,
            parse_mode,
            enable_reasoning,
            reasoning_params,
            max_tokens,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    def get_default_model(self) -> str:
        return "gpt-4o"

    def requires_api_key(self) -> bool:
        return True

    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")

    def _initialize_client(self) -> None:
        try:
            from openai import AsyncOpenAI, OpenAI

            self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
            self.async_client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)
        except ImportError:
            raise ConfigurationError("openai package is required for OpenAI provider")

    def _make_api_call(
        self,
        prompt: str,
        response_format: Optional[Type[T]] = None,
        is_async: bool = False,
    ) -> Any:
        """Unified API call method for OpenAI."""
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self._create_messages(prompt),
            "max_completion_tokens": self.max_tokens,
        }

        # Add reasoning effort for o3 models
        if (
            self.enable_reasoning
            and "o3" in self.model.lower()
            and self.reasoning_params.get("effort")
        ):
            call_kwargs["reasoning_effort"] = self.reasoning_params["effort"]

        # Add response format if provided
        if response_format:
            call_kwargs["response_format"] = response_format
            method = "parse"
        else:
            call_kwargs["temperature"] = LLM_GENERATION_TEMPERATURE
            method = "create"

        client = self.async_client if is_async else self.client
        api_method = getattr(client.chat.completions, method)

        if is_async:
            return api_method(**call_kwargs)
        else:
            response = api_method(**call_kwargs)
            return self._extract_response_content(response, response_format is not None)

    async def _make_api_call_async(
        self, prompt: str, response_format: Optional[Type[T]] = None
    ) -> Union[T, Tuple[str, Optional[str]]]:
        """Async unified API call method for OpenAI."""
        response = await cast(
            Awaitable[Any], self._make_api_call(prompt, response_format, is_async=True)
        )
        return self._extract_response_content(response, response_format is not None)

    def _extract_response_content(
        self, response: Any, is_structured: bool
    ) -> Union[Any, Tuple[str, Optional[str]]]:
        """Extract content and reasoning trace from OpenAI response."""
        if not response.choices or not response.choices[0].message:
            raise LLMError("Empty response from OpenAI API")

        message = response.choices[0].message

        # Extract reasoning trace if available
        reasoning_trace = None
        if hasattr(message, "reasoning") and message.reasoning:
            reasoning_trace = message.reasoning

        if is_structured:
            if message.parsed is None:
                raise LLMError("Failed to parse response from OpenAI API")
            return message.parsed, reasoning_trace
        else:
            if not message.content:
                raise LLMError("Empty content in OpenAI response")
            return message.content, reasoning_trace

    def _get_structured_response(
        self, prompt: str, response_format: Type[T]
    ) -> Tuple[T, Optional[str]]:
        """Get structured response from OpenAI API."""
        result = self._make_api_call(prompt, response_format)
        return cast(Tuple[T, Optional[str]], result)

    async def _get_structured_response_async(
        self, prompt: str, response_format: Type[T]
    ) -> Tuple[T, Optional[str]]:
        """Get structured response from OpenAI API (async)."""
        result = await self._make_api_call_async(prompt, response_format)
        return cast(Tuple[T, Optional[str]], result)

    def _get_text_response(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Get text response from OpenAI API."""
        result = self._make_api_call(prompt)
        return cast(Tuple[str, Optional[str]], result)

    async def _get_text_response_async(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Get text response from OpenAI API (async)."""
        result = await self._make_api_call_async(prompt)
        return result


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) provider implementation with thinking support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        parse_mode: ParseMode = ParseMode.MANUAL,  # Anthropic doesn't have structured API
        enable_reasoning: bool = False,
        reasoning_params: Optional[Dict[str, Any]] = None,
        max_tokens: int = 64000,
    ) -> None:
        super().__init__(
            api_key,
            model,
            timeout,
            parse_mode,
            enable_reasoning,
            reasoning_params,
            max_tokens,
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def get_default_model(self) -> str:
        return "claude-sonnet-4-0"

    def requires_api_key(self) -> bool:
        return True

    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv("ANTHROPIC_API_KEY")

    def _initialize_client(self) -> None:
        try:
            from anthropic import Anthropic, AsyncAnthropic

            self.client = Anthropic(api_key=self.api_key, timeout=self.timeout)
            self.async_client = AsyncAnthropic(
                api_key=self.api_key, timeout=self.timeout
            )
        except ImportError:
            raise ConfigurationError(
                "anthropic package is required for Anthropic provider"
            )

    def _make_api_call(self, prompt: str, is_async: bool = False) -> Any:
        """Unified API call method for Anthropic."""
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._create_messages(prompt),
        }

        # Add thinking configuration if enabled
        if self.enable_reasoning and self.reasoning_params.get("thinking_budget"):
            call_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.reasoning_params["thinking_budget"],
            }

        client = self.async_client if is_async else self.client

        if is_async:
            return client.messages.create(**call_kwargs)
        else:
            response = client.messages.create(**call_kwargs)
            return self._extract_response_content(response)

    async def _make_api_call_async(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Async unified API call method for Anthropic."""
        response = await cast(
            Awaitable[Any], self._make_api_call(prompt, is_async=True)
        )
        return self._extract_response_content(response)

    def _extract_response_content(self, response: Any) -> Tuple[str, Optional[str]]:
        """Extract content and thinking trace from Anthropic response."""
        if not response.content:
            raise LLMError("Empty response from Anthropic API")

        # Extract text and thinking blocks separately
        text_blocks = [block for block in response.content if block.type == "text"]
        thinking_blocks = [
            block for block in response.content if block.type == "thinking"
        ]

        if not text_blocks:
            raise LLMError("No text content found in Anthropic response")

        content = text_blocks[0].text
        thinking_trace = None

        if thinking_blocks:
            thinking_trace = "\n\n".join([block.thinking for block in thinking_blocks])

        return content, thinking_trace

    def _get_structured_response(
        self, prompt: str, response_format: Type[T]
    ) -> Tuple[T, Optional[str]]:
        """Anthropic doesn't have structured API - redirect to text generation + parsing."""
        text_response, reasoning_trace = self._get_text_response(prompt)
        response_data = self._extract_json_from_response(text_response)
        parsed_response = self._validate_response_structure(
            response_data, response_format
        )
        return parsed_response, reasoning_trace

    async def _get_structured_response_async(
        self, prompt: str, response_format: Type[T]
    ) -> Tuple[T, Optional[str]]:
        """Anthropic doesn't have structured API - redirect to text generation + parsing."""
        text_response, reasoning_trace = await self._get_text_response_async(prompt)
        response_data = self._extract_json_from_response(text_response)
        parsed_response = self._validate_response_structure(
            response_data, response_format
        )
        return parsed_response, reasoning_trace

    def _get_text_response(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Get text response from Anthropic API."""
        return cast(Tuple[str, Optional[str]], self._make_api_call(prompt))

    async def _get_text_response_async(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Get text response from Anthropic API (async)."""
        return await self._make_api_call_async(prompt)


class VLLMProvider(LLMProvider):
    """vLLM local server provider implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        parse_mode: ParseMode = ParseMode.STRUCTURED,
        enable_reasoning: bool = False,
        reasoning_params: Optional[Dict[str, Any]] = None,
        max_tokens: int = 64000,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        super().__init__(
            api_key,
            model,
            timeout,
            parse_mode,
            enable_reasoning,
            reasoning_params,
            max_tokens,
        )

    @property
    def provider_name(self) -> str:
        return "vllm"

    def get_default_model(self) -> str:
        return "mistralai/Mistral-7B-Instruct-v0.2"

    def requires_api_key(self) -> bool:
        return False

    def _get_api_key_from_env(self) -> Optional[str]:
        return "dummy"  # vLLM often accepts any value

    def _initialize_client(self) -> None:
        try:
            from openai import AsyncOpenAI, OpenAI

            self.client = OpenAI(
                api_key=self.api_key or "dummy",
                base_url=self.base_url,
                timeout=self.timeout,
            )
            self.async_client = AsyncOpenAI(
                api_key=self.api_key or "dummy",
                base_url=self.base_url,
                timeout=self.timeout,
            )
        except ImportError:
            raise ConfigurationError("openai package is required for vLLM provider")

    def _make_api_call(
        self,
        prompt: str,
        response_format: Optional[Type[T]] = None,
        is_async: bool = False,
    ) -> Any:
        """Unified API call method for vLLM."""
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self._create_messages(prompt),
            "max_tokens": self.max_tokens,
            "temperature": LLM_GENERATION_TEMPERATURE,
        }

        # Add guided JSON for structured output
        if response_format:
            call_kwargs["response_format"] = {"type": "json_object"}
            call_kwargs["extra_body"] = {
                "guided_json": response_format.model_json_schema()
            }

        client = self.async_client if is_async else self.client

        if is_async:
            return client.chat.completions.create(**call_kwargs)
        else:
            response = client.chat.completions.create(**call_kwargs)
            return self._extract_response_content(response, response_format is not None)

    async def _make_api_call_async(
        self, prompt: str, response_format: Optional[Type[T]] = None
    ) -> Union[Dict[str, Any], Tuple[str, Optional[str]]]:
        """Async unified API call method for vLLM."""
        response = await cast(
            Awaitable[Any], self._make_api_call(prompt, response_format, is_async=True)
        )
        return self._extract_response_content(response, response_format is not None)

    def _extract_response_content(
        self, response: Any, is_structured: bool
    ) -> Union[Any, Tuple[str, Optional[str]]]:
        """Extract content from vLLM response."""
        if not response.choices or not response.choices[0].message.content:
            raise LLMError("Empty response from vLLM API")

        content = response.choices[0].message.content

        if is_structured:
            try:
                response_data: Dict[str, Any] = json.loads(content)
                return response_data, None
            except json.JSONDecodeError as e:
                raise LLMError(f"Invalid JSON response from vLLM: {e}") from e
        else:
            return content, None

    def _get_structured_response(
        self, prompt: str, response_format: Type[T]
    ) -> Tuple[T, Optional[str]]:
        """Get structured response from vLLM API."""
        response_data, reasoning_trace = cast(
            Tuple[Dict[str, Any], Optional[str]],
            self._make_api_call(prompt, response_format),
        )
        parsed_response = self._validate_response_structure(
            response_data, response_format
        )
        return parsed_response, reasoning_trace

    async def _get_structured_response_async(
        self, prompt: str, response_format: Type[T]
    ) -> Tuple[T, Optional[str]]:
        """Get structured response from vLLM API (async)."""
        response_data, reasoning_trace = cast(
            Tuple[Dict[str, Any], Optional[str]],
            await self._make_api_call_async(prompt, response_format),
        )
        parsed_response = self._validate_response_structure(
            response_data, response_format
        )
        return parsed_response, reasoning_trace

    def _get_text_response(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Get text response from vLLM API."""
        result = self._make_api_call(prompt)
        return cast(Tuple[str, Optional[str]], result)

    async def _get_text_response_async(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Get text response from vLLM API (async)."""
        result = await self._make_api_call_async(prompt)
        return cast(Tuple[str, Optional[str]], result)


def create_llm_provider(
    provider_name: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
    parse_mode: ParseMode = ParseMode.STRUCTURED,
    enable_reasoning: bool = False,
    openai_reasoning_effort: Optional[str] = None,
    anthropic_thinking_budget: Optional[int] = None,
    max_tokens: int = 64000,
    **kwargs: Any,
) -> LLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider_name: Name of the provider ('mistral', 'openai', 'anthropic', 'vllm')
        api_key: API key for the provider (resolved from environment if None)
        model: Model name to use
        timeout: Request timeout in seconds (None for no timeout)
        parse_mode: Whether to use structured output API or manual JSON parsing
        enable_reasoning: Whether to enable reasoning/thinking for compatible models
        openai_reasoning_effort: Reasoning effort for OpenAI o3 models (low, medium, high)
        anthropic_thinking_budget: Thinking budget for Anthropic models (in tokens, min 1024)
        max_tokens: Maximum number of tokens to generate (default: 64000)
        **kwargs: Additional provider-specific arguments

    Returns:
        LLMProvider instance
    """
    provider_name = provider_name.lower()

    # Prepare reasoning parameters
    reasoning_params: Dict[str, Any] = {}
    if enable_reasoning:
        if provider_name == "openai" and openai_reasoning_effort:
            reasoning_params["effort"] = openai_reasoning_effort
        elif provider_name == "anthropic" and anthropic_thinking_budget:
            reasoning_params["thinking_budget"] = anthropic_thinking_budget

    common_kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "model": model,
        "timeout": timeout,
        "parse_mode": parse_mode,
        "enable_reasoning": enable_reasoning,
        "reasoning_params": reasoning_params,
        "max_tokens": max_tokens,
    }

    if provider_name == "mistral":
        return MistralProvider(**common_kwargs)
    elif provider_name == "openai":
        return OpenAIProvider(**common_kwargs)
    elif provider_name == "anthropic":
        return AnthropicProvider(**common_kwargs)
    elif provider_name == "vllm":
        return VLLMProvider(
            **common_kwargs, base_url=kwargs.get("base_url", DEFAULT_BASE_URL)
        )
    else:
        raise ConfigurationError(
            f"Unknown provider: {provider_name}. "
            "Supported: mistral, openai, anthropic, vllm"
        )
