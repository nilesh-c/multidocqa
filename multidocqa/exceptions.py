"""
This module defines the exception hierarchy used throughout the codebase
for structured error handling and debugging.
"""

from typing import Optional


class MultidocQAError(Exception):
    """Base exception for all multidocqa-related errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        """
        Initialize the exception.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class LLMError(MultidocQAError):
    """Raised when there's an error with LLM API calls or responses."""

    pass


class ConfigurationError(MultidocQAError):
    """Raised when there's a configuration issue."""

    pass


class DataLoadError(MultidocQAError):
    """Raised when there's an error loading or parsing data files."""

    pass


class ValidationError(MultidocQAError):
    """Raised when data validation fails."""

    pass


class DatasetError(MultidocQAError):
    """Raised when there's an error in dataset construction or processing."""

    pass


class ModelError(MultidocQAError):
    """Raised when there's an error with model loading or inference."""

    pass


class TemplateError(MultidocQAError):
    """Raised when there's an error with prompt template processing."""

    pass
