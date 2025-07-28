"""
Template loading and rendering system for legal document QA.

Simple wrapper around Jinja2 for prompt templates.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from multidocqa.exceptions import TemplateError

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


class TemplateLoader:
    """Template loader with Jinja2 environment."""

    def __init__(self, template_dir: Path = DEFAULT_TEMPLATE_DIR):
        """Initialize template loader with directory."""
        if not template_dir.exists():
            raise TemplateError(f"Template directory does not exist: {template_dir}")

        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        logger.debug(f"Initialized template loader with directory: {template_dir}")

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Render a template with the provided variables."""
        try:
            template = self.env.get_template(template_name)
            rendered: str = template.render(**kwargs)
            logger.debug(f"Rendered template: {template_name}")
            return rendered.strip()
        except Exception as e:
            raise TemplateError(f"Error rendering template {template_name}: {e}") from e


# Default loader instance
_default_loader = None


def get_default_loader() -> TemplateLoader:
    """Get the default template loader."""
    global _default_loader
    if _default_loader is None:
        _default_loader = TemplateLoader()
    return _default_loader


def render_template(
    template_name: str,
    template_dir: Optional[Path] = None,
    **kwargs: Any,
) -> str:
    """
    Render a template with the provided variables.

    Args:
        template_name: Name of the template file
        template_dir: Template directory (uses default if None)
        **kwargs: Variables to pass to the template

    Returns:
        Rendered template string

    Raises:
        TemplateError: If template cannot be rendered
    """
    if template_dir is None:
        loader = get_default_loader()
    else:
        loader = TemplateLoader(template_dir)

    return loader.render(template_name, **kwargs)


def render_entailment_prompt(
    articles: List[Dict[str, Any]],
    statement: str,
    template_dir: Optional[Path] = None,
) -> str:
    """
    Render the entailment prediction prompt template.

    Args:
        articles: List of civil code articles with 'number' and 'content' keys
        statement: Legal statement to check entailment for
        template_dir: Template directory (uses default if None)

    Returns:
        Rendered prompt string
    """
    return render_template(
        "entailment.txt",
        template_dir=template_dir,
        articles=articles,
        statement=statement,
    )


def render_relevance_prompt(
    article_content: str,
    statement: str,
    article_number: str = "",
    template_dir: Optional[Path] = None,
) -> str:
    """
    Render the relevance prediction prompt template.

    Args:
        article_content: Content of the civil code article
        statement: Legal statement to check relevance for
        article_number: Article number for display
        template_dir: Template directory (uses default if None)

    Returns:
        Rendered prompt string
    """
    return render_template(
        "relevance.txt",
        template_dir=template_dir,
        article_content=article_content,
        statement=statement,
        article_number=article_number,
    )
