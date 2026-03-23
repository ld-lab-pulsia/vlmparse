"""Reusable connection parameters for any model-serving endpoint.

Kept in a standalone module (only depends on ``base_model``) so that both
converter configs and page-processor configs can import it without
circular-dependency issues.
"""

from __future__ import annotations

from pydantic import Field

from vlmparse.base_model import VLMParseBaseModel


class ModelEndpointConfig(VLMParseBaseModel):
    """Connection parameters for any model-serving endpoint
    (vLLM, OpenAI, Gemini, Azure, Mistral, …).

    Fields
    ------
    base_url : server endpoint (e.g. ``http://localhost:8000/v1``).
    api_key  : authentication token (empty string → no auth).
    model_name : the *served* model name sent in API calls
        (e.g. ``"vllm-model"`` for local vLLM, ``"gpt-4o"`` for OpenAI).
    timeout  : request timeout in seconds (``None`` → no timeout).
    max_retries : number of automatic retries on transient errors.
    """

    base_url: str | None = None
    api_key: str = Field(default="", exclude=True, repr=False)
    model_name: str = "vllm-model"
    timeout: int | None = None
    max_retries: int = 1


class ImageDescriptionConfig(VLMParseBaseModel):
    """Settings for the image-description post-processor.

    Pass an instance of this class to ``ConverterWithServer`` to enable
    per-item image description.  ``None`` disables the feature.

    Fields
    ------
    connection : endpoint override.  ``None`` means *inherit* connection
        parameters (base_url, api_key, model_name, …) from the main
        converter config.
    categories : item categories eligible for description
        (default: picture, image, figure, chart).
    prompt : instruction sent to the VLM for each crop.  ``None`` keeps
        the processor's built-in default.
    """

    connection: ModelEndpointConfig | None = None
    categories: list[str] = Field(
        default_factory=lambda: ["picture", "image", "figure", "chart"]
    )
    prompt: str | None = None
