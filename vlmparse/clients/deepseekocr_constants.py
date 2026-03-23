"""Shared DeepSeek-OCR constants.

Kept in a standalone module (no internal vlmparse imports) so that both
``deepseekocr.py`` and ``item_description_processors.py`` can import from
here without creating a circular dependency.
"""

DEEPSEEK_IMAGE_DESCRIPTION_PROMPT: str = "Describe this image in detail."
DEEPSEEK_WHITELIST_TOKEN_IDS: list[int] = [128821, 128822]


def deepseek_completion_kwargs(
    ngram_size: int, window_size: int, max_tokens: int = 8181
) -> dict:
    return {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "extra_body": {
            "skip_special_tokens": False,
            # args used to control custom logits processor
            "vllm_xargs": {
                "ngram_size": ngram_size,
                "window_size": window_size,
                # whitelist: <td>, </td>
                "whitelist_token_ids": DEEPSEEK_WHITELIST_TOKEN_IDS,
            },
        },
    }


DEEPSEEK_OCR_V1_COMPLETION_KWARGS: dict = deepseek_completion_kwargs(
    ngram_size=30, window_size=90, max_tokens=8181
)
DEEPSEEK_OCR_V2_COMPLETION_KWARGS: dict = deepseek_completion_kwargs(
    ngram_size=20, window_size=50, max_tokens=8170
)
