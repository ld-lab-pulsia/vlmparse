def clean_response(text):
    """Clean markdown/html markers from response text."""
    return (
        text.strip()
        .removeprefix("")
        .removesuffix("")
        .removeprefix("```")
        .removesuffix("```")
        .removeprefix("markdown")
        .removeprefix("html")
        .strip()
    )
