import re
import unicodedata

# Code adapted from olmocr.bench.tests.normalize_text


def normalize_text(
    md_content: str,
    additional_replacements: dict = {},
    only_alphanum: bool = False,
    remove_md_images: bool = True,
) -> str:
    """Normalise md text"""

    if md_content is None:
        return None

    # Normalize <br> and <br/> to newlines
    md_content = re.sub(r"<br/?>", " ", md_content)

    # Normalize whitespace in the md_content
    md_content = re.sub(r"[ \t]+", " ", md_content)

    # remove_title_emphasis:
    md_content = re.sub(r"[-=]{3,}", "", md_content)

    # remove_space_with_newlines:
    md_content = re.sub(r"\n *", "\n", md_content)

    # remove_more_than_2_newlines:
    md_content = re.sub(r"\n{2,}", "\n\n", md_content)

    # Remove markdown bold formatting (** or __ for bold)
    md_content = re.sub(r"\*\*(.*?)\*\*", r"\1", md_content)
    # md_content = re.sub(r"__(.*?)__", r"\1", md_content)
    md_content = re.sub(r"</?b>", "", md_content)  # Remove <b> tags if they exist
    md_content = re.sub(r"</?i>", "", md_content)  # Remove <i> tags if they exist

    # Remove markdown italics formatting (* or _ for italics)
    md_content = re.sub(r"\*(.*?)\*", r"\1", md_content)
    # md_content = re.sub(r"_(.*?)_", r"\1", md_content)

    # remove_more_than_1_spaces:
    md_content = re.sub(r" {2,}", " ", md_content)

    # Convert down to a consistent unicode form, so é == e + accent, unicode forms
    md_content = unicodedata.normalize("NFC", md_content)

    # Dictionary of characters to replace: keys are fancy characters, values are ASCII equivalents, unicode micro with greek mu comes up often enough too
    replacements = {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "＿": "_",
        "–": "-",
        "—": "-",
        "‑": "-",
        "‒": "-",
        "−": "-",
        "\u00b5": "\u03bc",
    } | additional_replacements

    # Apply all replacements from the dictionary
    for fancy_char, ascii_char in replacements.items():
        md_content = md_content.replace(fancy_char, ascii_char)

    if only_alphanum:
        md_content = re.sub(r"[^a-zA-Z0-9]", "", md_content)
    if remove_md_images:
        md_content = re.sub(r"!\[[\s\S]*?]\([\s\S]*?\)", "", md_content, flags=re.S)
    return md_content.strip()
