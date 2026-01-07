# Adapted from https://github.com/allenai/olmocr/blob/main/olmocr/bench/tests.py

import json
import math
import re
import unicodedata
from typing import List, Optional, Set, Tuple

import numpy as np
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
from typing_extensions import Literal
from unidecode import unidecode


class Match:
    def __init__(self, start, end, dist):
        self.start = start
        self.end = end
        self.dist = dist


def find_near_matches(pattern: str, text: str, max_l_dist: int) -> List[Match]:
    if not pattern or not text:
        return []

    matches = []
    pattern_len = len(pattern)

    for window_size in [pattern_len, pattern_len - 1, pattern_len + 1]:
        if window_size <= 0 or window_size > len(text):
            continue

        chunks = [
            (text[i : i + window_size], i) for i in range(len(text) - window_size + 1)
        ]
        if not chunks:
            continue

        result = process.extractOne(
            pattern, [c[0] for c in chunks], scorer=Levenshtein.distance
        )

        if result:
            matched_text, score, idx = result
            dist = int(score)
            if dist <= max_l_dist:
                start_pos = chunks[idx][1]
                matches.append(Match(start_pos, start_pos + window_size, dist))

    return matches


class RepeatDetector:
    def __init__(self, max_ngram_size: int = 10):
        self.max_ngram_size = max_ngram_size
        self.data = ""

    def add_letters(self, new_str: str):
        self.data += new_str

    def ngram_repeats(self) -> list[int]:
        result = [0] * self.max_ngram_size

        if not self.data:
            return result

        # Normalize all whitespace to single spaces
        text = re.sub(r"\s+", " ", self.data)

        # For each n-gram size
        for size in range(1, self.max_ngram_size + 1):
            if len(text) < size:
                continue

            # Get the last n-gram
            target = text[-size:]

            # Count backwards from the end to find repeats
            count = 0
            pos = len(text) - size  # Start position for previous n-gram

            while pos >= 0:
                if text[pos : pos + size] == target:
                    count += 1
                    pos -= size  # Move back by the size of the n-gram
                else:
                    break

            result[size - 1] = count

        return result


class TableData(BaseModel):
    """Class to hold table data and metadata about headers."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        strict=True,
    )

    data: np.ndarray  # The actual table data
    header_rows: Set[int] = Field(
        default_factory=set
    )  # Indices of rows that are headers
    header_cols: Set[int] = Field(
        default_factory=set
    )  # Indices of columns that are headers
    col_headers: dict = Field(
        default_factory=dict
    )  # Maps column index to header text, handling colspan
    row_headers: dict = Field(
        default_factory=dict
    )  # Maps row index to header text, handling rowspan

    def __repr__(self) -> str:
        """Returns a concise representation of the TableData object for debugging."""
        return f"TableData(shape={self.data.shape}, header_rows={len(self.header_rows)}, header_cols={len(self.header_cols)})"

    def __str__(self) -> str:
        """Returns a pretty string representation of the table with header information."""
        output = []

        # Table dimensions
        output.append(
            f"Table: {self.data.shape[0]} rows × {self.data.shape[1]} columns"
        )

        # Header info
        output.append(f"Header rows: {sorted(self.header_rows)}")
        output.append(f"Header columns: {sorted(self.header_cols)}")

        # Table content with formatting
        separator = "+" + "+".join(["-" * 17] * self.data.shape[1]) + "+"

        # Add a header for row indices
        output.append(separator)
        headers = [""] + [f"Column {i}" for i in range(self.data.shape[1])]
        output.append(
            "| {:<5} | ".format("Row")
            + " | ".join(["{:<15}".format(h) for h in headers[1:]])
            + " |"
        )
        output.append(separator)

        # Format each row
        for i in range(min(self.data.shape[0], 15)):  # Limit to 15 rows for readability
            # Format cells, mark header cells
            cells = []
            for j in range(self.data.shape[1]):
                cell = str(self.data[i, j])
                if len(cell) > 15:
                    cell = cell[:12] + "..."
                # Mark header cells with *
                if i in self.header_rows or j in self.header_cols:
                    cell = f"*{cell}*"
                cells.append(cell)

            row_str = (
                "| {:<5} | ".format(i)
                + " | ".join(["{:<15}".format(c) for c in cells])
                + " |"
            )
            output.append(row_str)
            output.append(separator)

        # If table is too large, indicate truncation
        if self.data.shape[0] > 15:
            output.append(f"... {self.data.shape[0] - 15} more rows ...")

        # Column header details if available
        if self.col_headers:
            output.append("\nColumn header mappings:")
            for col, headers in sorted(self.col_headers.items()):
                header_strs = [f"({row}, '{text}')" for row, text in headers]
                output.append(f"  Column {col}: {', '.join(header_strs)}")

        # Row header details if available
        if self.row_headers:
            output.append("\nRow header mappings:")
            for row, headers in sorted(self.row_headers.items()):
                header_strs = [f"({col}, '{text}')" for col, text in headers]
                output.append(f"  Row {row}: {', '.join(header_strs)}")

        return "\n".join(output)


TestType = Literal["baseline", "present", "absent", "order", "table", "math"]


TestChecked = Literal["verified", "rejected"]


class ValidationError(Exception):
    """Exception raised for validation errors."""

    pass


def normalize_text(md_content: str) -> str:
    if md_content is None:
        return None

    # Normalize <br> and <br/> to newlines
    md_content = re.sub(r"<br/?>", " ", md_content)

    # Normalize whitespace in the md_content
    md_content = re.sub(r"\s+", " ", md_content)

    # Remove markdown bold formatting (** or __ for bold)
    md_content = re.sub(r"\*\*(.*?)\*\*", r"\1", md_content)
    md_content = re.sub(r"\\\*(.*?)\\\*", r"\1", md_content)
    md_content = re.sub(r"__(.*?)__", r"\1", md_content)
    md_content = re.sub(r"</?b>", "", md_content)  # Remove <b> tags if they exist
    md_content = re.sub(r"</?i>", "", md_content)  # Remove <i> tags if they exist

    # Remove markdown italics formatting (* or _ for italics)
    md_content = re.sub(r"\*(.*?)\*", r"\1", md_content)
    md_content = re.sub(r"_(.*?)_", r"\1", md_content)

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
        "º": "°",
        "œ": "oe",
        r"\*": "",
        r"\*\*": "",
        "’": "'",  # noqa
        "« ": "«",
        " »": "»",
        " .": ".",
        " :": ":",
        " ,": ",",
        "É": "E",
        "☑": "[x]",
        "☐": "[ ]",
        "☒": "[x]",
        "✅": "[x]",
        "❌": "[x]",
        "❎": "[x]",
        "✓": "[x]",
        "✔": "[x]",
        "✗": "[x]",
        "✖": "[x]",
        "🗹": "[x]",
        "[X]": "[x]",
    }
    for fancy_char, ascii_char in replacements.items():
        md_content = md_content.replace(fancy_char, ascii_char)

    return md_content


def format_diff_text(reference: str, found: str) -> str:
    from rapidfuzz.distance import Indel

    opcodes = Indel.opcodes(reference, found)
    result = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            result.append(reference[i1:i2])
        elif tag == "delete":
            result.append(f":red-background[{reference[i1:i2]}]")
        elif tag == "insert":
            result.append(f":green-background[{found[j1:j2]}]")
        elif tag == "replace":
            result.append(f":red-background[{reference[i1:i2]}]")
            result.append(f":green-background[{found[j1:j2]}]")
    return "".join(result)


def parse_markdown_tables(md_content: str) -> List[TableData]:
    """
    Extract and parse all markdown tables from the provided content.
    Uses a direct approach to find and parse tables, which is more robust for tables
    at the end of files or with irregular formatting.

    Args:
        md_content: The markdown content containing tables

    Returns:
        A list of TableData objects, each containing the table data and header information
    """
    # Split the content into lines and process line by line
    lines = md_content.strip().split("\n")

    parsed_tables = []
    current_table_lines = []
    in_table = False

    # Identify potential tables by looking for lines with pipe characters
    for _, line in enumerate(lines):
        # Check if this line has pipe characters (a table row indicator)
        if "|" in line:
            # If we weren't in a table before, start a new one
            if not in_table:
                in_table = True
                current_table_lines = [line]
            else:
                # Continue adding to the current table
                current_table_lines.append(line)
        else:
            # No pipes in this line, so if we were in a table, we've reached its end
            if in_table:
                # Process the completed table if it has at least 2 rows
                if len(current_table_lines) >= 2:
                    table_data = _process_table_lines(current_table_lines)
                    if table_data and len(table_data) > 0:
                        # Convert to numpy array for easier manipulation
                        max_cols = max(len(row) for row in table_data)
                        padded_data = [
                            row + [""] * (max_cols - len(row)) for row in table_data
                        ]
                        table_array = np.array(padded_data)

                        # In markdown tables, the first row is typically a header row
                        header_rows = {0} if len(table_array) > 0 else set()

                        # Set up col_headers with first row headers for each column
                        col_headers = {}
                        if len(table_array) > 0:
                            for col_idx in range(table_array.shape[1]):
                                if col_idx < len(table_array[0]):
                                    col_headers[col_idx] = [
                                        (0, table_array[0, col_idx])
                                    ]

                        # Set up row_headers with first column headers for each row
                        row_headers = {}
                        if table_array.shape[1] > 0:
                            for row_idx in range(
                                1, table_array.shape[0]
                            ):  # Skip header row
                                row_headers[row_idx] = [
                                    (0, table_array[row_idx, 0])
                                ]  # First column as heading

                        # Create TableData object
                        parsed_tables.append(
                            TableData(
                                data=table_array,
                                header_rows=header_rows,
                                header_cols={0}
                                if table_array.shape[1] > 0
                                else set(),  # First column as header
                                col_headers=col_headers,
                                row_headers=row_headers,
                            )
                        )
                in_table = False

    # Process the last table if we're still tracking one at the end of the file
    if in_table and len(current_table_lines) >= 2:
        table_data = _process_table_lines(current_table_lines)
        if table_data and len(table_data) > 0:
            # Convert to numpy array
            max_cols = max(len(row) for row in table_data)
            padded_data = [row + [""] * (max_cols - len(row)) for row in table_data]
            table_array = np.array(padded_data)

            # In markdown tables, the first row is typically a header row
            header_rows = {0} if len(table_array) > 0 else set()

            # Set up col_headers with first row headers for each column
            col_headers = {}
            if len(table_array) > 0:
                for col_idx in range(table_array.shape[1]):
                    if col_idx < len(table_array[0]):
                        col_headers[col_idx] = [(0, table_array[0, col_idx])]

            # Set up row_headers with first column headers for each row
            row_headers = {}
            if table_array.shape[1] > 0:
                for row_idx in range(1, table_array.shape[0]):  # Skip header row
                    row_headers[row_idx] = [
                        (0, table_array[row_idx, 0])
                    ]  # First column as heading

            # Create TableData object
            parsed_tables.append(
                TableData(
                    data=table_array,
                    header_rows=header_rows,
                    header_cols={0}
                    if table_array.shape[1] > 0
                    else set(),  # First column as header
                    col_headers=col_headers,
                    row_headers=row_headers,
                )
            )

    return parsed_tables


def _process_table_lines(table_lines: List[str]) -> List[List[str]]:
    """
    Process a list of lines that potentially form a markdown table.

    Args:
        table_lines: List of strings, each representing a line in a potential markdown table

    Returns:
        A list of rows, each a list of cell values
    """
    table_data = []
    separator_row_index = None

    # First, identify the separator row (the row with dashes)
    for i, line in enumerate(table_lines):
        # Check if this looks like a separator row (contains mostly dashes)
        content_without_pipes = line.replace("|", "").strip()
        if content_without_pipes and all(c in "- :" for c in content_without_pipes):
            separator_row_index = i
            break

    # Process each line, filtering out the separator row
    for i, line in enumerate(table_lines):
        # Skip the separator row
        if i == separator_row_index:
            continue

        # Skip lines that are entirely formatting
        if line.strip() and all(c in "- :|" for c in line):
            continue

        # Process the cells in this row
        cells = [cell.strip() for cell in line.split("|")]

        # Remove empty cells at the beginning and end (caused by leading/trailing pipes)
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]

        if cells:  # Only add non-empty rows
            table_data.append(cells)

    return table_data


def parse_html_tables(html_content: str) -> List[TableData]:
    """
    Extract and parse all HTML tables from the provided content.
    Identifies header rows and columns, and maps them properly handling rowspan/colspan.

    Args:
        html_content: The HTML content containing tables

    Returns:
        A list of TableData objects, each containing the table data and header information
    """
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")

    parsed_tables = []

    for table in tables:
        rows = table.find_all(["tr"])
        table_data = []
        header_rows = set()
        header_cols = set()
        col_headers = {}  # Maps column index to all header cells above it
        row_headers = {}  # Maps row index to all header cells to its left

        # Find rows inside thead tags - these are definitely header rows
        thead = table.find("thead")
        if thead:
            thead_rows = thead.find_all("tr")
            for tr in thead_rows:
                header_rows.add(rows.index(tr))

        # Initialize a grid to track filled cells due to rowspan/colspan
        cell_grid = {}
        col_span_info = {}  # Tracks which columns contain headers
        row_span_info = {}  # Tracks which rows contain headers

        # First pass: process each row to build the raw table data and identify headers
        for row_idx, row in enumerate(rows):
            cells = row.find_all(["th", "td"])
            row_data = []
            col_idx = 0

            # If there are th elements in this row, it's likely a header row
            if row.find("th"):
                header_rows.add(row_idx)

            for cell in cells:
                # Skip positions already filled by rowspans from above
                while (row_idx, col_idx) in cell_grid:
                    row_data.append(cell_grid[(row_idx, col_idx)])
                    col_idx += 1

                # Replace <br> and <br/> tags with newlines before getting text
                for br in cell.find_all("br"):
                    br.replace_with("\n")
                cell_text = cell.get_text().strip()

                # Handle rowspan/colspan
                rowspan = int(cell.get("rowspan", 1))
                colspan = int(cell.get("colspan", 1))

                # Add the cell to the row data
                row_data.append(cell_text)

                # Fill the grid for this cell and its rowspan/colspan
                for i in range(rowspan):
                    for j in range(colspan):
                        if i == 0 and j == 0:
                            continue  # Skip the main cell position
                        # For rowspan cells, preserve the text in all spanned rows
                        if j == 0 and i > 0:  # Only for cells directly below
                            cell_grid[(row_idx + i, col_idx + j)] = cell_text
                        else:
                            cell_grid[(row_idx + i, col_idx + j)] = (
                                ""  # Mark other spans as empty
                            )

                # If this is a header cell (th), mark it and its span
                if cell.name == "th":
                    # Mark columns as header columns
                    for j in range(colspan):
                        header_cols.add(col_idx + j)

                    # For rowspan, mark spanned rows as part of header
                    for i in range(1, rowspan):
                        if row_idx + i < len(rows):
                            header_rows.add(row_idx + i)

                    # Record this header for all spanned columns
                    for j in range(colspan):
                        curr_col = col_idx + j
                        if curr_col not in col_headers:
                            col_headers[curr_col] = []
                        col_headers[curr_col].append((row_idx, cell_text))

                        # Store which columns are covered by this header
                        if cell_text and colspan > 1:
                            if cell_text not in col_span_info:
                                col_span_info[cell_text] = set()
                            col_span_info[cell_text].add(curr_col)

                    # Store which rows are covered by this header for rowspan
                    if cell_text and rowspan > 1:
                        if cell_text not in row_span_info:
                            row_span_info[cell_text] = set()
                        for i in range(rowspan):
                            row_span_info[cell_text].add(row_idx + i)

                # Also handle row headers from data cells that have rowspan
                if cell.name == "td" and rowspan > 1 and col_idx in header_cols:
                    for i in range(1, rowspan):
                        if row_idx + i < len(rows):
                            if row_idx + i not in row_headers:
                                row_headers[row_idx + i] = []
                            row_headers[row_idx + i].append((col_idx, cell_text))

                col_idx += colspan

            # Pad the row if needed to handle different row lengths
            table_data.append(row_data)

        # Second pass: expand headers to cells that should inherit them
        # First handle column headers
        for header_text, columns in col_span_info.items():
            for col in columns:
                # Add this header to all columns it spans over
                for row_idx in range(len(table_data)):
                    if row_idx not in header_rows:  # Only apply to data rows
                        for j in range(
                            col,
                            len(table_data[row_idx])
                            if row_idx < len(table_data)
                            else 0,
                        ):
                            # Add header info to data cells in these columns
                            if j not in col_headers:
                                col_headers[j] = []
                            if not any(h[1] == header_text for h in col_headers[j]):
                                header_row = min(
                                    [r for r, t in col_headers.get(col, [(0, "")])]
                                )
                                col_headers[j].append((header_row, header_text))

        # Handle row headers
        for header_text, rows in row_span_info.items():
            for row in rows:
                if row < len(table_data):
                    # Find first header column
                    header_col = min(header_cols) if header_cols else 0
                    if row not in row_headers:
                        row_headers[row] = []
                    if not any(h[1] == header_text for h in row_headers.get(row, [])):
                        row_headers[row].append((header_col, header_text))

        # Process regular row headers - each cell in a header column becomes a header for its row
        for col_idx in header_cols:
            for row_idx, row in enumerate(table_data):
                if col_idx < len(row) and row[col_idx].strip():
                    if row_idx not in row_headers:
                        row_headers[row_idx] = []
                    if not any(
                        h[1] == row[col_idx] for h in row_headers.get(row_idx, [])
                    ):
                        row_headers[row_idx].append((col_idx, row[col_idx]))

        # Calculate max columns for padding
        max_cols = max(len(row) for row in table_data) if table_data else 0

        # Ensure all rows have the same number of columns
        if table_data:
            padded_data = [row + [""] * (max_cols - len(row)) for row in table_data]
            table_array = np.array(padded_data)

            # Create TableData object with the table and header information
            parsed_tables.append(
                TableData(
                    data=table_array,
                    header_rows=header_rows,
                    header_cols=header_cols,
                    col_headers=col_headers,
                    row_headers=row_headers,
                )
            )

    return parsed_tables


class PageMetadata(BaseModel):
    doc_type: Literal["long_text", "multi_page", "large_table"] | None = None
    original_doc_path: str
    pdf: str
    page: int = Field(ge=0)


class BasePDFTest(BaseModel):
    """
    Base class for all PDF test types.

    Attributes:
        pdf: The PDF filename.
        page: The page number for the test.
        id: Unique identifier for the test.
        type: The type of test.
        threshold: A float between 0 and 1 representing the threshold for fuzzy matching.
    """

    pdf: str = Field(min_length=1)
    page: int
    id: str = Field(min_length=1)
    type: TestType
    max_diffs: int = Field(ge=0, default=0)
    alphanum: bool = False
    """Filter only on alphanumeric characters + dots and commas"""
    unidecode: bool = False
    """Convert text to ASCII using unidecode"""
    ignore_space_and_newlines: bool = False
    """Ignore space and newlines in the text deprecated, use ignore_space and ignore_newlines instead"""
    ignore_space: bool = False
    """Ignore space in the text"""
    ignore_newlines: bool = True
    """Ignore newlines in the text"""
    ignore_chars: str = ""
    """Characters to ignore in the text"""
    ignore_str: list[str] = []
    """Strings to ignore in the text"""
    checked: Optional[TestChecked] | bool = None
    url: Optional[str] = None
    category: Optional[str] = None
    """subcategory of the test to identify what the test is supposed to measure"""
    display_diffs: bool = True
    """Whether to display diffs in the explanation"""

    def normalise(self, text: str) -> str:
        text = normalize_text(text)
        if self.unidecode:
            text = unidecode(text, errors="preserve")

        if self.alphanum:
            text = re.sub(r"[^a-zA-Z0-9\.,:;\+\(\)\'\"]", "", text).lower()
            # text = text.replace(",", ".")
            # text = text.replace(";", ":")

        # if self.ignore_space_and_newlines:
        #     text = re.sub(r"\s+", "", text)

        if self.ignore_space:
            text = re.sub(r"[^\S\r\n]+", "", text)
        if self.ignore_newlines:
            text = re.sub(r"\n+", "", text)

        if self.ignore_chars:
            text = re.sub(f"[{self.ignore_chars}]", "", text)

        if self.ignore_str:
            for _str in self.ignore_str:
                text = text.replace(_str, "")

        return text

    def get_diff(self, reference: str, candidate: str) -> str:
        if self.display_diffs:
            matches = find_near_matches(
                reference, candidate, max_l_dist=len(reference) // 2
            )
            if matches:
                best_match = min(matches, key=lambda m: m.dist)
                best_match_text = candidate[best_match.start : best_match.end]
                diff_display = format_diff_text(reference, best_match_text)
                return diff_display

    def run(self, md_content: str) -> Tuple[bool, str, float]:
        """
        Run the test on the provided markdown content.

        Args:
            md_content: The content of the .md file.

        Returns:
            A tuple (passed, explanation) where 'passed' is True if the test passes,
            and 'explanation' provides details when the test fails.
        """
        raise NotImplementedError("Subclasses must implement the run method")

    def __repr__(self):
        from devtools import PrettyFormat

        pformat = PrettyFormat()
        return pformat(self, highlight=False)


class TextPresenceTest(BasePDFTest):
    """
    Test to verify the presence or absence of specific text in a PDF.

    Attributes:
        text: The text string to search for.
    """

    text: str = Field()
    case_sensitive: bool = True
    first_n: Optional[int] = None
    last_n: Optional[int] = None
    type: Literal["present", "absent"] = Field(default="present")
    layout_cat: Literal[
        "text", "footer", "header", "footnote", "image", "image_caption"
    ] = Field(default="text")

    def run(self, md_content: str) -> Tuple[bool, str]:
        reference_query = self.normalise(self.text)

        # Normalize whitespace in the md_content
        md_content_n = self.normalise(md_content)

        if not self.case_sensitive:
            reference_query = reference_query.lower()
            md_content_n = md_content_n.lower()

        if self.first_n and self.last_n:
            md_content_n = md_content_n[: self.first_n] + md_content_n[-self.last_n :]
        elif self.first_n:
            md_content_n = md_content_n[: self.first_n]
        elif self.last_n:
            md_content_n = md_content_n[-self.last_n :]

        # Threshold for fuzzy matching derived from max_diffs
        threshold = 1.0 - (
            self.max_diffs / (len(reference_query) if len(reference_query) > 0 else 1)
        )
        best_ratio = fuzz.partial_ratio(reference_query, md_content_n) / 100.0

        if self.type == "present":
            if best_ratio >= threshold:
                return True, "", best_ratio
            else:
                best_match_text = ""
                diff_display = "No match found"
                if md_content:
                    diff_display = self.get_diff(reference_query, md_content_n)
                msg = (
                    f"Expected '{reference_query[:40]}' with threshold {threshold} "
                    f"but best match ratio was {best_ratio:.3f}\n"
                    f"Diff:\n\n{diff_display}"
                )
                return False, msg, best_ratio
        else:  # ABSENT
            if best_ratio < threshold:
                return True, "", 1 - best_ratio
            else:
                reference = reference_query  # normalize_text(self.text)

                best_match_text = ""
                diff_display = "No match found"
                if md_content:
                    matches = find_near_matches(
                        reference, md_content, max_l_dist=len(reference) // 2
                    )
                    if matches:
                        best_match = min(matches, key=lambda m: m.dist)
                        best_match_text = md_content[best_match.start : best_match.end]
                        diff_display = format_diff_text(reference, best_match_text)
                msg = (
                    f"Expected absence of '{reference[:40]}' with threshold {threshold} "
                    f"but best match ratio was {best_ratio:.3f}\n"
                    f"Diff:\n\n{diff_display}"
                )
                return False, msg, 1 - best_ratio


class TextOrderTest(BasePDFTest):
    """
    Test to verify that one text appears before another in a PDF.

    Attributes:
        before: The text expected to appear first.
        after: The text expected to appear after the 'before' text.
    """

    before: str
    after: str
    type: Literal["order"] = Field(default="order")

    @model_validator(mode="after")
    def validate_max_diffs(self):
        if (
            self.max_diffs > len(self.before) // 2
            or self.max_diffs > len(self.after) // 2
        ):
            raise ValidationError(
                "Max diffs is too large for this test, greater than 50% of the search string"
            )
        return self

    def run(self, md_content: str) -> Tuple[bool, str, float]:
        md_content = self.normalise(md_content)
        before = self.normalise(self.before)
        after = self.normalise(self.after)

        before_matches = find_near_matches(
            before, md_content, max_l_dist=self.max_diffs
        )
        after_matches = find_near_matches(after, md_content, max_l_dist=self.max_diffs)

        if not before_matches:
            return (
                False,
                f"'before' text '{before[:40]}...' not found with max_l_dist {self.max_diffs}",
                0.0,
            )
        if not after_matches:
            return (
                False,
                f"'after' text '{after[:40]}...' not found with max_l_dist {self.max_diffs}",
                0.0,
            )

        for before_match in before_matches:
            for after_match in after_matches:
                if before_match.start < after_match.start:
                    return (
                        True,
                        "",
                        min(before_match.dist, after_match.dist)
                        / max(len(before), len(after)),
                    )
        return (
            False,
            f"Could not find a location where '{before[:40]}...' appears before "
            f"'{after[:40]}...'.",
            0.0,
        )


class TableTest(BasePDFTest):
    cell: str
    up: str = ""
    down: str = ""
    left: str = ""
    right: str = ""
    type: Literal["table"] = Field(default="table")
    top_heading: str = ""
    left_heading: str = ""

    def run(self, content: str) -> Tuple[bool, str, float]:
        from vlmparse.clients.pipe_utils.html_to_md_conversion import md_tables_to_html

        content = md_tables_to_html(content)
        # print(content)

        cell = self.normalise(self.cell)
        up = self.normalise(self.up)
        down = self.normalise(self.down)
        left = self.normalise(self.left)
        right = self.normalise(self.right)
        top_heading = self.normalise(self.top_heading)
        left_heading = self.normalise(self.left_heading)

        threshold = max(
            0.5, 1.0 - (self.max_diffs / (len(cell) if len(cell) > 0 else 1))
        )

        soup = BeautifulSoup(content, "html.parser")
        tables = soup.find_all("table")

        if not tables:
            return False, "No HTML tables found in the content", 0.0

        best_match_score = -1
        best_match_reasons = []

        for table in tables:
            rows = table.find_all("tr")
            cells_info = []
            occupied = {}

            for row_idx, row in enumerate(rows):
                cells = row.find_all(["th", "td"])
                col_idx = 0

                for html_cell in cells:
                    while (row_idx, col_idx) in occupied:
                        col_idx += 1
                    for br in html_cell.find_all("br"):
                        br.replace_with("\n")
                    cell_text_orig = html_cell.get_text().strip()
                    cell_text = self.normalise(cell_text_orig)

                    rowspan = int(html_cell.get("rowspan", 1))
                    colspan = int(html_cell.get("colspan", 1))
                    is_header = html_cell.name == "th"

                    cells_info.append(
                        {
                            "row": row_idx,
                            "col": col_idx,
                            "rowspan": rowspan,
                            "colspan": colspan,
                            "text": cell_text,
                            "text_orig": cell_text_orig,
                            "is_header": is_header,
                        }
                    )

                    for i in range(rowspan):
                        for j in range(colspan):
                            occupied[(row_idx + i, col_idx + j)] = True

                    col_idx += colspan

            if not cells_info:
                continue

            best_cell = None
            best_similarity = -1

            for cell_info in cells_info:
                cell_content = cell_info["text"]
                similarity = fuzz.ratio(cell, cell_content) / 100.0

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cell = cell_info

                if similarity < threshold:
                    continue

                all_satisfied = True
                reasons = []
                total_score = similarity

                row_start = cell_info["row"]
                row_end = row_start + cell_info["rowspan"]
                col_start = cell_info["col"]
                col_end = col_start + cell_info["colspan"]

                if up:
                    up_neighbors = [
                        c
                        for c in cells_info
                        if c != cell_info
                        and c["row"] + c["rowspan"] == row_start
                        and not (
                            c["col"] >= col_end or c["col"] + c["colspan"] <= col_start
                        )
                    ]
                    if up_neighbors:
                        up_sim = [
                            fuzz.ratio(up, n["text"]) / 100.0 for n in up_neighbors
                        ]
                        best_up_sim = max(up_sim)
                        best_up_neighbors = up_neighbors[np.argmax(up_sim)]
                        total_score += best_up_sim
                        if best_up_sim < max(
                            0.5,
                            1.0 - (self.max_diffs / (len(up) if len(up) > 0 else 1)),
                        ):
                            all_satisfied = False
                            diff_display = self.get_diff(up, best_up_neighbors["text"])
                            reasons.append(
                                f"Up cell not found (sim: {best_up_sim:.2f})\nDiff:\n\n{diff_display}"
                            )
                    else:
                        all_satisfied = False
                        reasons.append("Up cell not found (sim: 0.00)")

                if down:
                    down_neighbors = [
                        c
                        for c in cells_info
                        if c != cell_info
                        and c["row"] == row_end
                        and not (
                            c["col"] >= col_end or c["col"] + c["colspan"] <= col_start
                        )
                    ]
                    if down_neighbors:
                        down_sim = [
                            fuzz.ratio(down, n["text"]) / 100.0 for n in down_neighbors
                        ]
                        best_down_sim = max(down_sim)
                        best_down_neighbors = down_neighbors[np.argmax(down_sim)]

                        total_score += best_down_sim
                        if best_down_sim < max(
                            0.5,
                            1.0
                            - (self.max_diffs / (len(down) if len(down) > 0 else 1)),
                        ):
                            all_satisfied = False
                            diff_display = self.get_diff(
                                down, best_down_neighbors["text"]
                            )
                            reasons.append(
                                f"Down cell not found (sim: {best_down_sim:.2f})\nDiff:\n\n{diff_display}"
                            )
                    else:
                        all_satisfied = False
                        reasons.append("Down cell not found (sim: 0.00)")

                if left:
                    left_neighbors = [
                        c
                        for c in cells_info
                        if c != cell_info
                        and c["col"] + c["colspan"] == col_start
                        and not (
                            c["row"] >= row_end or c["row"] + c["rowspan"] <= row_start
                        )
                    ]
                    if left_neighbors:
                        left_sim = [
                            fuzz.ratio(left, n["text"]) / 100.0 for n in left_neighbors
                        ]

                        best_left_cell_sim = max(left_sim)
                        best_left_cell = left_neighbors[np.argmax(left_sim)]
                        total_score += best_left_cell_sim
                        if best_left_cell_sim < max(
                            0.5,
                            1.0
                            - (self.max_diffs / (len(left) if len(left) > 0 else 1)),
                        ):
                            all_satisfied = False
                            diff_display = self.get_diff(left, best_left_cell["text"])
                            reasons.append(
                                f"Left cell not found (sim: {best_left_cell_sim:.2f})\nDiff:\n\n{diff_display}"
                            )
                    else:
                        all_satisfied = False
                        reasons.append("Left cell not found (sim: 0.00)")

                if right:
                    right_neighbors = [
                        c
                        for c in cells_info
                        if c != cell_info
                        and c["col"] == col_end
                        and not (
                            c["row"] >= row_end or c["row"] + c["rowspan"] <= row_start
                        )
                    ]
                    if right_neighbors:
                        right_sim = [
                            fuzz.ratio(right, n["text"]) / 100.0
                            for n in right_neighbors
                        ]
                        best_right_cell_sim = max(right_sim)
                        best_right_cell = right_neighbors[np.argmax(right_sim)]
                        total_score += best_right_cell_sim
                        if best_right_cell_sim < max(
                            0.5,
                            1.0
                            - (self.max_diffs / (len(right) if len(right) > 0 else 1)),
                        ):
                            all_satisfied = False
                            diff_display = self.get_diff(right, best_right_cell["text"])
                            reasons.append(
                                f"Right cell not found (sim: {best_right_cell_sim:.2f})\nDiff:\n\n{diff_display}"
                            )
                    else:
                        all_satisfied = False
                        reasons.append("Right cell not found (sim: 0.00)")

                if top_heading:
                    header_cells = [
                        c
                        for c in cells_info
                        if c["is_header"]
                        and not (
                            c["col"] >= col_end or c["col"] + c["colspan"] <= col_start
                        )
                    ]
                    if header_cells:
                        headers_sim = [
                            fuzz.ratio(top_heading, n["text"]) / 100.0
                            for n in header_cells
                        ]
                        best_header_cell_sim = max(headers_sim)
                        best_header_cell = header_cells[np.argmax(headers_sim)]
                        total_score += best_header_cell_sim
                        if best_header_cell_sim < max(
                            0.5,
                            1.0
                            - (
                                self.max_diffs
                                / (len(top_heading) if len(top_heading) > 0 else 1)
                            ),
                        ):
                            all_satisfied = False
                            diff_display = self.get_diff(
                                top_heading, best_header_cell["text"]
                            )
                            reasons.append(
                                f"Top heading not found (sim: {best_header_cell_sim:.2f})\nDiff:\n\n{diff_display}"
                            )
                    else:
                        all_satisfied = False
                        reasons.append("Top heading not found (sim: 0.00)")

                if left_heading:
                    header_cells = [
                        c
                        for c in cells_info
                        if c["col"] == 0
                        and not (
                            c["row"] >= row_end or c["row"] + c["rowspan"] <= row_start
                        )
                    ]
                    if header_cells:
                        headers_sim = [
                            fuzz.ratio(left_heading, n["text"]) / 100.0
                            for n in header_cells
                        ]
                        best_header_cell_sim = max(headers_sim)
                        best_header_cell = header_cells[np.argmax(headers_sim)]
                        total_score += best_header_cell_sim
                        if best_header_cell_sim < max(
                            0.5,
                            1.0
                            - (
                                self.max_diffs
                                / (len(left_heading) if len(left_heading) > 0 else 1)
                            ),
                        ):
                            all_satisfied = False
                            diff_display = self.get_diff(
                                left_heading, best_header_cell["text"]
                            )
                            reasons.append(
                                f"Left heading not found (sim: {best_header_cell_sim:.2f})\nDiff:\n\n{diff_display}"
                            )
                    else:
                        all_satisfied = False
                        reasons.append("Left heading not found (sim: 0.00)")

                if all_satisfied:
                    return True, "", best_match_score

                if total_score > best_match_score:
                    best_match_score = total_score
                    best_match_reasons = reasons

        if best_match_score < 0:
            if best_cell:
                diff_display = self.get_diff(left_heading, best_cell["text"])
            else:
                diff_display = ""
            return (
                False,
                f"No cell matching '{cell}' found with threshold {threshold}\nDiff:\n\n{diff_display}",
                best_match_score,
            )
        else:
            exp = "\n\n".join(best_match_reasons)
            return (
                False,
                f"Found cells matching '{cell}' but relationships not satisfied: {exp}",
                best_match_score,
            )


class TableTestOld(BasePDFTest):
    """
    Test to verify certain properties of a table are held, namely that some cells appear relative to other cells correctly
    """

    cell: str
    up: str = ""
    down: str = ""
    left: str = ""
    right: str = ""
    type: Literal["table"] = Field(default="table")
    top_heading: str = ""
    left_heading: str = ""

    def run(self, content: str) -> Tuple[bool, str]:
        """
        Run the table test on provided content.

        Finds all tables (markdown and/or HTML based on content_type) and checks if any cell
        matches the target cell and satisfies the specified relationships.

        Args:
            content: The content containing tables (markdown or HTML)

        Returns:
            A tuple (passed, explanation) where 'passed' is True if the test passes,
            and 'explanation' provides details when the test fails.
        """
        cell = self.normalise(self.cell)
        up = self.normalise(self.up)
        down = self.normalise(self.down)
        left = self.normalise(self.left)
        right = self.normalise(self.right)
        top_heading = self.normalise(self.top_heading)
        left_heading = self.normalise(self.left_heading)
        # Initialize variables to track tables and results
        tables_to_check = []
        failed_reasons = []

        # Threshold for fuzzy matching derived from max_diffs
        threshold = 1.0 - (self.max_diffs / (len(cell) if len(cell) > 0 else 1))
        threshold = max(0.5, threshold)

        # Parse tables based on content_type
        md_tables = parse_markdown_tables(content)
        tables_to_check.extend(md_tables)

        html_tables = parse_html_tables(content)
        tables_to_check.extend(html_tables)

        # If no tables found, return failure
        if not tables_to_check:
            return False, "No tables found in the content"

        # Check each table
        for table_data in tables_to_check:
            # Removed debug print statement
            table_array = table_data.data
            header_rows = table_data.header_rows
            header_cols = table_data.header_cols

            # Find all cells that match the target cell using fuzzy matching
            matches = []
            for i in range(table_array.shape[0]):
                for j in range(table_array.shape[1]):
                    cell_content = self.normalise(table_array[i, j])
                    similarity = fuzz.ratio(cell, cell_content) / 100.0

                    if similarity >= threshold:
                        matches.append((i, j))

            # If no matches found in this table, continue to the next table
            if not matches:
                continue

            # Check the relationships for each matching cell
            for row_idx, col_idx in matches:
                all_relationships_satisfied = True
                current_failed_reasons = []

                # Check up relationship
                if up and row_idx > 0:
                    up_cell = self.normalise(table_array[row_idx - 1, col_idx])
                    up_similarity = fuzz.ratio(up, up_cell) / 100.0
                    if up_similarity < max(
                        0.5,
                        1.0 - (self.max_diffs / (len(up) if len(up) > 0 else 1)),
                    ):
                        all_relationships_satisfied = False
                        current_failed_reasons.append(
                            f"Cell above '{up_cell}' doesn't match expected '{up}' (similarity: {up_similarity:.2f})"
                        )

                # Check down relationship
                if down and row_idx < table_array.shape[0] - 1:
                    down_cell = self.normalise(table_array[row_idx + 1, col_idx])
                    down_similarity = fuzz.ratio(down, down_cell) / 100.0
                    if down_similarity < max(
                        0.5,
                        1.0 - (self.max_diffs / (len(down) if len(down) > 0 else 1)),
                    ):
                        all_relationships_satisfied = False
                        current_failed_reasons.append(
                            f"Cell below '{down_cell}' doesn't match expected '{down}' (similarity: {down_similarity:.2f})"
                        )

                # Check left relationship
                if left and col_idx > 0:
                    left_cell = self.normalise(table_array[row_idx, col_idx - 1])
                    left_similarity = fuzz.ratio(left, left_cell) / 100.0
                    if left_similarity < max(
                        0.5,
                        1.0 - (self.max_diffs / (len(left) if len(left) > 0 else 1)),
                    ):
                        all_relationships_satisfied = False
                        current_failed_reasons.append(
                            f"Cell to the left '{left_cell}' doesn't match expected '{left}' (similarity: {left_similarity:.2f})"
                        )

                # Check right relationship
                if right and col_idx < table_array.shape[1] - 1:
                    right_cell = self.normalise(table_array[row_idx, col_idx + 1])
                    right_similarity = fuzz.ratio(right, right_cell) / 100.0
                    if right_similarity < max(
                        0.5,
                        1.0 - (self.max_diffs / (len(right) if len(right) > 0 else 1)),
                    ):
                        all_relationships_satisfied = False
                        current_failed_reasons.append(
                            f"Cell to the right '{right_cell}' doesn't match expected '{right}' (similarity: {right_similarity:.2f})"
                        )

                # Check top heading relationship
                if top_heading:
                    # Try to find a match in the column headers
                    top_heading_found = False
                    best_match = ""
                    best_similarity = 0

                    # Check the col_headers dictionary first (this handles colspan properly)
                    if col_idx in table_data.col_headers:
                        for _, header_text in table_data.col_headers[col_idx]:
                            header_text = self.normalise(header_text)
                            similarity = fuzz.ratio(top_heading, header_text) / 100.0
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = header_text
                                if best_similarity >= max(
                                    0.5,
                                    1.0
                                    - (
                                        self.max_diffs
                                        / (
                                            len(top_heading)
                                            if len(top_heading) > 0
                                            else 1
                                        )
                                    ),
                                ):
                                    top_heading_found = True
                                    break

                    # If no match found in col_headers, fall back to checking header rows
                    if not top_heading_found and header_rows:
                        for i in sorted(header_rows):
                            if i < row_idx and table_array[i, col_idx].strip():
                                header_text = self.normalise(table_array[i, col_idx])
                                similarity = (
                                    fuzz.ratio(top_heading, header_text) / 100.0
                                )
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = header_text
                                    if best_similarity >= max(
                                        0.5,
                                        1.0
                                        - (
                                            self.max_diffs
                                            / (
                                                len(top_heading)
                                                if len(top_heading) > 0
                                                else 1
                                            )
                                        ),
                                    ):
                                        top_heading_found = True
                                        break

                    # If still no match, use any non-empty cell above as a last resort
                    if not top_heading_found and not best_match and row_idx > 0:
                        for i in range(row_idx):
                            if table_array[i, col_idx].strip():
                                header_text = self.normalise(table_array[i, col_idx])
                                similarity = (
                                    fuzz.ratio(top_heading, header_text) / 100.0
                                )
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = header_text

                    if not best_match:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(
                            f"No top heading found for cell at ({row_idx}, {col_idx})"
                        )
                    elif best_similarity < max(
                        0.5,
                        1.0
                        - (
                            self.max_diffs
                            / (len(top_heading) if len(top_heading) > 0 else 1)
                        ),
                    ):
                        all_relationships_satisfied = False
                        current_failed_reasons.append(
                            f"Top heading '{best_match}' doesn't match expected '{top_heading}' (similarity: {best_similarity:.2f})"
                        )

                # Check left heading relationship
                if left_heading:
                    # Try to find a match in the row headers
                    left_heading_found = False
                    best_match = ""
                    best_similarity = 0

                    # Check the row_headers dictionary first (this handles rowspan properly)
                    if row_idx in table_data.row_headers:
                        for _, header_text in table_data.row_headers[row_idx]:
                            header_text = self.normalise(header_text)
                            similarity = fuzz.ratio(left_heading, header_text) / 100.0
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = header_text
                                if best_similarity >= max(
                                    0.5,
                                    1.0
                                    - (
                                        self.max_diffs
                                        / (
                                            len(left_heading)
                                            if len(left_heading) > 0
                                            else 1
                                        )
                                    ),
                                ):
                                    left_heading_found = True
                                    break

                    # If no match found in row_headers, fall back to checking header columns
                    if not left_heading_found and header_cols:
                        for j in sorted(header_cols):
                            if j < col_idx and table_array[row_idx, j].strip():
                                header_text = self.normalise(table_array[row_idx, j])
                                similarity = (
                                    fuzz.ratio(left_heading, header_text) / 100.0
                                )
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = header_text
                                    if best_similarity >= max(
                                        0.5,
                                        1.0
                                        - (
                                            self.max_diffs
                                            / (
                                                len(left_heading)
                                                if len(left_heading) > 0
                                                else 1
                                            )
                                        ),
                                    ):
                                        left_heading_found = True
                                        break

                    # If still no match, use any non-empty cell to the left as a last resort
                    if not left_heading_found and not best_match and col_idx > 0:
                        for j in range(col_idx):
                            if table_array[row_idx, j].strip():
                                header_text = self.normalise(table_array[row_idx, j])
                                similarity = (
                                    fuzz.ratio(left_heading, header_text) / 100.0
                                )
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = header_text

                    if not best_match:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(
                            f"No left heading found for cell at ({row_idx}, {col_idx})"
                        )
                    elif best_similarity < max(
                        0.5,
                        1.0
                        - (
                            self.max_diffs
                            / (len(left_heading) if len(left_heading) > 0 else 1)
                        ),
                    ):
                        all_relationships_satisfied = False
                        current_failed_reasons.append(
                            f"Left heading '{best_match}' doesn't match expected '{left_heading}' (similarity: {best_similarity:.2f})"
                        )

                # If all relationships are satisfied for this cell, the test passes
                if all_relationships_satisfied:
                    return True, ""
                else:
                    failed_reasons.extend(current_failed_reasons)

        # If we've gone through all tables and all matching cells and none satisfied all relationships
        if not failed_reasons:
            return (
                False,
                f"No cell matching '{cell}' found in any table with threshold {threshold}",
            )
        else:
            return (
                False,
                f"Found cells matching '{cell}' but relationships were not satisfied: {'; '.join(failed_reasons)}",
            )


class BaselineTest(BasePDFTest):
    """
    This test makes sure that several baseline quality checks pass for the output generation.

    Namely, the output is not blank, not endlessly repeating, and contains characters of the proper
    character sets.

    """

    max_repeats: int = 30
    check_disallowed_characters: bool = True
    type: Literal["baseline"] = Field(default="baseline")

    def run(self, content: str) -> Tuple[bool, str, float]:
        if len("".join(c for c in content if c.isalnum()).strip()) == 0:
            return False, "The text contains no alpha numeric characters", 0.0

        # Makes sure that the content has no egregious repeated ngrams at the end, which indicate a degradation of quality
        # Honestly, this test doesn't seem to catch anything at the moment, maybe it can be refactored to a "text-quality"
        # test or something, that measures repetition, non-blanks, charsets, etc
        d = RepeatDetector(max_ngram_size=5)
        d.add_letters(content)
        repeats = d.ngram_repeats()

        for index, count in enumerate(repeats):
            if count > self.max_repeats:
                return (
                    False,
                    f"Text ends with {count} repeating {index+1}-grams, invalid",
                    0.0,
                )

        pattern = re.compile(
            r"["
            r"\u4e00-\u9FFF"  # CJK Unified Ideographs (Chinese characters)
            r"\u3040-\u309F"  # Hiragana (Japanese)
            r"\u30A0-\u30FF"  # Katakana (Japanese)
            r"\U0001F600-\U0001F64F"  # Emoticons (Emoji)
            r"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs (Emoji)
            r"\U0001F680-\U0001F6FF"  # Transport and Map Symbols (Emoji)
            r"\U0001F1E0-\U0001F1FF"  # Regional Indicator Symbols (flags, Emoji)
            r"]",
            flags=re.UNICODE,
        )

        matches = pattern.findall(content)
        if self.check_disallowed_characters and matches:
            return False, f"Text contains disallowed characters {matches}", 0.0

        return True, "", 1.0


def load_tests(jsonl_file: str) -> List[BasePDFTest]:
    """
    Load tests from a JSONL file using parallel processing with a ThreadPoolExecutor.

    Args:
        jsonl_file: Path to the JSONL file containing test definitions.

    Returns:
        A list of test objects.
    """

    def process_line(line_tuple: Tuple[int, str]) -> Optional[Tuple[int, BasePDFTest]]:
        """
        Process a single line from the JSONL file and return a tuple of (line_number, test object).
        Returns None for empty lines.
        """
        line_number, line = line_tuple
        line = line.strip()
        if not line:
            return None

        try:
            data = json.loads(line)
            if "resources" in data:
                data.pop("resources")
            if "tags" in data:
                data.pop("tags")

            _data = {}
            for k, v in data.items():
                if isinstance(v, float) and math.isnan(v) or v is None:
                    continue
                _data[k] = v
            data = _data
            test_type = data.get("type")

            if test_type in {"present", "absent"}:
                test = TextPresenceTest(**data)
            elif test_type == "order":
                test = TextOrderTest(**data)
            elif test_type == "table":
                test = TableTest(**data)
            elif test_type == "baseline":
                test = BaselineTest(**data)
            else:
                raise ValidationError(f"Unknown test type: {test_type}")
            return (line_number, test)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON on line {line_number}: {e}")
            raise
        except (ValidationError, KeyError) as e:
            print(f"Error on line {line_number}: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error on line {line_number}: {e}")
            raise

    tests = []

    # Read all lines along with their line numbers.
    with open(jsonl_file, "r") as f:
        lines = list(enumerate(f, start=1))
        for line in lines:
            tests.append(process_line(line)[1])

    # Check for duplicate test IDs after parallel processing.
    unique_ids = set()
    for test in tests:
        if test.id in unique_ids:
            raise ValidationError(
                f"Test with duplicate id {test.id} found, error loading tests."
            )
        unique_ids.add(test.id)

    return tests


def load_single_test(row: dict) -> Optional[BasePDFTest]:
    """
    Process a single line from the JSONL file and return a tuple of (line_number, test object).
    Returns None for empty lines.
    """

    try:
        _data = {}
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v) or v is None:
                continue
            _data[k] = v
        data = _data
        if "resources" in data:
            data.pop("resources")
        if "tags" in data:
            data.pop("tags")
        test_type = data.get("type")

        if test_type in {"present", "absent"}:
            test = TextPresenceTest(**data)
        elif test_type == "order":
            test = TextOrderTest(**data)
        elif test_type == "table":
            test = TableTest(**data)
        elif test_type == "baseline":
            test = BaselineTest(**data)
        else:
            raise ValidationError(f"Unknown test type: {test_type}")
        return test
    except json.JSONDecodeError as e:
        print(f"Error parsing ds on {row['id']}: {e}")
        raise
    except (ValidationError, KeyError) as e:
        print(f"Error on line {row['id']}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error on {row['id']}: {e}")
        raise


def load_tests_from_ds(ds) -> List[BasePDFTest]:
    """
    Load tests from a JSONL file using parallel processing with a ThreadPoolExecutor.

    Args:
        jsonl_file: Path to the JSONL file containing test definitions.

    Returns:
        A list of test objects.
    """

    tests = []

    # Read all lines along with their line numbers.
    for row in ds.to_dict(orient="records"):
        tests.append(load_single_test(row))

        # _data = {}
        # for k, v in row.items():
        #     if isinstance(v, float) and math.isnan(v) or v is None or k in ["pdf_path"]:
        #         continue
        #     _data[k] = v
        # data = _data
        # for k in ["max_diffs", "first_n", "last_n", "page"]:
        #     if k in data:
        #         data[k] = int(data[k])
        # tests.append(load_single_test(data))

    # Check for duplicate test IDs after parallel processing.
    unique_ids = set()
    for test in tests:
        if test.id in unique_ids:
            raise ValidationError(
                f"Test with duplicate id {test.id} found, error loading tests."
            )
        unique_ids.add(test.id)

    return tests


def save_tests(tests: List[BasePDFTest], jsonl_file: str) -> None:
    """
    Save tests to a JSONL file using model_dump for conversion.

    Args:
        tests: A list of test objects.
        jsonl_file: Path to the output JSONL file.
    """
    with open(jsonl_file, "w") as file:
        for test in tests:
            file.write(json.dumps(test.model_dump()) + "\n")
