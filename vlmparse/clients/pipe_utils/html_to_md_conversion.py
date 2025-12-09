import re

from bs4 import BeautifulSoup
from bs4.element import NavigableString
from html_to_markdown import convert_to_markdown


def html_to_md_keep_tables(html: str, remove_head: bool = False) -> str:
    # remove the whole <img>…</img> block (tag + contents)
    html = re.sub(r"<img>([\s\S]*?)</img>", "", html, flags=re.S)

    soup = BeautifulSoup(html, "lxml")
    if remove_head:
        # Remove global <head> tags if present
        if soup.head:
            soup.head.decompose()

    # --- recurse inside tables first ---------------------------------------
    for table in soup.find_all("table"):
        for cell in table.find_all(["td", "th"]):
            inner = "".join(map(str, cell.contents))
            cell_md = html_to_md_keep_tables(inner).strip()  # recursion
            cell.clear()
            cell.append(NavigableString(cell_md))

    # --- protect current-level tables --------------------------------------
    markers = {}
    for i, t in enumerate(soup.find_all("table")):
        key = f"%%%TAB{i}%%%"
        markers[key] = str(t)
        t.replace_with(key)

    # --- html → markdown (tables excluded) ---------------------------------
    if len(str(soup)) > 0:
        md_txt = convert_to_markdown(
            str(soup),
            strip=["table", "b", "strong", "i", "em"],
            heading_style="atx",
            escape_misc=False,
            bullets="-",
        )
    else:
        md_txt = ""

    # --- restore tables -----------------------------------------------------
    for k, raw in markers.items():
        md_txt = md_txt.replace(k, raw)
        # print(md_txt)

    return md_txt


# %%
def md_tables_to_html(md_text: str) -> str:
    """
    Convert all Markdown tables in the text to HTML tables.

    Args:
        md_text: Markdown text containing tables

    Returns:
        Text with Markdown tables replaced by HTML tables
    """
    # Pattern to match Markdown tables
    table_pattern = r"(\|[^\n]*\|\n(?:\|[-\s:]*\|\n)?(?:\|[^\n]*\|\n?)*)"

    def convert_md_table_to_html(match):
        table_text = match.group(1).strip()
        lines = table_text.split("\n")

        # Remove empty lines
        lines = [line for line in lines if line.strip()]

        if len(lines) < 2:
            return table_text  # Not a valid table

        # Check if second line is a separator (contains only |, -, :, and spaces)
        separator_pattern = r"^\|[-:\s|]+\|$"
        has_separator = bool(re.match(separator_pattern, lines[1]))

        if not has_separator:
            return table_text  # Not a valid table

        # Parse header
        header_row = lines[0]
        header_cells = [cell.strip() for cell in header_row.split("|")[1:-1]]

        # Parse data rows
        data_rows = []
        for line in lines[2:]:
            if line.strip():
                cells = [cell.strip() for cell in line.split("|")[1:-1]]
                data_rows.append(cells)

        # Build HTML table
        html_parts = ["<table>"]

        # Add header
        if header_cells:
            html_parts.append("<thead>")
            html_parts.append("<tr>")
            for cell in header_cells:
                html_parts.append(f"<th>{cell}</th>")
            html_parts.append("</tr>")
            html_parts.append("</thead>")

        # Add body
        if data_rows:
            html_parts.append("<tbody>")
            for row in data_rows:
                html_parts.append("<tr>")
                for cell in row:
                    html_parts.append(f"<td>{cell}</td>")
                html_parts.append("</tr>")
            html_parts.append("</tbody>")

        html_parts.append("</table>")

        return "".join(html_parts)

    # Replace all Markdown tables with HTML tables
    result = re.sub(
        table_pattern, convert_md_table_to_html, md_text, flags=re.MULTILINE
    )

    return result


if __name__ == "__main__":
    md_text = """| Name | Age | City |
    |------|-----|------|
    | John | 25  | NYC  |
    | Jane | 30  | LA   |"""

    html_result = md_tables_to_html(md_text)
    print(html_result)
