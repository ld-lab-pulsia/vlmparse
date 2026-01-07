import pytest
from pydantic import ValidationError

from vlmparse.benchpdf2md.bench_tests.benchmark_tsts import (
    TableTest,
    parse_html_tables,
    parse_markdown_tables,
)


@pytest.fixture
def markdown_table():
    return """
| Header 1 | Header 2 | Header 3 |
| -------- | -------- | -------- |
| Cell A1  | Cell A2  | Cell A3  |
| Cell B1  | Cell B2  | Cell B3  |
"""


@pytest.fixture
def html_table():
    return """
<table>
  <tr>
    <th>Header 1</th>
    <th>Header 2</th>
    <th>Header 3</th>
  </tr>
  <tr>
    <td>Cell A1</td>
    <td>Cell A2</td>
    <td>Cell A3</td>
  </tr>
  <tr>
    <td>Cell B1</td>
    <td>Cell B2</td>
    <td>Cell B3</td>
  </tr>
</table>
"""


def test_valid_initialization():
    test = TableTest(
        pdf="test.pdf", page=1, id="test_id", type="table", cell="target cell"
    )
    assert test.cell == "target cell"
    assert test.up == ""
    assert test.down == ""
    assert test.left == ""
    assert test.right == ""
    assert test.top_heading == ""
    assert test.left_heading == ""


def test_invalid_test_type():
    with pytest.raises(ValidationError):
        TableTest(
            pdf="test.pdf", page=1, id="test_id", type="present", cell="target cell"
        )


def test_parse_markdown_tables(markdown_table):
    _test = TableTest(
        pdf="test.pdf", page=1, id="test_id", type="table", cell="Cell A2"
    )
    tables = parse_markdown_tables(markdown_table)
    assert len(tables) == 1
    assert tables[0].data.shape == (3, 3)
    assert tables[0].data[0, 0] == "Header 1"
    assert tables[0].data[1, 1] == "Cell A2"
    assert tables[0].data[2, 2] == "Cell B3"


def test_parse_html_tables(html_table):
    _test = TableTest(
        pdf="test.pdf", page=1, id="test_id", type="table", cell="Cell A2"
    )
    tables = parse_html_tables(html_table)
    assert len(tables) == 1
    assert tables[0].data.shape == (3, 3)
    assert tables[0].data[0, 0] == "Header 1"
    assert tables[0].data[1, 1] == "Cell A2"
    assert tables[0].data[2, 2] == "Cell B3"


def test_match_cell(markdown_table):
    test = TableTest(pdf="test.pdf", page=1, id="test_id", type="table", cell="Cell A2")
    result, _, _ = test.run(markdown_table)
    assert result


def test_cell_not_found(markdown_table):
    test = TableTest(
        pdf="test.pdf", page=1, id="test_id", type="table", cell="Missing Cell"
    )
    result, explanation, _ = test.run(markdown_table)
    assert not result
    assert "No cell matching" in explanation


def test_up_relationship(markdown_table):
    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell A2",
        up="Header 2",
    )
    result, _, _ = test.run(markdown_table)
    assert result

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell A2",
        up="Wrong Header",
    )
    result, explanation, _ = test.run(markdown_table)
    assert not result


def test_down_relationship(markdown_table):
    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell A2",
        down="Cell B2",
    )
    result, _, _ = test.run(markdown_table)
    assert result

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell A2",
        down="Wrong Cell",
    )
    result, explanation, _ = test.run(markdown_table)
    assert not result


def test_left_relationship(markdown_table):
    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell A2",
        left="Cell A1",
    )
    result, _, _ = test.run(markdown_table)
    assert result

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell A2",
        left="Wrong Cell",
    )
    result, explanation, _ = test.run(markdown_table)
    assert not result


def test_right_relationship(markdown_table):
    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell A2",
        right="Cell A3",
    )
    result, _, _ = test.run(markdown_table)
    assert result

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell A2",
        right="Wrong Cell",
    )
    result, explanation, _ = test.run(markdown_table)
    assert not result


def test_top_heading_relationship(markdown_table):
    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell B2",
        top_heading="Header 2",
    )
    result, _, _ = test.run(markdown_table)
    assert result

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell B2",
        top_heading="Wrong Header",
    )
    result, explanation, _ = test.run(markdown_table)
    assert not result


def test_left_heading_relationship(markdown_table):
    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell A3",
        left_heading="Cell A1",
    )
    result, _, _ = test.run(markdown_table)
    assert result

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell A3",
        left_heading="Wrong Cell",
    )
    result, explanation, _ = test.run(markdown_table)
    assert not result


def test_multiple_relationships(markdown_table):
    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell A2",
        up="Header 2",
        down="Cell B2",
        left="Cell A1",
        right="Cell A3",
    )
    result, _, _ = test.run(markdown_table)
    assert result

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell A2",
        up="Header 2",
        down="Cell B2",
        left="Wrong Cell",
        right="Cell A3",
    )
    result, explanation, _ = test.run(markdown_table)
    assert not result


def test_no_tables_found():
    test = TableTest(pdf="test.pdf", page=1, id="test_id", type="table", cell="Cell A2")
    result, explanation, _ = test.run("This is plain text with no tables")
    assert not result
    assert explanation == "No HTML tables found in the content"


def test_fuzzy_matching(markdown_table):
    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Cell A2",
        max_diffs=1,
    )
    misspelled_table = markdown_table.replace("Cell A2", "Cel A2")
    result, _, _ = test.run(misspelled_table)
    assert result


def test_with_stripped_content(markdown_table):
    test = TableTest(pdf="test.pdf", page=1, id="test_id", type="table", cell="Cell A2")
    stripped_table = markdown_table.strip()
    result, explanation, _ = test.run(stripped_table)
    assert result, f"Table test failed with stripped content: {explanation}"


def test_table_at_end_of_file(markdown_table):
    test = TableTest(pdf="test.pdf", page=1, id="test_id", type="table", cell="Cell A2")
    content_with_table_at_end = "Some text before the table.\n" + markdown_table.strip()
    result, explanation, _ = test.run(content_with_table_at_end)
    assert result, f"Table at end of file not detected: {explanation}"


def test_table_at_end_with_no_trailing_newline(markdown_table):
    test = TableTest(pdf="test.pdf", page=1, id="test_id", type="table", cell="Cell A2")
    content_without_newline = markdown_table.rstrip()
    result, explanation, _ = test.run(content_without_newline)
    assert result, f"Table without trailing newline not detected: {explanation}"


@pytest.mark.skip(reason="We don't support parsing markdown tables with extra spaces")
def test_table_at_end_with_extra_spaces(markdown_table):
    test = TableTest(pdf="test.pdf", page=1, id="test_id", type="table", cell="Cell A2")
    lines = markdown_table.split("\n")
    content_with_extra_spaces = "\n".join([line + "   " for line in lines])
    result, explanation, _ = test.run(content_with_extra_spaces)
    assert result, f"Table with extra spaces not detected: {explanation}"


def test_table_at_end_with_mixed_whitespace(markdown_table):
    test = TableTest(pdf="test.pdf", page=1, id="test_id", type="table", cell="Cell A2")
    content_with_mixed_whitespace = (
        "Some text before the table.\n" + markdown_table.strip() + "  \t  "
    )
    result, explanation, _ = test.run(content_with_mixed_whitespace)
    assert result, f"Table with mixed whitespace not detected: {explanation}"


# def test_malformed_table_at_end():
#     test = TableTest(
#         pdf="test.pdf", page=1, id="test_id", type="table", cell="Cell A2"
#     )
#     malformed_table = """
# Some text before the table.
# | Header 1 | Header 2 | Header 3
# | -------- | -------- | --------
# | Cell A1  | Cell A2  | Cell A3  |
# | Cell B1  | Cell B2  | Cell B3"""
#     result, explanation, _ = test.run(malformed_table)
#     assert result, f"Malformed table at end not detected: {explanation}"


# def test_incomplete_table_at_end():
#     test = TableTest(
#         pdf="test.pdf", page=1, id="test_id", type="table", cell="Cell A2"
#     )
#     incomplete_table = """
# Some text before the table.
# | Header 1 | Header 2 | Header 3 |
# | Cell A1  | Cell A2  | Cell A3  |
# | Cell B1  | Cell B2  | Cell B3  |"""
#     result, explanation, _ = test.run(incomplete_table)
#     assert result, f"Incomplete table at end not detected: {explanation}"


def test_table_with_excessive_blank_lines_at_end(markdown_table):
    test = TableTest(pdf="test.pdf", page=1, id="test_id", type="table", cell="Cell A2")
    table_with_blanks = markdown_table + "\n\n\n\n\n\n\n\n\n\n"
    result, explanation, _ = test.run(table_with_blanks)
    assert result, f"Table with blank lines at end not detected: {explanation}"


def test_table_at_end_after_long_text(markdown_table):
    test = TableTest(pdf="test.pdf", page=1, id="test_id", type="table", cell="Cell A2")
    long_text = "Lorem ipsum dolor sit amet, " * 100
    content_with_long_text = long_text + "\n" + markdown_table.strip()
    result, explanation, _ = test.run(content_with_long_text)
    assert result, f"Table after long text not detected: {explanation}"


def test_valid_table_at_eof_without_newline():
    test = TableTest(pdf="test.pdf", page=1, id="test_id", type="table", cell="Cell A2")
    valid_table_eof = """
| Header 1 | Header 2 | Header 3 |
| -------- | -------- | -------- |
| Cell A1  | Cell A2  | Cell A3  |
| Cell B1  | Cell B2  | Cell B3  |""".strip()
    result, explanation, _ = test.run(valid_table_eof)
    assert result, f"Valid table at EOF without newline not detected: {explanation}"


def test_normalizing():
    table = """| Question - – Satisfaction on scale of 10 | Response | Resident Sample | Business Sample |
|----------------------------------------|----------|----------------|-----------------|
| Planning for and managing residential, commercial and industrial development | Rating of 8, 9 or 10 | 13% | 11% |
| | Average rating | 6.4 | 5.7 |
| | Don't know responses | 11% | 6% |
| Environmental protection, support for green projects (e.g. green grants, building retrofits programs, zero waste) | Rating of 8, 9 or 10 | 35% | 34% |
| | Average rating | 8.0 | 7.5 |
| | Don't know responses | 8% | 6% |
| Providing and maintaining parks and green spaces | Rating of 8, 9 or 10 | 42% | 41% |
| | Average rating | 7.7 | 7.3 |
| | Don't know responses | 1% | 1% |"""
    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="6%",
        top_heading="Business\nSample",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation


def test_mathematical_minus():
    table = """| Response | Chinese experimenter | White experimenter |
|----------|----------------------|--------------------|
|          | Divided attention    | Full attention     | Divided attention | Full attention |
| Nonverbal| −.34 (.22)           | .54* (.17)         | .12 (.27)         | −.20 (.24)     |
| Verbal   | −.25 (.23)           | .36 (.20)          | .12 (.27)         | −.34 (.22)     |
"""
    test = TableTest(
        pdf="test.pdf", page=1, id="test_id", type="table", cell="-.34 (.22)"
    )
    result, explanation, _ = test.run(table)
    assert result, explanation


def test_markdown_marker():
    table = """| CATEGORY     | POINTS EARNED |
|------------------------------|------------------|
| Sustainable Sites            | 9                |
| Water Efficiency             | 3                |
| Energy & Atmosphere          | 12               |
| Materials & Resources        | 6                |
| Indoor Environmental Quality | 11               |
| Innovation & Design Process  | 5                |
| TOTAL                        | 46               |"""
    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="9",
        up="POINTS EARNED",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation


def test_diffs():
    table = """| CATEGORY     | POINTS EARNED |
|------------------------------|------------------|
| Sustainable Sites            | 9                |
| Water Efficiency             | 3                |
| Energy & Atmosphere          | 12               |
| Materials & Resources        | 6                |
| Indoor Environmental Quality | 11               |
| Innovation & Design Process  | 5                |
| TOTAL                        | 46               |"""
    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="9",
        left="Sustl Sie",
        max_diffs=2,
    )
    result, explanation, _ = test.run(table)
    assert not result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="9",
        left="Sustainable Site",
        max_diffs=2,
    )
    result, explanation, _ = test.run(table)
    assert result, explanation


def test_markdown_marker2():
    table = """| Concentration
level | [CO]      | [SO2] | [NOx]    |
|------------------------|-----------|-------|----------|
| Control                | 0 μM      | 0 μM  | 0 nM     |
| Low                    | 250
μM | 8 μM  | 0.002 nM |
| Medium                 | 625 μM    | 20 μM | 0.005 nM |
| High                   | 1250 μM   | 40 μM | 0.01 nM  |"""
    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="20 μM",
        up=".002 nM",
    )
    result, explanation, _ = test.run(table)
    assert not result, explanation


def test_marker3():
    table = """|                                               | N     | Minimum | Maximum | Gemiddelde | Sd  |
|-----------------------------------------------|-------|---------|---------|------------|-----|
| Slaapkwaliteit tijdens
gewone nachten      | 2017  | 1,0     | 6,0     | 3,9        | 1,0 |
| Slaapkwaliteit tijdens
consignatiediensten | 19816 | 1,0     | 6,0     | 2,8        | 1,2 |
"""
    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="2,8",
        left_heading="Slaapkwaliteit tijdens\nconsignatiediensten",
    )
    result, explanation, _ = test.run(table)
    assert not result, explanation


def test_big_table():
    table = """    <table>
        <caption>Base: Resident respondents (n=1,315) and Business respondents (n=397)</caption>
        <thead>
            <tr>
                <th>Question – Satisfaction on scale of 10</th>
                <th>Response</th>
                <th>Resident Sample</th>
                <th>Business Sample</th>
            </tr>
        </thead>
        <tbody>
            <tr class="category-row">
                <td rowspan="3">Planning for and managing residential, commercial and industrial development</td>
                <td>Rating of 8, 9 or 10</td>
                <td>13%</td>
                <td>11%</td>
            </tr>
            <tr>
                <td class="subcategory">Average rating</td>
                <td>6.4</td>
                <td>5.7</td>
            </tr>
            <tr>
                <td class="subcategory">Don't know responses</td>
                <td>11%</td>
                <td>6%</td>
            </tr>
            
            <tr class="category-row">
                <td rowspan="3">Environmental protection, support for green projects (e.g. green grants, building retrofits programs, zero waste)</td>
                <td>Rating of 8, 9 or 10</td>
                <td>35%</td>
                <td>34%</td>
            </tr>
            <tr>
                <td class="subcategory">Average rating</td>
                <td>8.0</td>
                <td>7.5</td>
            </tr>
            <tr>
                <td class="subcategory">Don't know responses</td>
                <td>8%</td>
                <td>6%</td>
            </tr>
            
            <tr class="category-row">
                <td rowspan="3">Providing and maintaining parks and green spaces</td>
                <td>Rating of 8, 9 or 10</td>
                <td>42%</td>
                <td>41%</td>
            </tr>
            <tr>
                <td class="subcategory">Average rating</td>
                <td>7.7</td>
                <td>7.3</td>
            </tr>
            <tr>
                <td class="subcategory">Don't know responses</td>
                <td>1%</td>
                <td>1%</td>
            </tr>
        </tbody>
    </table>
"""
    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        max_diffs=5,
        cell="Planning for and managing residential, commercial and industrial development",
        down="Environmental protection,\nsupport for green projects\n(e.g. green grants,\nbuilding retrofits programs,\nzero waste)",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation


def test_html_rowspans_colspans():
    table = """    <table>
        <thead>
            <tr>
                <th rowspan="2">Product Category</th>
                <th rowspan="2">Product Subcategory</th>
                <th colspan="4">Quarterly Sales ($000s)</th>
                <th rowspan="2">Annual Total</th>
            </tr>
            <tr>
                <th>Q1</th>
                <th>Q2</th>
                <th>Q3</th>
                <th>Q4</th>
            </tr>
        </thead>
        <tbody>
            <tr class="category">
                <td rowspan="4">Electronics</td>
                <td>Smartphones</td>
                <td>245</td>
                <td>278</td>
                <td>312</td>
                <td>389</td>
                <td>1,224</td>
            </tr>
            <tr class="subcategory">
                <td>Laptops</td>
                <td>187</td>
                <td>192</td>
                <td>243</td>
                <td>297</td>
                <td>919</td>
            </tr>
            <tr class="subcategory">
                <td>Tablets</td>
                <td>95</td>
                <td>123</td>
                <td>135</td>
                <td>156</td>
                <td>509</td>
            </tr>
            <tr class="subcategory">
                <td>Accessories</td>
                <td>64</td>
                <td>72</td>
                <td>87</td>
                <td>105</td>
                <td>328</td>
            </tr>
            <tr class="category">
                <td rowspan="3">Home Appliances</td>
                <td>Refrigerators</td>
                <td>132</td>
                <td>145</td>
                <td>151</td>
                <td>162</td>
                <td>590</td>
            </tr>
            <tr class="subcategory">
                <td>Washing Machines</td>
                <td>98</td>
                <td>112</td>
                <td>127</td>
                <td>143</td>
                <td>480</td>
            </tr>
            <tr class="subcategory">
                <td>Microwaves</td>
                <td>54</td>
                <td>67</td>
                <td>72</td>
                <td>84</td>
                <td>277</td>
            </tr>
            <tr class="category">
                <td rowspan="3">Furniture</td>
                <td>Sofas</td>
                <td>112</td>
                <td>128</td>
                <td>134</td>
                <td>142</td>
                <td>516</td>
            </tr>
            <tr class="subcategory">
                <td>Tables</td>
                <td>87</td>
                <td>95</td>
                <td>103</td>
                <td>124</td>
                <td>409</td>
            </tr>
            <tr class="subcategory">
                <td>Chairs</td>
                <td>76</td>
                <td>84</td>
                <td>92</td>
                <td>110</td>
                <td>362</td>
            </tr>
            <tr class="total">
                <td colspan="2">Quarterly Totals</td>
                <td>1,150</td>
                <td>1,296</td>
                <td>1,456</td>
                <td>1,712</td>
                <td>5,614</td>
            </tr>
        </tbody>
    </table>"""

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Refrigerators",
        left="Home Appliances",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Washing Machines",
        left="Home Appliances",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Microwaves",
        left="Home Appliances",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Sofas",
        top_heading="Product Subcategory",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="135",
        top_heading="Q3",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="135",
        top_heading="Quarterly Sales ($000s)",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="1,712",
        top_heading="Quarterly Sales ($000s)",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="135",
        top_heading="Q2",
    )
    result, explanation, _ = test.run(table)
    assert not result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="135",
        top_heading="Q1",
    )
    result, explanation, _ = test.run(table)
    assert not result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="135",
        top_heading="Q4",
    )
    result, explanation, _ = test.run(table)
    assert not result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Home Appliances",
        top_heading="Product Category",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Washing Machines",
        top_heading="Product Category",
    )
    result, explanation, _ = test.run(table)
    assert not result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Washing Machines",
        top_heading="Q3",
    )
    result, explanation, _ = test.run(table)
    assert not result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Washing Machines",
        top_heading="Quarterly Sales ($000s)",
    )
    result, explanation, _ = test.run(table)
    assert not result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Electronics",
        right="Laptops",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Electronics",
        right="Accessories",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation


def test_multiple_markdown_tables():
    content = """
# First Table

| Name | Age | Role |
| ---- | --- | ---- |
| John | 28  | Developer |
| Jane | 32  | Designer |
| Bob  | 45  | Manager |

Some text between tables...

# Second Table

| Department | Budget | Employees |
| ---------- | ------ | --------- |
| Engineering | 1.2M  | 15 |
| Design      | 0.8M  | 8  |
| Marketing   | 1.5M  | 12 |
| HR          | 0.5M  | 5  |
"""

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="John",
        right="28",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="32",
        left="Jane",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Engineering",
        right="1.2M",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="12",
        left="1.5M",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Bob",
        top_heading="Name",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="HR",
        top_heading="Department",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation


def test_multiple_html_tables():
    content = """
<h1>First Table</h1>

<table>
  <thead>
    <tr>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>USA</td>
      <td>Washington DC</td>
      <td>331M</td>
    </tr>
    <tr>
      <td>France</td>
      <td>Paris</td>
      <td>67M</td>
    </tr>
    <tr>
      <td>Japan</td>
      <td>Tokyo</td>
      <td>126M</td>
    </tr>
  </tbody>
</table>

<p>Some text between tables...</p>

<h1>Second Table</h1>

<table>
  <thead>
    <tr>
      <th>Company</th>
      <th>Industry</th>
      <th>Revenue</th>
      <th>Employees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ABC Corp</td>
      <td>Technology</td>
      <td>$5B</td>
      <td>10,000</td>
    </tr>
    <tr>
      <td>XYZ Inc</td>
      <td>Healthcare</td>
      <td>$2.5B</td>
      <td>8,500</td>
    </tr>
    <tr>
      <td>Acme Co</td>
      <td>Manufacturing</td>
      <td>$1.8B</td>
      <td>15,000</td>
    </tr>
    <tr>
      <td>Global LLC</td>
      <td>Finance</td>
      <td>$3.2B</td>
      <td>6,200</td>
    </tr>
  </tbody>
</table>
"""

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="USA",
        right="Washington DC",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="126M",
        left="Tokyo",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="XYZ Inc",
        right="Healthcare",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="15,000",
        left="$1.8B",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Tokyo",
        top_heading="Capital",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Finance",
        top_heading="Industry",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation


def test_mixed_markdown_and_html_tables():
    content = """
# Markdown Table

| Product | Price | Quantity |
| ------- | ----- | -------- |
| Apple   | $1.20 | 100      |
| Orange  | $0.80 | 150      |
| Banana  | $0.60 | 200      |

<h1>HTML Table</h1>

<table>
  <tr>
    <th>Month</th>
    <th>Income</th>
    <th>Expenses</th>
    <th>Profit</th>
  </tr>
  <tr>
    <td>January</td>
    <td>$10,000</td>
    <td>$8,000</td>
    <td>$2,000</td>
  </tr>
  <tr>
    <td>February</td>
    <td>$12,000</td>
    <td>$9,500</td>
    <td>$2,500</td>
  </tr>
  <tr>
    <td>March</td>
    <td>$15,000</td>
    <td>$10,200</td>
    <td>$4,800</td>
  </tr>
</table>
"""

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Orange",
        right="$0.80",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="February",
        right="$12,000",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="100",
        top_heading="Quantity",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="$4,800",
        top_heading="Profit",
    )
    result, explanation, _ = test.run(content)
    assert result, explanation


def test_br_tags_replacement():
    table = """<table>
          <tr>
            <th>Header 1</th>
            <th>Header 2</th>
          </tr>
          <tr>
            <td>Line 1<br/>Line 2<br/>Line 3</td>
            <td>Single line</td>
          </tr>
        </table>"""

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="Line 1 Line 2 Line 3",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation


def test_real_complicated_table():
    table = """    <table>
        <thead>
            <tr>
                <th colspan="7">Table 1 &nbsp;&nbsp; Differences in diagnoses, gender and family status for participants with a suicide attempt and those without a suicide attempt within the 12-month follow-up interval</th>
            </tr>
            <tr class="header-row">
                <th rowspan="2"></th>
                <th colspan="2">Participants with no<br>suicide attempt<br>(n = 132)<sup>a</sup></th>
                <th colspan="2">Participants with a<br>suicide attempt<br>(n = 43)<sup>b</sup></th>
                <th colspan="3"></th>
            </tr>
            <tr class="header-row">
                <th>n</th>
                <th>%</th>
                <th>n</th>
                <th>%</th>
                <th>χ<sup>2</sup></th>
                <th>d.f.</th>
                <th>P</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td class="section-header">ICD-10 diagnoses</td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;F0</td>
                <td>1</td>
                <td>0.76</td>
                <td>0</td>
                <td>0.00</td>
                <td>0.00</td>
                <td>1</td>
                <td>1.00</td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;F1</td>
                <td>17</td>
                <td>12.88</td>
                <td>12</td>
                <td>27.91</td>
                <td>4.39</td>
                <td>1</td>
                <td>0.04</td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;F2</td>
                <td>1</td>
                <td>0.76</td>
                <td>0</td>
                <td>0.00</td>
                <td>0.00</td>
                <td>1</td>
                <td>1.00</td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;F3</td>
                <td>106</td>
                <td>80.30</td>
                <td>31</td>
                <td>72.09</td>
                <td>0.74</td>
                <td>1</td>
                <td>0.39</td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;F4</td>
                <td>42</td>
                <td>31.82</td>
                <td>17</td>
                <td>39.53</td>
                <td>0.61</td>
                <td>1</td>
                <td>0.43</td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;F5</td>
                <td>5</td>
                <td>3.79</td>
                <td>5</td>
                <td>11.63</td>
                <td>2.44</td>
                <td>1</td>
                <td>0.12</td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;F6</td>
                <td>20</td>
                <td>15.15</td>
                <td>19</td>
                <td>44.19</td>
                <td>14.48</td>
                <td>1</td>
                <td>0.00</td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;F7</td>
                <td>0</td>
                <td>0.00</td>
                <td>0</td>
                <td>0.00</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;F8</td>
                <td>1</td>
                <td>0.76</td>
                <td>0</td>
                <td>0.00</td>
                <td>0.00</td>
                <td>1</td>
                <td>1.00</td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;F9</td>
                <td>2</td>
                <td>1.52</td>
                <td>1</td>
                <td>2.33</td>
                <td>0.00</td>
                <td>1</td>
                <td>1.00</td>
            </tr>
            <tr>
                <td class="section-header">Gender</td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
                <td>3.09</td>
                <td>2</td>
                <td>0.21</td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;Female</td>
                <td>75</td>
                <td>56.8</td>
                <td>24</td>
                <td>55.8</td>
                <td></td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;Male</td>
                <td>57</td>
                <td>43.2</td>
                <td>18</td>
                <td>41.9</td>
                <td></td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;Diverse</td>
                <td>0</td>
                <td>0</td>
                <td>1</td>
                <td>2.3</td>
                <td></td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td class="section-header">Family status</td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
                <td>4.87</td>
                <td>4</td>
                <td>0.30</td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;Single</td>
                <td>55</td>
                <td>41.7</td>
                <td>14</td>
                <td>32.6</td>
                <td></td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;Partnership</td>
                <td>25</td>
                <td>18.9</td>
                <td>9</td>
                <td>20.9</td>
                <td></td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;Married</td>
                <td>27</td>
                <td>20.5</td>
                <td>5</td>
                <td>11.6</td>
                <td></td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;Divorced</td>
                <td>20</td>
                <td>15.2</td>
                <td>11</td>
                <td>25.6</td>
                <td></td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>&nbsp;&nbsp;Widowed</td>
                <td>1</td>
                <td>0.8</td>
                <td>1</td>
                <td>2.3</td>
                <td></td>
                <td></td>
                <td></td>
            </tr>
        </tbody>
        <tfoot>
            <tr>
                <td colspan="8" class="footnote">
                    F0: Organic, including symptomatic, mental disorders; F1: Mental and behavioural disorders due to psychoactive substance use; F2: Schizophrenia, schizotypal and delusional disorders; F3: affective disorders; F4: Neurotic, stress-related and somatoform disorders; F5: Behavioural syndromes associated with physiological disturbances and physical factors; F6: Disorders of adult personality and behaviour; F7: Mental retardation; F8: Disorders of psychological development; F9: Behavioural and emotional disorders with onset usually occurring in childhood and adolescence.<br>
                    a. 75.43% of the total sample with full information on suicide reattempts within the entire 12-month follow-up interval.<br>
                    b. 24.57% of the total sample with full information on suicide reattempts within the entire 12-month follow-up interval.
                </td>
            </tr>
        </tfoot>
    </table>"""

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="4.39",
        top_heading="χ2",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="12.88",
        top_heading="%",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="12.88",
        top_heading="Participants with no suicide attempt (n = 132)a",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation

    test = TableTest(
        pdf="test.pdf",
        page=1,
        id="test_id",
        type="table",
        cell="12.88",
        top_heading="Table 1    Differences in diagnoses, gender and family status for participants with a suicide attempt and those without a suicide attempt within the 12-month follow-up interval",
    )
    result, explanation, _ = test.run(table)
    assert result, explanation
