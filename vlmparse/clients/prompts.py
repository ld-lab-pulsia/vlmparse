PDF2MD_PROMPT = r"""Please follow these instructions for the conversion:
1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring. Strictly adhere to the text of the PDF image without reformulating.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.
- Convert Tables of Content (TOC) as numbered lists.
- For footnotes references with uppercase letters, use the following syntax: [^1]

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with $. For example: This is an inline formula $ E = mc^2 $
- Enclose block formulas with $$ $$. For example: $$ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} $$

3. Table Processing:
- Tables should be formatted as HTML with <table> tags.

4. Figure Handling:
- For any images encountered, insert an HTML <img> tag:
  - Use "[image-placeholder]" as the source.
  - Include an alt attribute with the executive summary of the image content.
  - Be very descriptive only if the image is a graph or chart (translate the image as a table, a mermaid diagram or any other format suitable).

5. Lists:
- Use * for unordered lists.

6. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.
- Ignore headers or footers.
- Do not surround your output with triple backticks.
- If there is nothing on the image, just return ONLY the tag <blank> without anything else.
- Convert checked and unchecked boxes to [x] and [ ] respectively.
- For form fields with individual character boxes, transcribe the text as a continuous string instead of splitting characters into separate table cells or using '|' as a separator.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments."""


PDF2HTML_PROMPT = r"""You are an AI assistant specialized in converting PDF images to HTML format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring. Strictly adhere to the text of the PDF image without reformulating.
- Convert the recognized text into HTML format.
- For footnotes references with uppercase letters, use the <sup> tag. For example: <sup>1</sup>
- Do not add a headers or footers .
- Return only the part of the html page between <body> and </body> tags, the html string returned should therefore be: "<body>html content</body>".
- Pay attention to the titles, sections and sub-sections, they should be with <h1>, <h2>, <h3> etc...

2. Figure Handling:
- For any images encountered, insert an HTML <img> tag:
  - Use "[image-placeholder]" as the source.
  - Include an alt attribute with the executive summary of the image content.
  - Be very descriptive only if the image is a graph or chart (translate the image as a table, a mermaid diagram or any other format suitable).

3. Lists:
- Use `<ul>` and `<li>` for unordered lists, and `<ol>` and `<li>` for ordered lists.

5. Output Format:
- Ensure the output HTML document has a simple linear structure without complex hierarchy such as <div> tags.
- Do not translate formatting such as color, bold, italic, etc.
- Do not surround your output with triple backticks.
- If there is nothing on the image, just return ONLY the tag <blank> without anything else.
- Ignore headers or footers, do not add them to the transcription.
- Convert checked and unchecked boxes to [x] and [ ] respectively.
- For form fields with individual character boxes, transcribe the text as a continuous string instead of splitting characters into separate table cells or using '|' as a separator.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into HTML format without adding any extra explanations or comments."""
