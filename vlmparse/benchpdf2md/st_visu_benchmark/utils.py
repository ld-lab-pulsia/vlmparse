import io
from pathlib import Path

import pypdfium2 as pdfium
import streamlit as st

from vlmparse.data_model.document import Document


@st.cache_data
def get_pdf_bytes(pdf_path, page_no=0):
    pdf_reader = pdfium.PdfDocument(pdf_path)
    if page_no >= len(pdf_reader):
        pdf_reader.close()
        return None

    # Create a new PDF
    new_pdf = pdfium.PdfDocument.new()

    # Import the chosen page into the new PDF
    new_pdf.import_pages(pdf_reader, pages=[page_no])

    bytes_io = io.BytesIO()
    # Get bytes
    new_pdf.save(bytes_io)

    pdf_bytes = bytes_io.getvalue()

    # Clean up
    new_pdf.close()
    pdf_reader.close()

    return pdf_bytes


@st.cache_data
def get_doc(doc_path: Path):
    return Document.from_zip(doc_path)


def save_new_test(tests, test_obj_edited, test_path):
    from vlmparse.benchpdf2md.bench_tests.benchmark_tsts import save_tests

    for test in tests:
        if test.id == test_obj_edited.id:
            test = test_obj_edited
        else:
            test = test
    save_tests(tests, test_path)
    st.success("Test updated successfully!")
