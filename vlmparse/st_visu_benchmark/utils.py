import io
import os
from pathlib import Path

import pypdfium2 as pdfium
import streamlit as st
from joblib import Parallel, delayed
from tqdm import tqdm

from docparser import dm
from benchdocparser.bench_tests.benchmark_tsts import save_tests


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
    return dm.Document.from_json(doc_path)


@st.cache_data
def get_pred_file_names(preds_folder: Path):
    print("Getting pred file names")
    preds_folder = Path(preds_folder)
    all_res = [f for f in os.listdir(preds_folder) if f.endswith(".json.zip")]

    def get_doc_path(f):
        doc = dm.Document.from_json(preds_folder / f)
        return doc.input_document.file_path

    pdf_paths = Parallel(n_jobs=-1)(delayed(get_doc_path)(f) for f in tqdm(all_res))

    mapping = {}

    for json_path, pdf_path in zip(all_res, pdf_paths, strict=False):
        mapping[str(pdf_path)] = json_path

    return mapping


def save_new_test(tests, test_obj_edited, test_path):
    for test in tests:
        if test.id == test_obj_edited.id:
            test = test_obj_edited
        else:
            test = test
    save_tests(tests, test_path)
    st.success("Test updated successfully!")
