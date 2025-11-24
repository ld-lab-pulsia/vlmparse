from pathlib import Path
from typing import Optional

import streamlit as st

from vlmparse.benchpdf2md.st_visu_benchmark.utils import get_pdf_bytes


def download_pdf_page(
    pdf_path: Path, page_no: int = 0, file_name: Optional[str] = None
):
    pdf_bytes = get_pdf_bytes(pdf_path, page_no)
    if pdf_bytes:
        st.download_button(
            label="📄 Download PDF Page",
            data=pdf_bytes,
            file_name=file_name if file_name else f"{pdf_path.stem}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
