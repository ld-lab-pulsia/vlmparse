import argparse
import subprocess
import sys

import streamlit as st
from streamlit import runtime

from vlmparse.data_model.document import Document
from vlmparse.st_viewer.fs_nav import file_selector

st.set_page_config(layout="wide")


@st.cache_resource
def get_doc(file_path):
    return Document.from_zip(file_path)


def render_sidebar_controls(doc, file_path):
    """Render sidebar controls and return settings."""
    return {
        "page_no": st.number_input("Page", 0, len(doc.pages) - 1, 0),
        "plot_layouts": st.checkbox("Plot layouts", value=True),
    }


def run_streamlit(folder: str) -> None:
    with st.sidebar:
        file_path = file_selector(folder)

    if not file_path:
        st.info("Please select a file from the sidebar.")
        return

    doc = get_doc(file_path)

    with st.sidebar:
        settings = render_sidebar_controls(doc, file_path)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(height=700):
            st.write(doc.pages[settings["page_no"]].text)

    with col2:
        if settings["plot_layouts"]:
            st.image(doc.pages[settings["page_no"]].get_image_with_boxes(layout=True))
        else:
            st.image(doc.pages[settings["page_no"]].image)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document viewer with Streamlit")
    parser.add_argument(
        "folder", type=str, nargs="?", default=".", help="Root folder path"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    folder = parse_args().folder

    if runtime.exists():
        run_streamlit(folder)
    else:
        try:
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", __file__, "--", folder],
                check=True,
            )
        except KeyboardInterrupt:
            print("\nStreamlit app terminated by user.")
        except subprocess.CalledProcessError as e:
            print(f"Error while running Streamlit: {e}")


if __name__ == "__main__":
    main()
