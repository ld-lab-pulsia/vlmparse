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
        "show_items": st.checkbox("Show items", value=False),
        "show_text_cells": st.checkbox("Show native text cells", value=False),
        "overlay_text_cells": st.checkbox("Overlay text cells on image", value=False),
    }


def _crop_image(image, box):
    """Return a PIL crop of *image* for the given BoundingBox."""
    if box.coord_origin == "BOTTOMLEFT":
        box = box.to_top_left_origin(image.size[1])

    if box.relative and box.reference is not None:
        box = box.to_absolute(box.reference)

    l = max(0, int(box.l))
    t = max(0, int(box.t))
    r = min(image.size[0], int(box.r))
    b = min(image.size[1], int(box.b))
    return image.crop((l, t, r, b))


def render_items(page):
    """Render page items one by one.

    - All items: their text is displayed.
    - Items with category='picture': the image crop is displayed alongside the text.
    """
    items = page.items
    image = page.image

    if not items:
        st.info("No items available for this page.")
        return

    for item in items:
        if item.category == "picture":
            col_img, col_txt = st.columns([1, 2])
            with col_img:
                if image is not None:
                    try:
                        st.image(_crop_image(image, item.box))
                    except Exception as exc:
                        st.warning(f"Could not crop image: {exc}")
                else:
                    st.caption("No image available.")
            with col_txt:
                st.markdown(item.text, unsafe_allow_html=True)
        else:
            st.markdown(item.text, unsafe_allow_html=True)


def render_text_cells(page):
    """Render native docling-parse text cells as a table."""
    cells = page.text_cells
    if not cells:
        st.info("No native text cells available for this page.")
        return
    rows = [
        {
            "text": cell.text,
            "l": round(cell.box.l, 1),
            "t": round(cell.box.t, 1),
            "r": round(cell.box.r, 1),
            "b": round(cell.box.b, 1),
        }
        for cell in cells
    ]
    st.dataframe(rows, use_container_width=True)


def get_image_with_text_cells(page):
    """Return the page image with native text-cell bounding boxes drawn in blue."""
    from PIL import ImageDraw

    image = page.image
    if image is None or not page.text_cells:
        return image

    image = image.copy()
    draw = ImageDraw.Draw(image)

    for cell in page.text_cells:
        b = cell.box
        draw.rectangle(
            (int(b.l), int(b.t), int(b.r), int(b.b)),
            outline=(0, 0, 255),
            width=2,
        )
    return image


def run_streamlit(folder: str) -> None:
    with st.sidebar:
        file_path = file_selector(folder)

    if not file_path:
        st.info("Please select a file from the sidebar.")
        return

    doc = get_doc(file_path)

    with st.sidebar:
        settings = render_sidebar_controls(doc, file_path)

    page = doc.pages[settings["page_no"]]

    col1, col2 = st.columns(2)
    with col1:
        with st.container(height=700):
            if settings["show_text_cells"] and page.text_cells:
                render_text_cells(page)
            elif settings["show_items"] and page.items:
                render_items(page)
            else:
                st.markdown(page.text, unsafe_allow_html=True)
    with col2:
        if settings["overlay_text_cells"] and page.text_cells:
            st.image(get_image_with_text_cells(page))
        elif settings["plot_layouts"]:
            st.image(page.get_image_with_boxes(layout=True))
        else:
            st.image(page.image)


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
