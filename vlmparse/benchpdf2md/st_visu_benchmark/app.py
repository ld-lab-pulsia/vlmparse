import argparse
import subprocess
import sys
from glob import glob
from pathlib import Path

import pandas as pd
import streamlit as st
from huggingface_hub import snapshot_download
from pypdfium2.internal.bases import uuid
from streamlit import runtime

from vlmparse.benchpdf2md.bench_tests.benchmark_tsts import load_tests, save_tests
from vlmparse.benchpdf2md.st_visu_benchmark.highligh_text import highlight_text
from vlmparse.benchpdf2md.st_visu_benchmark.test_form import edit_test_form
from vlmparse.benchpdf2md.st_visu_benchmark.ui_elements import download_pdf_page
from vlmparse.benchpdf2md.st_visu_benchmark.utils import get_doc, save_new_test


@st.cache_data
def load_df(results_file):
    return pd.read_parquet(results_file).set_index("test_id")


@st.cache_data
def get_pdf_map(folder: Path) -> dict[str, Path]:
    return {path.name: path for path in Path(folder).rglob("*.pdf")}


@st.cache_data
def get_doc_zip_map(folder: Path) -> dict[str, Path]:
    return {path.name: path for path in Path(folder).rglob("*.zip")}


def run_streamlit(folder: str, dataset_path="pulseia/fr-bench-pdf2md") -> None:
    st.set_page_config(layout="wide")
    # tests_folder = Path(folder) / "tests"
    preds_folder = Path(folder)

    # tests = glob(str(tests_folder / "**/**/tests.jsonl"))
    files = glob(str(preds_folder / "**/**/test_results/**/test_results.parquet"))

    # map_tests = {Path(t).parent.name: t for t in tests}
    if dataset_path == "pulseia/fr-bench-pdf2md":
        local_folder_path = snapshot_download(
            repo_id="pulseia/fr-bench-pdf2md",
            repo_type="dataset",
        )
        dataset_path = local_folder_path

    tests = glob(str(Path(dataset_path) / "**/*.jsonl"), recursive=True)

    map_tests = {Path(t).parent.name: t for t in tests}
    with st.sidebar:
        sel_folders = [
            (
                Path(f).parent.parent.parent.parent.name,
                Path(f).parent.parent.parent.name,
                Path(f).parent.name,
            )
            for f in files
        ]

        if len(sel_folders) == 0:
            st.error(f"No results found in folder {preds_folder}")
            return
        pipe_folder, date1, date2 = st.selectbox("Dir", sel_folders, index=0)
        res_folder = preds_folder / pipe_folder / date1 / "test_results" / date2
        df = load_df(res_folder / "test_results.parquet")

        test_type = st.selectbox("Test type", ["present", "absent", "order", "table"])
        if "category" not in df.columns:
            df["category"] = None
        df["category"] = df["category"].map(str)
        test_category = st.selectbox("Test category", df.category.map(str).unique())

        only_failed = st.checkbox("Only failed", value=False)
        only_not_checked = st.checkbox("Only not checked", value=False)

        display_image = st.checkbox("Display image", value=False)

        preds_folder = preds_folder / pipe_folder / date1 / "results"

        df_sel = df.loc[(df.type == test_type) & (df.category == test_category)]
        if only_failed:
            df_sel = df_sel[~df_sel.result]
        if only_not_checked:
            df_sel = df_sel[df_sel.checked != True]  # noqa: E712

        if df_sel.shape[0] == 0:
            st.markdown("No failed tests found")
            st.stop()
        idx = st.number_input(
            f"Test index (out of {df_sel.shape[0]})",
            value=0,
            min_value=0,
            max_value=df_sel.shape[0] - 1,
            step=1,
        )

        row = df_sel.iloc[idx]

        display_markdown = st.checkbox("Display markdown", value=True)
        show_layout = st.checkbox("Show layout", value=False)
        display_original_text = st.checkbox("Display original text", value=False)
        pdf_map = get_pdf_map(Path(dataset_path))

        pdf_path = pdf_map[row.pdf_path.split("/")[-1]]

        download_pdf_page(pdf_path, page_no=0, file_name=f"{row.tests_name}.pdf")

    doc_path = get_doc_zip_map(preds_folder)[row.doc_path.split("/")[-1]]
    doc = get_doc(doc_path)

    col1_head, col2_head = st.columns(2)
    with col1_head:
        pos_buttons = st.container()
        st.markdown(f"Test: {row.id}" + ("✅" if row.checked else ""))
        st.markdown("Success: " + str(row.result))
        st.markdown("Reason: " + row.explanation)

    tests_path = map_tests[row.tests_name]

    if (
        "tests" not in st.session_state
        or st.session_state.get("current_tests_path") != tests_path
    ):
        st.session_state["tests"] = load_tests(tests_path)
        st.session_state["current_tests_path"] = tests_path

    if "current_tests_path" not in st.session_state:
        st.session_state["current_tests_path"] = tests_path

    if display_original_text:
        res = doc.pages[0].text
    else:

        @st.cache_data
        def get_doc_page_md(doc_path):
            return doc.pages[0].text

        res = get_doc_page_md(row.doc_path)

    with col2_head:
        _tests = [test for test in st.session_state["tests"] if test.id == row.id]

        if len(_tests) < 1:
            st.error("No test found")
            st.stop()
        elif len(_tests) > 1:
            st.error("Multiple tests found")
        test_obj = _tests[0]

        if st.button("Run test"):
            success, message, best_match_score = test_obj.run(res)
            st.markdown(f"Success: {success}, score: {best_match_score:.3f}")
            st.markdown(message)

        add_presence_test = st.checkbox("Add presence test")
        if add_presence_test:
            from vlmparse.benchpdf2md.bench_tests.benchmark_tsts import TextPresenceTest

            test_obj_edited = edit_test_form(
                TextPresenceTest(
                    pdf=row.pdf_path,
                    page=0,
                    id=f"presence_test_{uuid.uuid4()}",
                    type="present",
                    text="",
                ),
                "present",
            )
            if test_obj_edited is not None:
                st.session_state["tests"].append(test_obj_edited)
        else:
            test_obj_edited = edit_test_form(
                test_obj,
                test_type,
            )

        if test_obj_edited is not None:
            save_new_test(
                st.session_state["tests"],
                test_obj_edited,
                st.session_state["current_tests_path"],
            )

    col1_button, col2_button, col3_button = pos_buttons.columns(3)

    with col1_button:
        if st.button("✅ Validate"):
            test_obj.checked = True
            save_new_test(
                st.session_state["tests"],
                test_obj,
                st.session_state["current_tests_path"],
            )

    with col2_button:
        if test_type != "baseline":
            if st.button("❌ Reject"):
                st.session_state["tests"] = [
                    test for test in st.session_state["tests"] if test.id != row.id
                ]
                save_tests(
                    st.session_state["tests"], st.session_state["current_tests_path"]
                )
    with col3_button:
        if st.button("Supress page (Warning, this is irreversible)"):
            import shutil

            shutil.rmtree(Path(row.pdf_path).parent)

    def show_text(res):
        if test_obj:
            res = highlight_text(test_obj, res)

        with st.container(height=700):
            if display_markdown:
                st.markdown(res, unsafe_allow_html=True)
            else:
                st.text(res)

    if display_image:
        with col1_head:
            show_text(res)

        with col2_head:

            @st.cache_data
            def get_image(pipe_folder, date, test_id, show_layout):
                return doc.pages[0].image

            st.image(get_image(pipe_folder, date1, row.id, show_layout))
    else:
        show_text(res)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document viewer with Streamlit")
    parser.add_argument(
        "folder", type=str, nargs="?", default=".", help="Root folder path"
    )
    parser.add_argument(
        "--ds", type=str, default="pulseia/fr-bench-pdf2md", help="Dataset path"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    folder = args.folder

    if runtime.exists():
        run_streamlit(folder, dataset_path=args.ds)
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
