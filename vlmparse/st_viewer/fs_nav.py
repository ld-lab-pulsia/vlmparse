from __future__ import annotations

import glob
import os
from typing import List, Optional

import streamlit as st


def get_gz_files_count(folder_path: str) -> int:
    return len(glob.glob(os.path.join(folder_path, "*.json*")))


def get_subdirectories(path: str) -> List[str]:
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def file_selector(root_folder: str) -> Optional[str]:
    st.title("Folder Navigation")
    if not root_folder or not os.path.exists(root_folder):
        return None

    current_path = selected_path = root_folder
    level = 0
    while True:
        subdirs = get_subdirectories(current_path)
        if not subdirs:
            break
        dir_options = [
            f"{d} ({get_gz_files_count(os.path.join(current_path, d))} .zip files)"
            for d in subdirs
        ]
        selected = st.selectbox(
            f"Level {level} Selection",
            ["--Select--"] + dir_options,
            key=f"level_{level}",
        )
        if selected == "--Select--" or not selected:
            break
        selected_dir = selected.split(" (", 1)[0]
        current_path = os.path.join(current_path, selected_dir)
        selected_path = current_path
        level += 1

    gz_files = sorted(glob.glob(os.path.join(selected_path, "*.zip")))
    if gz_files:
        selected_file = st.selectbox(
            "Select .zip file",
            ["--Select--"] + [os.path.basename(f) for f in gz_files],
        )
        if selected_file != "--Select--":
            return os.path.join(selected_path, selected_file)
    return None
