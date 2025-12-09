import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def datadir():
    return Path(__file__).parent / "data"


os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"


@pytest.fixture(scope="session", autouse=True)
def file_path():
    return Path(__file__).parent / "data" / "Fiche_Graines_A5.pdf"
