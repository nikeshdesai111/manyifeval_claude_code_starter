import pytest

from mife_ifb.config import nltk_data_dir, source_dir


def pytest_collection_modifyitems(config, items):
    if not (source_dir() / "ifbench").is_dir() or not nltk_data_dir().is_dir():
        skip = pytest.mark.skip(reason="pinned IFBench/NLTK data missing: run ifbench/scripts/fetch_source.sh")
        for it in items:
            if "isolation" not in it.nodeid:
                it.add_marker(skip)
