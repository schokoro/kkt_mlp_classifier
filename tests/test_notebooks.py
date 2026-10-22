"""DoD 1, 2, 5: ноутбуки — только фасад и исполняются end-to-end на ODS на CPU."""
import ast
import os

import nbformat
import pandas as pd
import pytest

from kkt_clf.fixtures import build_all
from kkt_clf.inference import output_columns

from .conftest import ROOT

NB_DIR = os.path.join(ROOT, "notebooks")
NOTEBOOKS = ["train.ipynb", "inference.ipynb"]


@pytest.mark.parametrize("nb", NOTEBOOKS)
def test_notebook_is_facade(nb):
    """В ячейках нет определений функций, классов и лямбд."""
    book = nbformat.read(os.path.join(NB_DIR, nb), as_version=4)
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        bad = [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))]
        assert not bad, f"{nb}: определение в ячейке:\n{cell.source[:200]}"


@pytest.mark.slow
def test_notebooks_run_all(tmp_path):
    from nbconvert.preprocessors import ExecutePreprocessor

    fixtures = os.path.join(ROOT, "fixtures")
    if not os.path.exists(os.path.join(fixtures, "ods.parquet")):
        build_all(fixtures)
    for nb in NOTEBOOKS:
        book = nbformat.read(os.path.join(NB_DIR, nb), as_version=4)
        ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(book, {"metadata": {"path": NB_DIR}})
        nbformat.write(book, tmp_path / nb)
    art = os.path.join(ROOT, "outputs", "train_ods", "artifact")
    assert os.path.exists(os.path.join(art, "model.pt"))
    out = pd.read_parquet(os.path.join(ROOT, "outputs", "inference_ods", "predictions.parquet"))
    assert set(output_columns(3)) <= set(out.columns)
