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

    fixtures = tmp_path / "fixtures"
    build_all(str(fixtures))
    train_output = tmp_path / "outputs" / "train_ods"
    inference_output = tmp_path / "outputs" / "inference_ods"
    for nb in NOTEBOOKS:
        book = nbformat.read(os.path.join(NB_DIR, nb), as_version=4)
        for cell in book.cells:
            if cell.cell_type != "code":
                continue
            cell.source = cell.source.replace('sys.path.insert(0, "..")', f'sys.path.insert(0, {str(ROOT)!r})')
            if nb == "train.ipynb":
                cell.source = cell.source.replace(
                    'cfg = build_config(PRESET, OVERRIDES)',
                    f'OVERRIDES["output_dir"] = {str(train_output)!r}\n'
                    f'OVERRIDES["data"]["path"] = {str(fixtures / "ods.parquet")!r}\n'
                    'cfg = build_config(PRESET, OVERRIDES)',
                )
            else:
                cell.source = cell.source.replace(
                    'icfg = InferenceConfig.from_dict(CONFIG)',
                    f'CONFIG.update(artifact_dir={str(train_output / "artifact")!r}, '
                    f'input_path={str(fixtures / "infer_unknown_labels.parquet")!r}, '
                    f'output_path={str(inference_output / "predictions.parquet")!r}, '
                    f'report_dir={str(inference_output / "reports")!r})\n'
                    'icfg = InferenceConfig.from_dict(CONFIG)',
                )
        ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(book, {"metadata": {"path": str(tmp_path)}})
        nbformat.write(book, tmp_path / nb)
    art = train_output / "artifact"
    assert os.path.exists(os.path.join(art, "model.pt"))
    out = pd.read_parquet(inference_output / "predictions.parquet")
    assert set(output_columns(3)) <= set(out.columns)
