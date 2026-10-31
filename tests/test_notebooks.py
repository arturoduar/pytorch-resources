"""Checks for the example notebooks.

* Structure tests (always run): valid JSON, valid nbformat 4, and every code cell parses.
* Execution tests (marker ``notebooks``): run each notebook top to bottom in a fresh kernel.
  They are skipped unless the ``notebooks`` extra is installed. Deselect with
  ``pytest -m "not notebooks"``.
"""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def test_notebooks_exist():
    assert len(NOTEBOOKS) >= 3


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_structure(path):
    nb = json.loads(path.read_text())
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook has no cells"
    assert any(c["cell_type"] == "markdown" for c in nb["cells"]), "needs explanatory text"
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # Drop IPython magics and shell escapes, which are not valid Python.
        lines = [ln for ln in source.splitlines() if not ln.lstrip().startswith(("%", "!"))]
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{path.name} cell {i} has a syntax error: {exc}")


@pytest.mark.notebooks
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_executes(path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    pytest.importorskip("matplotlib")

    # The kernel is a subprocess, so hand it the source tree via the environment.
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    monkeypatch.setenv("MPLBACKEND", "Agg")

    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=300, kernel_name="python3", cwd=str(NOTEBOOK_DIR))
    client.execute()  # raises CellExecutionError on any failing cell or assert
