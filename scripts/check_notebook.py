"""Validate the outline schema and clean-output policy; does not execute training."""

from pathlib import Path

import nbformat


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "notebooks/iql_halfcheetah.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert not cell.outputs, "Clear notebook outputs before committing"
            assert cell.execution_count is None
    print("Notebook outline: schema valid; outputs empty; not a final Run all check")


if __name__ == "__main__":
    main()
