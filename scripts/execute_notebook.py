"""Run the notebook once on a complete package (data/, results/, traces/ present) and save it WITH outputs.

Usage (from anywhere):  python scripts/execute_notebook.py  PACKAGE_DIR  [OUT.ipynb]
  PACKAGE_DIR : the unzipped bigdata_project folder (or the repo with results/ and traces/ copied in)
  OUT.ipynb   : default PACKAGE_DIR/notebook/PaperSearchQA_ReAct_Agent.ipynb (overwritten with the executed copy)
The %pip install cell is skipped during execution and restored in the saved file. Needs: nbformat, nbclient, ipykernel, pandas,
numpy, scipy, matplotlib, rouge_score, langgraph. Exits non-zero if a cell fails or prints "(not found: ...)".
"""
import sys, pathlib, nbformat
from nbclient import NotebookClient

pkg = pathlib.Path(sys.argv[1]).resolve()
nb_path = pkg / "notebook" / "PaperSearchQA_ReAct_Agent.ipynb"
out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else nb_path
nb = nbformat.read(nb_path, 4)
saved = {}
for i, c in enumerate(nb.cells):
    if c.cell_type == "code" and "%pip install" in c.source:
        saved[i] = c.source
        c.source = "# (pip install skipped while executing locally)"
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(nb_path.parent)}}).execute()
for i, src in saved.items():
    nb.cells[i].source = src
bad = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and any(
    "(not found:" in o.get("text", "") or "no trace file" in o.get("text", "") for o in c.outputs)]
nbformat.write(nb, out)
n_code = sum(c.cell_type == "code" for c in nb.cells)
print(f"executed {n_code} code cells -> {out}; cells with missing files: {bad or 'none'}")
sys.exit(1 if bad else 0)
