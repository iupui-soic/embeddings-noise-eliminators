#!/usr/bin/env python3
"""
build_notebooks.py
==================

Convert v4 experiment .py files (jupytext percent format) to .ipynb
notebooks.  Uses only the standard library + nbformat (pip install nbformat).

Usage:
    python build_notebooks.py              # builds all notebooks/*.ipynb
    python build_notebooks.py 04 05        # builds only exp 04 and 05
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell


PCT_SPLIT = re.compile(r"^# %%(?:\s*\[(\w+)\])?.*$", re.MULTILINE)


def parse_percent_file(path: Path):
    text = path.read_text(encoding="utf-8")
    # Strip the jupytext YAML header if present
    text = re.sub(r"^#\s*---\n(.*?)^#\s*---\n", "", text,
                  count=1, flags=re.DOTALL | re.MULTILINE)

    cells = []
    lines = text.splitlines(keepends=True)
    i = 0
    # Walk line-by-line finding % boundaries
    current_type = "code"
    current_buf = []
    for line in lines:
        m = re.match(r"^# %%(?:\s*\[(\w+)\])?.*$", line)
        if m:
            if current_buf:
                cells.append((current_type, "".join(current_buf).rstrip() + "\n"))
            current_type = "markdown" if (m.group(1) == "markdown") else "code"
            current_buf = []
        else:
            current_buf.append(line)
    if current_buf:
        cells.append((current_type, "".join(current_buf).rstrip() + "\n"))

    # Strip the leading "# " from markdown cells
    clean = []
    for t, src in cells:
        if t == "markdown":
            src = re.sub(r"(?m)^#\s?", "", src)
        clean.append((t, src))
    return clean


def py_to_ipynb(py_path: Path, ipynb_path: Path):
    cells = parse_percent_file(py_path)
    nb = new_notebook()
    nb_cells = []
    for t, src in cells:
        if not src.strip():
            continue
        if t == "markdown":
            nb_cells.append(new_markdown_cell(src.rstrip()))
        else:
            nb_cells.append(new_code_cell(src.rstrip()))
    nb.cells = nb_cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    ipynb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ipynb_path, "w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)


def main():
    root = Path(__file__).resolve().parent
    exp_dir = root / "experiments"
    out_dir = root / "notebooks"
    selected = set(sys.argv[1:])
    for py_path in sorted(exp_dir.glob("*.py")):
        stem = py_path.stem
        prefix = stem.split("_", 1)[0]
        if selected and prefix not in selected:
            continue
        ipynb_path = out_dir / f"{stem}.ipynb"
        py_to_ipynb(py_path, ipynb_path)
        print(f"  wrote {ipynb_path.relative_to(root)}")


if __name__ == "__main__":
    main()
