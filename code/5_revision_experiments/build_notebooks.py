"""
build_notebooks.py  —  turn the percent-cell experiment .py files into
SELF-CONTAINED Colab .ipynb notebooks.

Each generated notebook:
  1. mounts Google Drive,
  2. writes the four shared modules (config_paths, models, metrics, ieee_style)
     to /content via %%writefile  -> NO separate `shared/` upload is needed,
  3. then runs the experiment cells.

So you upload ONE .ipynb to Colab and press Run. Edit paths only in the
config_paths writefile cell at the top.

Run locally:  python build_notebooks.py
Outputs to:   Colab_Experiments/notebooks/<exp>.ipynb
"""

import os
import json
import glob

HERE     = os.path.dirname(os.path.abspath(__file__))
SHARED   = os.path.join(HERE, "shared")
OUT_DIR  = os.path.join(HERE, "notebooks")
SHARED_MODULES = ["config_paths.py", "models.py", "metrics.py", "ieee_style.py"]


# ---------------------------------------------------------------------------
# percent-format -> list of (cell_type, source_str)
# ---------------------------------------------------------------------------
def parse_percent(text):
    cells, buf, ctype, started = [], [], "code", False
    for ln in text.splitlines():
        if ln.lstrip().startswith("# %%"):
            if started:
                cells.append((ctype, "\n".join(buf).strip("\n")))
            buf, started = [], True
            ctype = "markdown" if "[markdown]" in ln else "code"
            continue
        buf.append(ln)
    if started:
        cells.append((ctype, "\n".join(buf).strip("\n")))
    elif buf:                                   # no markers: one code cell
        cells.append(("code", "\n".join(buf).strip("\n")))
    # markdown cells: strip the leading "# " comment prefix
    out = []
    for ctype, src in cells:
        if not src.strip():
            continue
        if ctype == "markdown":
            src = "\n".join(l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l)
                            for l in src.splitlines())
        out.append((ctype, src))
    return out


def _src_lines(s):
    """nbformat wants a list of lines, each keeping its trailing newline."""
    lines = s.splitlines()
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _src_lines(src)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src_lines(src)}


# ---------------------------------------------------------------------------
# setup cells (prepended to every experiment)
# ---------------------------------------------------------------------------
def setup_cells(exp_title):
    cells = [
        md(f"# {exp_title}\n\n"
           "**Self-contained Colab notebook.** Run the setup cells once (top to "
           "bottom), then the experiment. The four shared modules are written to "
           "`/content` automatically, so you do NOT need to upload a `shared/` "
           "folder. Edit Drive paths only in the `config_paths.py` cell below.\n\n"
           "> Runtime: set **Runtime -> Change runtime type -> GPU** for the "
           "inference and training experiments."),
        md("## Setup"),
        code("from google.colab import drive\n"
             "drive.mount('/content/drive')\n"
             "import sys\n"
             "if '/content' not in sys.path:\n"
             "    sys.path.insert(0, '/content')\n"
             "print('Drive mounted; /content on path.')"),
        code("# scikit-image backs metrics.calc_ssim; guarantee it for every notebook\n"
             "import subprocess, sys\n"
             "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'scikit-image'])\n"
             "print('scikit-image ready')"),
    ]
    for mod in SHARED_MODULES:
        with open(os.path.join(SHARED, mod), encoding="utf-8") as f:
            body = f.read().rstrip("\n")
        cells.append(code(f"%%writefile /content/{mod}\n{body}"))
    cells.append(code(
        "import importlib\n"
        "import config_paths, models, metrics, ieee_style\n"
        "for m in (config_paths, models, metrics, ieee_style):\n"
        "    importlib.reload(m)\n"
        "ieee_style.set_ieee_style()\n"
        "print('shared modules ready | device =', config_paths.DEVICE)"))
    cells.append(md("## Experiment"))
    return cells


def _is_header_only(src):
    """True only if EVERY meaningful (non-blank, non-comment) line is Drive /
    sys.path / shared-module-import boilerplate. A cell that also does real work
    (third-party imports, aliases, constants) is NOT header-only and is kept."""
    for ln in src.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if ("drive.mount" in s or "sys.path" in s or s.startswith("import sys")
                or s.startswith("from google.colab") or s.startswith("import importlib")
                or "importlib.reload" in s or "set_ieee_style" in s):
            continue
        if s.startswith("import "):                        # bare shared-module imports only
            mods = [t.strip().split(" as ")[0] for t in s[len("import "):].split(",")]
            if all(m in ("config_paths", "models", "metrics", "ieee_style",
                          "sys", "importlib") for m in mods):
                continue
        return False                                        # a real-work line -> keep the cell
    return True


def strip_agent_header(cells):
    """Drop a leading code cell ONLY when it is pure Drive-mount / sys.path /
    shared-import boilerplate (our setup already did all of that). Cells that mix
    boilerplate with real imports/aliases/constants are kept whole; a redundant
    drive.mount inside a kept cell is harmless (Colab's mount is idempotent)."""
    out, dropping = [], True
    for ctype, src in cells:
        if dropping and ctype == "code":
            if _is_header_only(src):
                continue                                    # pure boilerplate -> drop
            dropping = False                                # real work -> stop dropping
        out.append((ctype, src))
    return out


def build_one(py_path):
    name = os.path.splitext(os.path.basename(py_path))[0]
    title = name.replace("_", " ")
    with open(py_path, encoding="utf-8") as f:
        cells = parse_percent(f.read())
    cells = strip_agent_header(cells)
    nb_cells = setup_cells(title)
    for ctype, src in cells:
        nb_cells.append(md(src) if ctype == "markdown" else code(src))
    nb = {
        "cells": nb_cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name + ".ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return out


def main():
    pys = sorted(glob.glob(os.path.join(HERE, "exp*.py")))
    if not pys:
        print("No exp*.py files found yet.")
        return
    for p in pys:
        print("built", build_one(p))


if __name__ == "__main__":
    main()
