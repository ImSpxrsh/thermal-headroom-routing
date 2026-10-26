"""Build a self-contained arXiv source bundle from paper/.

arXiv compiles with pdflatex only and is picky about file layout, so the bundle has:
  paper.tex   tables inlined (no \\input of generated files), all author emails listed
  paper.bbl   pre-built bibliography (arXiv uses it directly)
  figures/    only the PDFs the paper includes
and is test-compiled in a clean folder exactly the way arXiv does it.

    python tools/make_arxiv.py        # writes arxiv/ and arxiv_submission.zip
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "paper"
OUT = ROOT / "arxiv"

# Author emails live in tools/arxiv_authors.tex (git-ignored, kept out of the public repo).
AUTHORS_FILE = Path(__file__).resolve().parent / "arxiv_authors.tex"


def build_tex() -> str:
    tex = (SRC / "paper.tex").read_text()
    # inline every generated table
    tex = re.sub(r"\\tabinput\{([^}]*)\}", lambda m: (SRC / m.group(1)).read_text().rstrip("\n"), tex)
    tex = tex.replace("% expandable \\input so generated table rows may start with \\multicolumn\n", "")
    tex = tex.replace("\\makeatletter\\newcommand{\\tabinput}[1]{\\@@input #1 }\\makeatother\n", "")
    old = "\\IEEEauthorblockA{\\IEEEauthorrefmark{3}Corresponding author: sparshr@mit.edu}"
    assert old in tex, "author block changed; update make_arxiv.py"
    tex = tex.replace(old, AUTHORS_FILE.read_text().rstrip("\n"))
    assert "\\tabinput" not in tex and "\\input" not in tex
    return tex


def pdflatex(d: Path, runs=2, bib=False):
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"]
    subprocess.run(cmd, cwd=d, capture_output=True)
    if bib:
        subprocess.run(["bibtex", "paper"], cwd=d, capture_output=True)
    for _ in range(runs):
        subprocess.run(cmd, cwd=d, capture_output=True)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)
    tex = build_tex()
    (OUT / "paper.tex").write_text(tex)
    for f in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", tex):
        shutil.copy(SRC / "figures" / f, OUT / "figures" / f)

    # build the .bbl once, with bibtex
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        shutil.copytree(OUT, t, dirs_exist_ok=True)
        shutil.copy(SRC / "references.bib", t / "references.bib")
        pdflatex(t, bib=True)
        shutil.copy(t / "paper.bbl", OUT / "paper.bbl")

    # test exactly like arXiv: only the bundle, pdflatex only, no bibtex
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        shutil.copytree(OUT, t, dirs_exist_ok=True)
        pdflatex(t, runs=3)
        log = (t / "paper.log").read_text(errors="ignore")
        errors = [l for l in log.splitlines() if l.startswith("!")]
        undefined = [l for l in log.splitlines() if "undefined" in l and ("Citation" in l or "Reference" in l)]
        ok = (t / "paper.pdf").exists() and not errors and not undefined
        print("clean-room compile:", "OK" if ok else "FAILED", "| errors:", errors[:3], "| undefined:", undefined[:3])
        if (t / "paper.pdf").exists():
            shutil.copy(t / "paper.pdf", OUT.parent / "arxiv_preview.pdf")

    z = ROOT / "arxiv_submission.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(OUT))
    print("wrote", z, [n for n in zipfile.ZipFile(z).namelist()])


if __name__ == "__main__":
    main()
