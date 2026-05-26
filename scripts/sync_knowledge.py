
"""
One command: rebuild FAQ CSV → SQLite (bot) → static wiki site.

Source of truth: data/faq_seed.csv
Bot reads SQLite (faster than parsing wiki at runtime).
Wiki is a human-readable mirror of the same cards.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"


def run(script: str) -> None:
    cmd = [str(PY), str(ROOT / "scripts" / script)]
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    run("build_faq_extended.py")
    run("clean_faq_seed.py")
    run("init_all.py")
    run("build_wiki_site.py")
    print("\nDone. Bot DB updated; wiki: wiki/site/index.html")
    print("Open wiki: make wiki-serve  (port 8091)  or  http://localhost:8080/wiki/ with admin")


if __name__ == "__main__":
    main()
