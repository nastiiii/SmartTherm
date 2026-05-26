
"""Run full data pipeline from result.json (in order)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
JSON = DATA / "result.json"

STEPS = [
    ([sys.executable, str(ROOT / "scripts/extract_messages.py"), str(JSON)], "extract messages"),
    ([sys.executable, str(ROOT / "scripts/split_into_dialogs.py")], "split dialogs"),
    ([sys.executable, str(ROOT / "scripts/extract_questions_only.py")], "extract questions"),
    ([sys.executable, str(ROOT / "scripts/filter_domain_questions.py")], "domain questions"),
    ([sys.executable, str(ROOT / "scripts/cluster_questions.py")], "cluster questions"),
    ([sys.executable, str(ROOT / "scripts/top_clusters_domain_questions.py")], "top clusters"),
    ([sys.executable, str(ROOT / "scripts/extract_qa_from_sessions.py")], "extract QA"),
    ([sys.executable, str(ROOT / "scripts/filter_domain_qa.py")], "domain QA"),
    ([sys.executable, str(ROOT / "scripts/filter_good_answers.py")], "good QA"),
    ([sys.executable, str(ROOT / "scripts/find_experts.py")], "experts"),
]


def main() -> None:
    if not JSON.exists():
        print("Missing", JSON)
        sys.exit(1)
    for cmd, name in STEPS:
        print("\n===", name, "===")
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print("FAILED:", name)
            sys.exit(r.returncode)
    print("\nPipeline done.")


if __name__ == "__main__":
    main()
