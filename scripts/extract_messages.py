import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.anonymize import anonymize

def normalize_text(x) -> str:

    if isinstance(x, str):
        return x
    if isinstance(x, list):
        parts = []
        for el in x:
            if isinstance(el, str):
                parts.append(el)
            elif isinstance(el, dict):
                t = el.get("text", "")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_messages.py /path/to/result.json")
        sys.exit(1)

    src = Path(sys.argv[1])
    data = json.loads(src.read_text(encoding="utf-8"))
    msgs = data.get("messages", [])
    print("Total objects in JSON messages:", len(msgs))

    out_csv = src.parent / "messages_extracted.csv"
    kept = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "date", "from", "from_id", "text_clean"])

        for m in msgs:

            if m.get("type") != "message":
                continue

            text = normalize_text(m.get("text", ""))
            if not text:
                continue


            if m.get("action"):
                continue

            text_clean = anonymize(text).strip()
            if not text_clean:
                continue

            w.writerow([m.get("id"), m.get("date"), m.get("from"), m.get("from_id"), text_clean])
            kept += 1

    print("Saved:", out_csv)
    print("Text messages kept:", kept)

if __name__ == "__main__":
    main()