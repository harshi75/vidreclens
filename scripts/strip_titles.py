import csv
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "bundle")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "bundle_public")
shutil.copytree(src, dst, dirs_exist_ok=True)
with open(src / "items.csv", newline="", encoding="utf-8") as f:
    ids = [r["item"] for r in csv.DictReader(f)]
with open(dst / "items.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["item", "title"])
    for i in ids:
        w.writerow([i, ""])
print(f"wrote {dst} with {len(ids)} items and no titles")
