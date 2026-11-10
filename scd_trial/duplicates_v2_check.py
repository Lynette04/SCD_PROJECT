"""
check_duplicates_v2.py
 
READ-ONLY duplicate check for data/train, data/val and data/test.
It does NOT move, delete, rename or modify any image, model or training file.
It writes one file, duplicate_report_v2.csv, next to this script.
 
Why a second version: with a coarse 8x8 hash, many unrelated blood smear fields looked
"similar". This version uses a finer 16x16 hash (256 bits) and prints a table showing, for
each distance threshold, how many pairs are flagged and how often those pairs sit in
different splits or have different class labels.
 
How to read the table: two unrelated images land in different splits about 47% of the time
and in different classes about 38% of the time. Real duplicates almost never have different
class labels. So where the label-conflict percentage is still near 38%, the flagged pairs are
just coincidentally similar images. Where it falls toward 0%, they are genuine duplicates.
 
Run from the project folder that contains the 'data' folder:
    python check_duplicates_v2.py
"""
import csv
import hashlib
import os
 
import numpy as np
from PIL import Image
 
DATA_ROOT = "data"
SPLITS = ["train", "val", "test"]
HASH_SIZE = 16                                 # 16x16 difference hash = 256 bits
TABLE_THRESHOLDS = [0, 2, 4, 8, 12, 16, 24]    # bit-distance thresholds shown in the table
REPORT_THRESHOLD = 8                           # pairs at or below this go into the CSV
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
REPORT_FILE = "duplicate_report_v2.csv"
 
 
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
 
 
def difference_hash(img):
    small = img.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS)
    a = np.asarray(small, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()
 
 
def load_records():
    records = []
    for split in SPLITS:
        split_dir = os.path.join(DATA_ROOT, split)
        if not os.path.isdir(split_dir):
            raise SystemExit(f"Folder not found: {split_dir}. Run this from the project folder.")
        for cls in sorted(os.listdir(split_dir)):
            folder = os.path.join(split_dir, cls)
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if not name.lower().endswith(EXTS):
                    continue
                path = os.path.join(folder, name)
                with Image.open(path) as im:
                    im.load()
                    records.append({"path": path, "split": split, "cls": cls,
                                    "sha": file_sha256(path), "hash": difference_hash(im)})
    return records
 
 
def main():
    records = load_records()
    n = len(records)
    print(f"Images read: {n}")
 
    hashes = np.stack([r["hash"] for r in records])
    splits = np.array([r["split"] for r in records])
    classes = np.array([r["cls"] for r in records])
 
    # full distance matrix, built one row at a time
    dist = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        dist[i] = (hashes != hashes[i]).sum(axis=1)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    cross_split = splits[:, None] != splits[None, :]
    conflict = classes[:, None] != classes[None, :]
 
    total_pairs = upper.sum()
    print("\nExpected for unrelated image pairs: "
          f"{100 * (cross_split & upper).sum() / total_pairs:.0f}% in different splits, "
          f"{100 * (conflict & upper).sum() / total_pairs:.0f}% with different class labels")
 
    print("\nThreshold (bits of 256) | pairs flagged | in different splits | with different labels")
    print("-" * 78)
    for t in TABLE_THRESHOLDS:
        mask = upper & (dist <= t)
        count = int(mask.sum())
        if count == 0:
            print(f"{t:>23} | {0:>13} | {'-':>19} | {'-':>21}")
            continue
        pct_cross = 100 * (mask & cross_split).sum() / count
        pct_conf = 100 * (mask & conflict).sum() / count
        print(f"{t:>23} | {count:>13} | {pct_cross:>18.0f}% | {pct_conf:>20.0f}%")
 
    # identical files are always listed in full
    sha_to_idx = {}
    for idx, r in enumerate(records):
        sha_to_idx.setdefault(r["sha"], []).append(idx)
    exact_groups = [g for g in sha_to_idx.values() if len(g) > 1]
    print(f"\nByte-identical file groups: {len(exact_groups)}")
    for g in exact_groups:
        print("  " + "  <->  ".join(f"{records[i]['path']} ({records[i]['split']}/{records[i]['cls']})" for i in g))
 
    rows = []
    ii, jj = np.nonzero(upper & (dist <= REPORT_THRESHOLD))
    for i, j in zip(ii, jj):
        a, b = records[i], records[j]
        rows.append({"image_a": a["path"], "image_b": b["path"],
                     "type": "exact" if a["sha"] == b["sha"] else "near",
                     "hash_distance_bits": int(dist[i, j]),
                     "split_a": a["split"], "split_b": b["split"],
                     "class_a": a["cls"], "class_b": b["cls"],
                     "crosses_splits": a["split"] != b["split"],
                     "label_conflict": a["cls"] != b["cls"]})
    rows.sort(key=lambda r: r["hash_distance_bits"])
    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image_a"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} pairs at or below {REPORT_THRESHOLD} bits written to {REPORT_FILE}, closest first.")
    print("Open the closest pairs and compare the images yourself before deciding anything.")
 
 
if __name__ == "__main__":
    main()