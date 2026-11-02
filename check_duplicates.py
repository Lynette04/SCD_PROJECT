"""
check_duplicates.py
 
READ-ONLY check for exact and near-duplicate images in data/train, data/val and data/test,
plus simple per-class image statistics (a quick check for shortcut cues such as image size
or brightness that differ between the classes).
 
It does NOT move, delete, rename or modify any image, model or training file.
The only file it creates is duplicate_report.csv, next to this script.
 
Run it from the project folder that contains the 'data' folder:
    python check_duplicates.py
 
Requires: numpy and Pillow (pip install pillow, if missing).
"""
import csv
import hashlib
import os
from collections import Counter
 
import numpy as np
from PIL import Image
 
DATA_ROOT = "data"
SPLITS = ["train", "val", "test"]
HASH_SIZE = 8           # 8x8 difference hash = 64 bits
NEAR_THRESHOLD = 5      # max differing bits (out of 64) to flag two images as near-duplicates
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
REPORT_FILE = "duplicate_report.csv"
 
 
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
 
 
def difference_hash(img):
    """64-bit perceptual hash: compares neighbouring pixels of a small grayscale copy."""
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
                    rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
                    records.append({
                        "path": path,
                        "split": split,
                        "cls": cls,
                        "sha": file_sha256(path),
                        "hash": difference_hash(im),
                        "size": im.size,
                        "mean_rgb": rgb.mean(axis=(0, 1)),
                    })
    return records
 
 
def find_pairs(records):
    n = len(records)
    hashes = np.stack([r["hash"] for r in records])          # n x 64 booleans
    pairs = []
    for i in range(n):
        dist = (hashes[i + 1:] != hashes[i]).sum(axis=1)     # distances to all later images
        for offset in np.nonzero(dist <= NEAR_THRESHOLD)[0]:
            j = i + 1 + int(offset)
            a, b = records[i], records[j]
            pairs.append({
                "image_a": a["path"], "image_b": b["path"],
                "type": "exact" if a["sha"] == b["sha"] else "near",
                "hash_distance": int(dist[offset]),
                "split_a": a["split"], "split_b": b["split"],
                "class_a": a["cls"], "class_b": b["cls"],
                "crosses_splits": a["split"] != b["split"],
                "label_conflict": a["cls"] != b["cls"],
            })
    return pairs
 
 
def class_statistics(records):
    print("\nPer-class image statistics (all splits together)")
    print("-" * 60)
    for cls in sorted({r["cls"] for r in records}):
        subset = [r for r in records if r["cls"] == cls]
        sizes = Counter(r["size"] for r in subset)
        rgb = np.stack([r["mean_rgb"] for r in subset])
        brightness = rgb.mean(axis=1)
        print(f"{cls}: {len(subset)} images")
        print("  most common sizes (width, height):", sizes.most_common(3))
        print(f"  mean brightness: {brightness.mean():.1f} (spread across images: {brightness.std():.1f})")
        print("  mean R, G, B:", [round(float(v), 1) for v in rgb.mean(axis=0)])
    print("If image size or brightness differs sharply between the classes, a model could")
    print("learn those cues instead of cell shape. Treat that as a warning, not a verdict.")
 
 
def main():
    records = load_records()
    print(f"Images read: {len(records)}")
    for split in SPLITS:
        print(f"  {split}: {sum(1 for r in records if r['split'] == split)}")
 
    pairs = find_pairs(records)
    exact = [p for p in pairs if p["type"] == "exact"]
    cross = [p for p in pairs if p["crosses_splits"]]
    conflict = [p for p in pairs if p["label_conflict"]]
 
    print("\nDuplicate check")
    print("-" * 60)
    print(f"Pairs flagged (hash distance <= {NEAR_THRESHOLD}): {len(pairs)}")
    print(f"  identical files: {len(exact)}")
    print(f"  pairs that cross splits (possible leakage): {len(cross)}")
    print(f"  pairs with different class labels (label conflict): {len(conflict)}")
 
    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:
        fields = ["image_a", "image_b", "type", "hash_distance", "split_a", "split_b",
                  "class_a", "class_b", "crosses_splits", "label_conflict"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)
    print(f"Full list written to {REPORT_FILE}")
    print("Open the flagged pairs and look at them yourself: blood smear fields can look")
    print("alike without being the same image, so a flag is a candidate, not a verdict.")
 
    class_statistics(records)
 
 
if __name__ == "__main__":
    main()