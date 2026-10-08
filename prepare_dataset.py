"""Build dataset/{train,val,test}/<class>/ from the raw Kaggle 'Garbage Classification' folders.

Usage:  python prepare_dataset.py path/to/raw_dataset
The split is done on FILES, before any augmentation -> no train/val/test leakage.
"""
import random, shutil, sys
from pathlib import Path

MAP = {  # raw folder name -> EcoVision class
    "plastic": "plastic", "paper": "paper", "cardboard": "paper", "metal": "metal",
    "biological": "organic", "brown-glass": "glass", "green-glass": "glass", "white-glass": "glass",
}
SPLIT = (0.70, 0.15, 0.15)
random.seed(42)

raw, out = Path(sys.argv[1]), Path("dataset")
files = {}
for folder in raw.rglob("*"):
    if folder.is_dir() and folder.name.lower() in MAP:
        files.setdefault(MAP[folder.name.lower()], []).extend(
            p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

for cls, items in files.items():
    random.shuffle(items)
    n = len(items); a, b = int(n * SPLIT[0]), int(n * (SPLIT[0] + SPLIT[1]))
    for name, chunk in (("train", items[:a]), ("val", items[a:b]), ("test", items[b:])):
        d = out / name / cls; d.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(chunk):
            shutil.copy(p, d / f"{p.parent.name}_{i}{p.suffix.lower()}")
    print(f"{cls}: {n} images")
