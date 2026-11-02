"""EuroSAT RGB loading: 27,000 Sentinel-2 64x64 JPEGs, 10 classes.

Put EuroSAT_RGB.zip (https://zenodo.org/records/7711810) in data/ or extract it
to data/eurosat_raw/. First call builds a cached uint8 array file.
Split: stratified 70 / 15 / 15 (seed 42).
"""
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from common import DATA, EUROSAT_CLASSES, SEED

CACHE = DATA / "eurosat_rgb_64.npz"


def _find_root():
    for cand in [DATA / "eurosat_raw" / "EuroSAT_RGB", DATA / "eurosat_raw" / "2750", DATA / "EuroSAT_RGB"]:
        if cand.exists():
            return cand
    z = DATA / "EuroSAT_RGB.zip"
    if z.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA / "eurosat_raw")
        return _find_root()
    raise FileNotFoundError("Download EuroSAT_RGB.zip from https://zenodo.org/records/7711810 into data/")


def load_eurosat():
    if not CACHE.exists():
        root = _find_root()
        X, y = [], []
        for ci, cname in enumerate(EUROSAT_CLASSES):
            for p in sorted((root / cname).glob("*.jpg")):
                X.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)); y.append(ci)
        np.savez_compressed(CACHE, X=np.stack(X), y=np.array(y, np.int64))
    d = np.load(CACHE)
    return d["X"], d["y"]


def eurosat_splits():
    X, y = load_eurosat()
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])
