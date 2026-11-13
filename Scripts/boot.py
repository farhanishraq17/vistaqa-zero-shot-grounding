"""python Scripts/boot.py (from the repo root) A B [A B ...]: paired bootstrap of GROVE(A) - GROVE(B) over per_sample.jsonl (10,000 resamples, seed 0).
A run may be R1+R2 (pooled over two datasets, keyed by position, e.g. wsp1-dev-s3+wsp1-p2-s3 wsp0-dev-s3+wsp0-p2-s3)."""
import json, sys
import numpy as np


def ps(runs):
    return {(k, x["image_id"]): x["standard_average"] for k, r in enumerate(runs.split("+"))
            for x in map(json.loads, open(f"runs/{r}/per_sample.jsonl", encoding="utf-8"))}


for a, b in zip(sys.argv[1::2], sys.argv[2::2]):
    A, B = ps(a), ps(b)
    ids = sorted(A)
    assert set(ids) == set(B)
    d = np.array([A[i] - B[i] for i in ids]) * 100
    idx = np.random.default_rng(0).integers(0, len(d), (10000, len(d)))
    m = d[idx].mean(1)
    print(f"{a} - {b}: {d.mean():+.2f} (95% CI {np.percentile(m, 2.5):+.2f} to {np.percentile(m, 97.5):+.2f})")
