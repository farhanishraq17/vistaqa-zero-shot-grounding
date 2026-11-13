"""Premise probe: does the premise check misfire outside COCO (math / science / charts / documents)?

MathVista testmini (CC BY-SA 4.0): 200 questions on their own image (premise holds) + 100 questions moved onto an image
of a different visual context (premise fails). No masks — only the premise check's flag rate is measured.
    python src/build_premise_probe.py   ->  Competition Data/premise_probe/<id>.json
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO, FILE = "AI4Math/MathVista", "data/testmini-00000-of-00001-725687bf7a18d64b.parquet"
OUT = Path(__file__).resolve().parent.parent / "Competition Data" / "premise_probe"

rows = pq.read_table(hf_hub_download(REPO, FILE, repo_type="dataset"), columns=["pid", "question", "metadata"]).to_pylist()
for i, r in enumerate(rows):
    r["row"] = i
rng = random.Random(0)
by_ctx = defaultdict(list)
for r in rows:
    by_ctx[r["metadata"]["context"]].append(r)
for v in by_ctx.values():
    rng.shuffle(v)
valid = []
while len(valid) < 200:  # round-robin over contexts -> charts, geometry, science figures, tables, scenes all present
    for v in by_ctx.values():
        if v and len(valid) < 200:
            valid.append(v.pop())
rest = [r for v in by_ctx.values() for r in v]
rng.shuffle(rest)
moved = []
for q in rest:
    if len(moved) == 100:
        break
    img = rng.choice([r for r in rows if r["metadata"]["context"] != q["metadata"]["context"]])
    moved.append((q, img))

OUT.mkdir(parents=True, exist_ok=True)
for k, (q, img, ok) in enumerate([(r, r, True) for r in valid] + [(q, i, False) for q, i in moved], 1):
    m = img["metadata"]
    (OUT / f"{k}.json").write_text(json.dumps({
        "image": {"image_id": k, "width": m["img_width"], "height": m["img_height"]},
        "question": q["question"], "premise_ok": ok, "context": m["context"], "q_context": q["metadata"]["context"],
        "source": {"url": f"hfpq:{REPO}|{FILE}|{img['row']}", "q_pid": q["pid"]}}, indent=1), encoding="utf-8")
print(f"{len(valid)} valid + {len(moved)} moved -> {OUT}")
