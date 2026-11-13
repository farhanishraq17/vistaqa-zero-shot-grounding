"""Math probe: does thinking help on figures that need reasoning? (proxy2's synthetic math is too easy: text 97%)

MathVista testmini (CC BY-SA 4.0), free-form questions on figures, not COUNT_Q (those are the count probe), no natural images.
Fixed seed, 200 questions. Text GT only (no masks), measured by judge text accuracy.
    python src/build_math_probe.py   ->  Competition Data/math_probe/<id>.json
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO, FILE = "AI4Math/MathVista", "data/testmini-00000-of-00001-725687bf7a18d64b.parquet"
OUT = Path(__file__).resolve().parent.parent / "Competition Data" / "math_probe"
COUNT_Q = re.compile(r"\b(how many|count|number of)\b", re.I)  # = pipeline.COUNT_Q

rows = pq.read_table(hf_hub_download(REPO, FILE, repo_type="dataset"),
                     columns=["pid", "question", "answer", "question_type", "answer_type", "metadata"]).to_pylist()
pool = [(i, r) for i, r in enumerate(rows) if r["question_type"] == "free_form" and not COUNT_Q.search(r["question"])
        and r["metadata"]["context"] != "natural image"]
sel = sorted(random.Random(20261104).sample(pool, 200), key=lambda x: x[0])
OUT.mkdir(parents=True, exist_ok=True)
for k, (row, r) in enumerate(sel, 1):
    m = r["metadata"]
    (OUT / f"{k}.json").write_text(json.dumps({
        "image": {"image_id": k, "width": m["img_width"], "height": m["img_height"]},
        "question": r["question"], "answer": r["answer"], "task_type": "math", "task_domain": m["context"],
        "hallucination": 0, "num_instances": 0, "annotations": [],
        "source": {"url": f"hfpq:{REPO}|{FILE}|{row}", "pid": r["pid"], "source": m["source"]}}, indent=1), encoding="utf-8")
print(f"{len(sel)} of {len(pool)} -> {OUT}", Counter(r["metadata"]["context"] for _, r in sel).most_common())
