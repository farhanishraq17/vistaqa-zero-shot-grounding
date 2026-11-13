"""Count probe: does countfix (raise the 27B count to max(8B, SAM3)) hold outside COCO?

MathVista testmini (CC BY-SA 4.0) free-form integer questions matching the pipeline's COUNT_Q: CLEVR "subtract all X"
scenes, tables, bar charts, plots, natural images. Text GT only (no masks) — measured by judge text accuracy.
    python src/build_count_probe.py   ->  Competition Data/count_probe/<id>.json
"""
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO, FILE = "AI4Math/MathVista", "data/testmini-00000-of-00001-725687bf7a18d64b.parquet"
OUT = Path(__file__).resolve().parent.parent / "Competition Data" / "count_probe"
COUNT_Q = re.compile(r"\b(how many|count|number of)\b", re.I)  # = pipeline.COUNT_Q

rows = pq.read_table(hf_hub_download(REPO, FILE, repo_type="dataset"),
                     columns=["pid", "question", "answer", "question_type", "answer_type", "metadata"]).to_pylist()
sel = [(i, r) for i, r in enumerate(rows)
       if COUNT_Q.search(r["question"]) and r["question_type"] == "free_form" and r["answer_type"] == "integer"]
OUT.mkdir(parents=True, exist_ok=True)
for k, (row, r) in enumerate(sel, 1):
    m = r["metadata"]
    (OUT / f"{k}.json").write_text(json.dumps({
        "image": {"image_id": k, "width": m["img_width"], "height": m["img_height"]},
        "question": r["question"], "answer": r["answer"], "task_type": "counting", "task_domain": m["context"],
        "hallucination": 0, "num_instances": 0, "annotations": [],
        "source": {"url": f"hfpq:{REPO}|{FILE}|{row}", "pid": r["pid"], "source": m["source"]}}, indent=1), encoding="utf-8")
print(f"{len(sel)} counting questions -> {OUT}")
