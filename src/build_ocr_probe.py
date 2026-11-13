"""OCR probe: does the zoom re-read help on clean scene-text GT? (COCO-Text in proxy dev is too noisy to tell.)

TextVQA val (CC BY 4.0; OpenImages photos), first validation shard: questions matching the pipeline's READ_Q (= the
zoom trigger) whose top answer has >= 5/10 annotator votes. Text GT only (no masks) — measured by judge text accuracy.
    python src/build_ocr_probe.py   ->  Competition Data/ocr_probe/<id>.json
"""
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO, FILE = "lmms-lab/textvqa", "data/validation-00000-of-00003.parquet"
OUT = Path(__file__).resolve().parent.parent / "Competition Data" / "ocr_probe"
READ_Q = re.compile(r"\b(text|written|writing|says?|read|words?|letters?|spell\w*)\b", re.I)  # = pipeline.READ_Q

f = pq.ParquetFile(hf_hub_download(REPO, FILE, repo_type="dataset"))
rows = f.read(columns=["question_id", "question", "answers", "image_width", "image_height"]).to_pylist()
OUT.mkdir(parents=True, exist_ok=True)
k = 0
for row, r in enumerate(rows):
    ans, votes = Counter(a.lower().strip() for a in r["answers"]).most_common(1)[0]
    if not READ_Q.search(r["question"]) or votes < 5 or ans in ("unanswerable", "yes", "no"):
        continue
    k += 1
    (OUT / f"{k}.json").write_text(json.dumps({
        "image": {"image_id": k, "width": r["image_width"], "height": r["image_height"]},
        "question": r["question"], "answer": ans, "task_type": "ocr", "task_domain": "textvqa",
        "hallucination": 0, "num_instances": 0, "annotations": [],
        "source": {"url": f"hfpq:{REPO}|{FILE}|{row}", "question_id": r["question_id"], "votes": votes}}, indent=1),
        encoding="utf-8")
print(f"{k} reading questions -> {OUT}")
