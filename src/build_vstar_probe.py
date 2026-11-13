"""High-resolution probe (round R6): does a larger input cap than max_pixels 1280x1280 help the answerer?

46% of VistaQA images are larger than 1.64 MP, but no proxy dev or proxy2 image is, so neither set can test it.
V*Bench (craigwu/vstar_bench; SA-1B photos, ~2246x1582; small targets in large scenes), all 238 questions:
direct_attributes 115, relative_position 76, OCR 30, GPT4V-hard 17. Free-form question; GT answer = options[0]
(V* lists the correct option first). Text GT only (no masks); the target boxes (xywh px) stay in `source` for the
box-hit metric, read only locally. Images are fetched in the kernel from the pinned HF revision.
    python src/build_vstar_probe.py   ->  Competition Data/vstar_probe/<id>.json
"""
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from PIL import Image

REPO = "craigwu/vstar_bench"
ROOT = Path(__file__).resolve().parent.parent / "Competition Data"
OUT = ROOT / "vstar_probe"
DIRS = ("direct_attributes", "relative_position", "OCR", "GPT4V-hard")

rev = HfApi().dataset_info(REPO, token=False).sha
src = Path(snapshot_download(REPO, repo_type="dataset", revision=rev, local_dir=ROOT / "external" / "vstar_bench", token=False))
OUT.mkdir(parents=True, exist_ok=True)
k = 0
for d in DIRS:
    for js in sorted((src / d).glob("*.json")):
        im = next(p for p in js.parent.glob(js.stem + ".*") if p.suffix.lower() != ".json")
        r = json.loads(js.read_text(encoding="utf-8"))
        w, h = Image.open(im).size
        k += 1
        (OUT / f"{k}.json").write_text(json.dumps({
            "image": {"image_id": k, "width": w, "height": h},
            "question": r["question"], "answer": r["options"][0], "task_type": d, "task_domain": "vstar",
            "hallucination": 0, "num_instances": 0, "annotations": [],
            "source": {"url": f"https://huggingface.co/datasets/{REPO}/resolve/{rev}/{d}/{im.name}",
                       "targets": r["target_object"], "bbox_xywh": r["bbox"], "options": r["options"]}}, indent=1),
            encoding="utf-8")
print(f"{k} questions (revision {rev[:8]}) -> {OUT}")
