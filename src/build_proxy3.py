"""Build proxy3, the crowded-scene dev set of the mask verifier round (VistaQA JSON format). Deterministic (seed 0).

RefCOCOg val (umd split, HF jxu124/refcocog): COCO train2014 images that hold 2+ objects of one category. Questions name
objects by referring expressions, so the other objects of that category are distractors. Templates and answer wording follow
build_proxy_dev.py: side of the image (70), left/right of another referent (70, no horizontal overlap), which appears larger
(60, mask area ratio >= 2.5). One question per image; images already in proxy dev are skipped. No false premises, no counting.

    python src/build_proxy3.py            # -> Competition Data/proxy3/<id>.json + <id>.jpg
"""
import json
import random
import re
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from build_proxy_dev import poly_rle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Competition Data" / "proxy3"
N = dict(comparison=60, spatial=70, side=70)  # rarest first (one question per image)
# expressions that give the answer away (position, depth, size) are skipped
GIVEAWAY = re.compile(r"\b(left|right|middle|cent(er|re)|side|top|bottom|front|back|behind|near(er|est)?|clos(e|er|est)|"
                      r"far(ther|thest)?|foreground|background|big(ger|gest)?|large(r|st)?|small(er|est)?|little|tiny|huge|"
                      r"tall(er|est)?|short(er|est)?)\b")


def phrase(raw):
    """Referring expression -> "the ..." (2-12 words after the article), or None."""
    s = re.sub(r"\s+", " ", raw.lower()).strip(" .,!?;:")
    s = re.sub(r"^(a|an|the|this|that) ", "", s)
    if not 2 <= len(s.split()) <= 12 or GIVEAWAY.search(s) or not re.fullmatch(r"[a-z0-9 ,'-]+", s):
        return None
    if re.search(r"\b(is|are)\b", s) and not re.search(r"\b(who|that|which) (is|are)\b", s):
        return None  # a sentence ("a man is standing"), not a noun phrase
    return "the " + s


def cap(s):
    return s[0].upper() + s[1:]


def build():
    rng = random.Random(0)
    pq_path = hf_hub_download("jxu124/refcocog", "data/validation-00000-of-00001-15168dfe7b5961e5.parquet", repo_type="dataset")
    seen = {json.loads(p.read_text(encoding="utf-8"))["source"]["coco_image_id"]
            for p in (ROOT / "Competition Data" / "proxy_dev").glob("*.json")}
    refs, info = defaultdict(list), {}
    for r in pq.read_table(pq_path).to_pylist():
        a, im = json.loads(r["raw_anns"]), json.loads(r["raw_image_info"])
        names = [n for n in (phrase(s["raw"]) for s in r["sentences"]) if n]
        if (names and im["id"] not in seen and not a["iscrowd"] and isinstance(a["segmentation"], list)
                and a["area"] >= 0.01 * im["width"] * im["height"]):
            info[im["id"]] = im
            refs[im["id"]].append((rng.choice(names), a))
    cands = defaultdict(list)
    for img_id, rs in sorted(refs.items()):
        half = info[img_id]["width"] / 2
        for n, a in rs:
            x, _, w, _ = a["bbox"]
            if x + w < half or x > half:  # the whole object on one side
                side = "left" if x + w < half else "right"
                cands["side"].append((img_id, f"Is {n} on the left or the right side of the image?",
                                      f"{cap(n)} is on the {side} side of the image.", [a]))
        for (na, a), (nb, b) in [(p, q) for p in rs for q in rs if p[1]["id"] < q[1]["id"] and p[0] != q[0]]:
            ax, _, aw, _ = a["bbox"]; bx, _, bw, _ = b["bbox"]
            if ax + aw < bx or bx + bw < ax:  # horizontally separated
                cands["spatial"].append((img_id, f"Is {na} to the left or to the right of {nb}?",
                                         f"{cap(na)} is to the {'left' if ax < bx else 'right'} of {nb}.", [a, b]))
            r = a["area"] / b["area"]
            if r >= 2.5 or r <= 1 / 2.5:
                cands["comparison"].append((img_id, f"Which appears larger in the image, {na} or {nb}?",
                                            f"{cap(na if r > 1 else nb)} appears larger.", [a, b]))
    used, rows = set(), []
    for task, n in N.items():
        pool = cands[task]; rng.shuffle(pool)
        for c in pool:
            if sum(t == task for t, _ in rows) == n:
                break
            if c[0] not in used:
                used.add(c[0]); rows.append((task, c))
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, (task, (img_id, q, ans, anns)) in enumerate(rows, 1):
        im = info[img_id]
        url = f"http://images.cocodataset.org/train2014/{im['file_name']}"
        rec = {"image": {"image_id": i, "width": im["width"], "height": im["height"], "file_name": f"{i}.jpg"},
               "question": q, "answer": ans, "task_type": "comparison" if task == "comparison" else "spatial",
               "task_domain": "coco", "num_instances": len(anns), "hallucination": 0,
               "annotations": [{"segmentation": poly_rle(a["segmentation"], im["height"], im["width"])} for a in anns],
               "source": {"coco_image_id": img_id, "url": url, "template": task, "refcocog_ann_ids": [a["id"] for a in anns]}}
        (OUT / f"{i}.json").write_text(json.dumps(rec), encoding="utf-8")
        jobs.append((url, OUT / f"{i}.jpg"))
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda j: j[1].exists() or urllib.request.urlretrieve(*j), jobs))
    print({t: sum(r[0] == t for r in rows) for t in N}, "of candidates", {t: len(v) for t, v in cands.items()}, "total", len(rows))


if __name__ == "__main__":
    sys.exit(build())
