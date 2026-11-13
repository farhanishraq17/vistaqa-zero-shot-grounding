"""Practice set for the weight-space programming test (WSP, EXPERIMENTS.md): COCO val2017 images proxy dev does not use.

Per image the same question twice: about an absent, frequently co-occurring category (absent=1) and about a present
single-instance category (absent=0). Templates are not proxy dev's. Split A (analysis) = first 200 images, B (feedback) = next 100.

    python src/build_wsp_practice.py   ->  Competition Data/wsp_practice/practice.json + <coco id>.jpg
"""
import json
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "Competition Data" / "external" / "annotations" / "instances_val2017.json"
OUT = ROOT / "Competition Data" / "wsp_practice"
TEMPLATES = ["What is the {x} doing?", "Where in the image is the {x}?", "What is the {x} made of?", "What is on top of the {x}?",
             "Describe the {x} in one sentence.", "What shape is the {x}?", "What is behind the {x}?",
             "Is the {x} in the foreground or the background?"]


def build():
    rng = random.Random(1)
    coco = COCO(str(ANN))
    name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    dev = {json.loads(p.read_text(encoding="utf-8"))["source"].get("coco_image_id")
           for p in (ROOT / "Competition Data" / "proxy_dev").glob("*.json")}
    per = {i: coco.loadAnns(coco.getAnnIds(imgIds=i)) for i in sorted(coco.getImgIds())}
    cooc = defaultdict(lambda: defaultdict(int))
    for anns in per.values():
        cs = {a["category_id"] for a in anns}
        for a in cs:
            for b in cs - {a}:
                cooc[a][b] += 1
    pool = []
    for i, anns in per.items():
        im = coco.loadImgs(i)[0]
        if i in dev or not anns:
            continue
        by = defaultdict(list)
        for a in anns:
            by[a["category_id"]].append(a)
        single = [c for c, v in sorted(by.items()) if len(v) == 1 and not v[0]["iscrowd"]
                  and v[0]["area"] >= 0.01 * im["width"] * im["height"]]
        score = defaultdict(int)
        for c in by:
            for d, k in cooc[c].items():
                if d not in by:
                    score[d] += k
        if single and score:  # absent: one of the 5 categories that co-occur most with what is there (not always "person")
            top = sorted(score, key=lambda d: (-score[d], d))[:5]
            pool.append((im, name[rng.choice(single)], name[rng.choice(top)]))
    rng.shuffle(pool)
    rows = []
    for k, (im, present, absent) in enumerate(pool[:300]):
        t = rng.choice(TEMPLATES)
        for x, ab in ((absent, 1), (present, 0)):
            rows.append({"coco_id": im["id"], "split": "A" if k < 200 else "B", "absent": ab, "object": x,
                         "question": t.format(x=x), "width": im["width"], "height": im["height"]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "practice.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    jobs = [(im["coco_url"], OUT / f"{im['id']}.jpg") for im, _, _ in pool[:300]]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda j: j[1].exists() or urllib.request.urlretrieve(*j), jobs))
    print(len(rows), "questions on", len(jobs), "images;", sum(r["coco_id"] in dev for r in rows), "proxy dev images")


if __name__ == "__main__":
    build()
