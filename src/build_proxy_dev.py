"""Build the external proxy dev set (VistaQA JSON format). Deterministic (seed 0).
v1 (ids 1-420): COCO val2017 instances. v2 adds attribute (PACO-LVIS val), OCR (COCO-Text v2), +30 hallucination.

    python src/build_proxy_dev.py            # -> Competition Data/proxy_dev/<id>.json + <id>.jpg

Only public VistaQA *statistics* inform the mix (task types, ~27% hallucination); no VistaQA sample is used.
"""
import json
import random
import re
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "Competition Data" / "external" / "annotations" / "instances_val2017.json"
OUT = ROOT / "Competition Data" / "proxy_dev"
N = dict(counting=100, identification=70, spatial=70, comparison=60, hallucination=120)

PLURAL = {"person": "people", "mouse": "mice", "sheep": "sheep", "skis": "pairs of skis", "scissors": "pairs of scissors",
          "knife": "knives", "bus": "buses", "wine glass": "wine glasses", "couch": "couches", "sandwich": "sandwiches",
          "bench": "benches", "toothbrush": "toothbrushes", "broccoli": "pieces of broccoli", "skateboard": "skateboards"}
SUPER = {"animal": "animal", "vehicle": "vehicle", "food": "food item", "furniture": "piece of furniture",
         "electronic": "electronic device", "appliance": "appliance", "sports": "piece of sports equipment",
         "kitchen": "kitchen item", "accessory": "accessory"}


def plural(n):
    return PLURAL.get(n, n + "s")


def article(n):
    return ("an " if n[0] in "aeiou" else "a ") + n


def build():
    rng = random.Random(0)
    coco = COCO(str(ANN))
    cat = {c["id"]: c for c in coco.loadCats(coco.getCatIds())}
    cooc = defaultdict(lambda: defaultdict(int))  # POPE-style adversarial absent categories
    per_img = {}
    for img_id in sorted(coco.getImgIds()):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        if not anns:
            continue
        per_img[img_id] = anns
        cs = {a["category_id"] for a in anns}
        for a in cs:
            for b in cs:
                if a != b:
                    cooc[a][b] += 1

    def rle(a):
        r = coco.annToRLE(a)
        return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode()}

    cands = defaultdict(list)
    for img_id, anns in per_img.items():
        im = coco.loadImgs(img_id)[0]
        area = im["width"] * im["height"]
        by = defaultdict(list)
        for a in anns:
            by[a["category_id"]].append(a)
        clean = {c: v for c, v in by.items() if not any(a["iscrowd"] for a in v)}
        # counting: every instance clearly visible (no tiny/crowd ambiguity)
        for c, v in clean.items():
            if 2 <= len(v) <= 12 and all(a["area"] >= 0.002 * area for a in v):
                n = plural(cat[c]["name"])
                q = rng.choice([f"How many {n} are there in the image?", f"How many {n} can you see?", f"Count the {n} in the picture."])
                cands["counting"].append((img_id, q, f"There are {len(v)} {n}.", v))
        singles = {c: v[0] for c, v in clean.items() if len(v) == 1 and v[0]["area"] >= 0.01 * area}
        # identification: exactly one instance within a supercategory
        sup = defaultdict(list)
        for a in anns:
            sup[cat[a["category_id"]]["supercategory"]].append(a)
        for s, v in sup.items():
            if s in SUPER and len(v) == 1 and not v[0]["iscrowd"] and v[0]["area"] >= 0.02 * area:
                q = rng.choice([f"What kind of {SUPER[s]} is shown in the image?", f"Which {SUPER[s]} appears in this picture?"])
                cands["identification"].append((img_id, q, f"It is {article(cat[v[0]['category_id']]['name'])}.", v))
        pairs = [(a, b) for a in singles.values() for b in singles.values() if a["id"] < b["id"]]
        for a, b in pairs:
            na, nb = cat[a["category_id"]]["name"], cat[b["category_id"]]["name"]
            ax, ay, aw, ah = a["bbox"]; bx, by_, bw, bh = b["bbox"]
            if ax + aw < bx or bx + bw < ax:  # horizontally separated
                side = "left" if ax < bx else "right"
                cands["spatial"].append((img_id, f"Is the {na} to the left or to the right of the {nb}?",
                                         f"The {na} is to the {side} of the {nb}.", [a, b]))
            elif ay + ah < by_ or by_ + bh < ay:
                side = "above" if ay < by_ else "below"
                cands["spatial"].append((img_id, f"Is the {na} above or below the {nb}?", f"The {na} is {side} the {nb}.", [a, b]))
            r = a["area"] / b["area"]
            if r >= 2.5 or r <= 1 / 2.5:
                big = na if r > 1 else nb
                cands["comparison"].append((img_id, f"Which appears larger in the image, the {na} or the {nb}?",
                                            f"The {big} appears larger.", [a, b]))
        # hallucination: frequently co-occurring category that is absent (incl. crowd annotations)
        present = set(by)
        scores = defaultdict(int)
        for c in present:
            for d, k in cooc[c].items():
                if d not in present:
                    scores[d] += k
        if scores and singles:
            absent = cat[max(scores, key=scores.get)]["name"]
            ref = cat[rng.choice(sorted(singles))]["name"]
            q = rng.choice([f"How many {plural(absent)} are there in the image?", f"What color is the {absent}?",
                            f"Is the {absent} to the left or to the right of the {ref}?",
                            f"What is written on the {absent}?", f"Which is larger, the {absent} or the {ref}?",
                            f"What is the {absent} next to?"])
            cands["hallucination"].append((img_id, q, f"There is no {absent} in the image.", []))

    used, rows, pools = set(), [], {}

    def take(task, n):  # one sample per image across the whole set
        if task not in pools:
            pools[task] = list(cands[task]); rng.shuffle(pools[task])
        k = 0
        while k < n and pools[task]:
            c = pools[task].pop(0)
            if c[0] not in used:
                im = coco.loadImgs(c[0])[0]
                used.add(c[0]); rows.append((task, (im, c[1], c[2], [rle(a) for a in c[3]]))); k += 1

    for task, n in N.items():  # v1 (ids 1..420) — order must stay fixed
        take(task, n)
    rows += attribute_rows(rng, 60) + ocr_rows(rng, 50)  # v2 additions (ids 421+)
    take("hallucination", 30)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, (task, (im, q, ans, rles)) in enumerate(rows, 1):
        rec = {"image": {"image_id": i, "width": im["width"], "height": im["height"], "file_name": f"{i}.jpg"},
               "question": q, "answer": ans, "task_type": task, "task_domain": "coco",
               "num_instances": len(rles), "hallucination": int(task == "hallucination"),
               "annotations": [{"segmentation": r} for r in rles], "source": {"coco_image_id": im["id"], "url": im["coco_url"]}}
        (OUT / f"{i}.json").write_text(json.dumps(rec), encoding="utf-8")
        jobs.append((im["coco_url"], OUT / f"{i}.jpg"))
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda j: j[1].exists() or urllib.request.urlretrieve(*j), jobs))
    print({t: sum(r[0] == t for r in rows) for t in [*N, "attribute", "ocr"]}, "total", len(rows))


def poly_rle(polys, h, w):
    from pycocotools import mask as mu
    r = mu.merge(mu.frPyObjects(polys, h, w))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode()}


def attribute_rows(rng, n):
    """PACO-LVIS val: single-instance objects with exactly one plain dominant color."""
    d = json.loads((ANN.parent.parent / "paco_lvis_v1_val.json").read_text())
    cats = {c["id"]: re.sub(r"_\(.*?\)", "", c["name"]).replace("_", " ") for c in d["categories"]}
    colors = {a["id"]: a["name"] for a in d["attributes"]}
    imgs = {i["id"]: i for i in d["images"]}
    objs = [a for a in d["annotations"] if a["obj_ann_id"] == a["id"]]
    per = defaultdict(lambda: defaultdict(list))
    for a in objs:
        per[a["image_id"]][a["category_id"]].append(a)
    pool = []
    for img_id in sorted(per):
        im = imgs[img_id]
        for c, v in sorted(per[img_id].items()):
            a = v[0]
            col = [colors[i] for i in a["dom_color_ids"]]
            if len(v) == 1 and len(col) == 1 and "_" not in col[0] and not a["unknown_color"] and a["area"] >= 0.02 * im["width"] * im["height"]:
                q = rng.choice([f"What color is the {cats[c]}?", f"What is the color of the {cats[c]} in the image?"])
                pool.append(("attribute", (im, q, f"The {cats[c]} is {col[0]}.", [poly_rle(a["segmentation"], im["height"], im["width"])])))
    rng.shuffle(pool)
    out, seen = [], set()
    for r in pool:
        if r[1][0]["id"] not in seen and len(out) < n:
            seen.add(r[1][0]["id"]); out.append(r)
    return out


def ocr_rows(rng, n):
    """COCO-Text v2: images whose only text annotation is one legible, English, machine-printed word."""
    d = json.loads((ANN.parent.parent / "cocotext.v2.json").read_text())
    pool = []
    for img_id, ann_ids in sorted(d["imgToAnns"].items(), key=lambda x: int(x[0])):
        if len(ann_ids) != 1:
            continue
        a, im = d["anns"][str(ann_ids[0])], d["imgs"][img_id]
        s = a["utf8_string"].strip()
        if (a["legibility"] == "legible" and a["class"] == "machine printed" and a["language"] == "english"
                and len(re.sub(r"[^A-Za-z0-9]", "", s)) >= 3 and a["area"] >= 0.003 * im["width"] * im["height"]):
            im = {**im, "coco_url": f"http://images.cocodataset.org/train2014/{im['file_name']}"}
            q = rng.choice(["What does the text in the image say?", "What word is written in this picture?",
                            "Read the text that appears in the image."])
            pool.append(("ocr", (im, q, f'The text says "{s}".', [poly_rle([a["mask"]], im["height"], im["width"])])))
    rng.shuffle(pool)
    return pool[:n]


if __name__ == "__main__":
    sys.exit(build())
