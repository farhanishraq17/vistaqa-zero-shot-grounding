"""Proxy-dev diagnostics for scored runs (never run this on VistaQA — zero-shot rule).

    python Scripts/analyze.py RUN [RUN ...]
Needs runs/RUN/{raw.jsonl, per_sample.jsonl, judge_out/verdicts.jsonl, gt.jsonl} (from fleet collect + judge score).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Competition Data" / "VistaQA-Evaluation" / "src"))
from mask_evaluator import MaskEvaluator  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from pipeline import ABSENT  # noqa: E402
NUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def count_in(text):
    m = re.search(r"\b(\d+|" + "|".join(NUM) + r")\b", text.lower())
    return None if not m else int(m.group(1)) if m.group(1).isdigit() else NUM[m.group(1)]


def read(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def analyze(run):
    d = ROOT / "runs" / run
    gt = {r["image"]["image_id"]: r for r in read(d / "gt.jsonl")}
    raw = {r["image_id"]: r for r in read(d / "raw.jsonl")}
    pred = {r["image"]["image_id"]: r for r in read(d / "preds.jsonl")}
    ver = {r["image_id"]: r["correct"] for r in read(d / "judge_out" / "verdicts.jsonl")}
    per = {r["image_id"]: r["standard_average"] for r in read(d / "per_sample.jsonl")}
    me = MaskEvaluator()
    g = defaultdict(lambda: defaultdict(list))
    flags = defaultdict(int)
    for i, r in gt.items():
        t, p = r["task_type"], pred[i]["prediction"]
        s = me.score([a["segmentation"] for a in r["annotations"]], p["masks"])
        for k in (t, "ALL"):
            g[k]["grove"].append(per[i]); g[k]["text"].append(ver[i]); g[k]["miou"].append(s)
            g[k]["nmask"].append(len(p["masks"])); g[k]["sec"].append(raw[i].get("sec", 0))
        flags["parse_fail"] += not raw[i]["parse_ok"]
        if ABSENT.search(p["text"]) and p["masks"]:
            flags["says_absent_but_masks"] += 1
        if not ABSENT.search(p["text"]) and not p["masks"] and not r["hallucination"]:
            flags["grounded_no_mask"] += 1
        if r["hallucination"] and p["masks"]:
            flags["halluc_with_masks"] += 1
        c = count_in(p["text"])
        if t == "counting" and c is not None and c != len(p["masks"]):
            flags["count_ne_masks"] += 1
    print(f"== {run}")
    print(f"{'task':15s} {'n':>4s} {'GROVE':>6s} {'text':>6s} {'mIoU':>6s} {'#m':>5s} {'s/it':>5s}")
    for k in sorted(g, key=lambda k: (k == "ALL", k)):
        v = g[k]; n = len(v["grove"]); m = lambda x: 100 * sum(x) / n  # noqa: E731
        print(f"{k:15s} {n:4d} {m(v['grove']):6.1f} {m(v['text']):6.1f} {m(v['miou']):6.1f} {sum(v['nmask']) / n:5.2f} {sum(v['sec']) / n:5.1f}")
    print("flags", dict(flags))


if __name__ == "__main__":
    for r in sys.argv[1:]:
        analyze(r)
