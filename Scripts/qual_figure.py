"""qual_figure.py: the report's qualitative appendix figure (made after development closed; rule in EXPERIMENTS.md).

One panel per outcome of the entry `final` (AWQ verdicts, official per-sample mask score S), drawn with random.Random(0)
from id-sorted candidates, redrawing when the domain is already used. GT masks: orange outline; predicted masks: blue fill.
Under each panel, a box sets our GROVE beside the best published system's on the panel's task, domain and question type
(the published systems' per-sample outputs are not released, so the comparison is by category).
    python Scripts/qual_figure.py   ->  report/qual.pdf
"""
import json
import random
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from pycocotools import mask as mu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Competition Data" / "VistaQA-Evaluation" / "src"))
sys.path.insert(0, str(ROOT / "src"))
from data_loader import load_and_merge_samples  # noqa: E402
from mask_evaluator import MaskEvaluator  # noqa: E402
from pipeline import ABSENT  # noqa: E402

DATA, RUN = ROOT / "Competition Data" / "VistaQA", ROOT / "runs" / "final"
CATS = [("Right answer, good masks", lambda g, t, s, a, m: not g["hallucination"] and t and s >= 0.8),
        ("False premise caught", lambda g, t, s, a, m: g["hallucination"] and t and not m),
        ("Right answer, poor masks", lambda g, t, s, a, m: not g["hallucination"] and t and s < 0.3),
        ("Wrong answer", lambda g, t, s, a, m: not g["hallucination"] and not t and not ABSENT.search(a)),
        ("False alarm: says absent", lambda g, t, s, a, m: not g["hallucination"] and ABSENT.search(a) and not m),
        ("False premise, judged wrong", lambda g, t, s, a, m: g["hallucination"] and not t)]

# category GROVE: (label, ours with the AWQ judge, best published, its system), copied from report table tab:main
CAT = {"identification": ("Identification", 50.77, 53.62, "GPT-5.4-T"), "attribute": ("Attribute", 52.61, 47.46, "Qwen3-VL-4B"),
       "ocr": ("OCR", 58.37, 44.16, "GPT-5.4-T"), "spatial": ("Spatial", 41.75, 38.86, "GPT-5.4-T"),
       "counting": ("Counting", 44.57, 52.21, "GPT-5.4-T"), "comparison": ("Comparison", 43.24, 38.93, "GPT-5.4-T"),
       "indoor": ("Indoor", 54.09, 45.39, "GPT-5.4-T"), "outdoor": ("Outdoor", 55.65, 47.63, "Qwen3-VL-32B"),
       "math": ("Math", 26.14, 41.51, "GPT-5.4-T"), "av": ("AV", 67.55, 59.39, "Qwen3-VL-4B"),
       "robotics": ("Robotics", 32.60, 47.01, "GPT-5.4-T"), "science": ("Science", 48.63, 39.87, "Gemini 3"),
       0: ("Grounded", 40.03, 44.65, "GPT-5.4-T"), 1: ("False premise", 70.88, 60.72, "Qwen3-VL-4B")}
OURS_C, THEM_C, GREY = "#1f5fbf", "#3a3a3a", "#6f6f6f"


def compare(g):
    """Highlighted box: our category GROVE beside the best published system's, the higher one bold."""
    rows = [CAT[g["task_type"]], CAT[g["task_domain"]], CAT[g["hallucination"]]]
    ta = lambda s, **k: TextArea(s, textprops={"fontsize": 5.5, "family": "DejaVu Sans", **k})
    col = lambda cells, align: VPacker(children=cells, align=align, pad=0, sep=1.6)
    head = dict(color=GREY, fontsize=5.0)
    cols = [col([ta("GROVE", **head)] + [ta(r[0], color=GREY) for r in rows], "left"),
            col([ta("Ours", color=OURS_C, fontweight="bold", fontsize=5.0)]
                + [ta(f"{r[1]:.1f}", color=OURS_C, fontweight="bold" if r[1] > r[2] else "normal") for r in rows], "right"),
            col([ta("Best published", **head)]
                + [HPacker(children=[ta(f"{r[2]:.1f}", color=THEM_C, fontweight="bold" if r[2] > r[1] else "normal"),
                                     ta(r[3], color=GREY, fontsize=4.8)], align="baseline", pad=0, sep=2.5) for r in rows], "left")]
    return HPacker(children=cols, align="top", pad=0, sep=5)


gt = {int(p.stem): json.loads(p.read_text(encoding="utf-8")) for p in DATA.glob("*.json")}
pred = {(p := json.loads(l))["image"]["image_id"]: p["prediction"] for l in (RUN / "preds.jsonl").open(encoding="utf-8") if l.strip()}
ver = {(v := json.loads(l))["image_id"]: v["correct"] for l in (RUN / "judge_out" / "verdicts.jsonl").open(encoding="utf-8") if l.strip()}
me = MaskEvaluator()
S = {s.image_id: me.score(s.gt_masks, s.pred_masks) for s in load_and_merge_samples(str(RUN / "gt.jsonl"), str(RUN / "preds.jsonl"))}

rng, used, picks = random.Random(0), set(), []
for name, ok in CATS:
    cands = [i for i in sorted(gt) if ok(gt[i], ver[i], S[i], pred[i]["text"], pred[i]["masks"])]
    for _ in range(50):
        i = rng.choice(cands)
        if gt[i]["task_domain"] not in used:
            break
    used.add(gt[i]["task_domain"]); picks.append((name, i, len(cands)))
print([(n, i, c) for n, i, c in picks])


def dec(rle, size):
    m = mu.decode({"size": rle["size"], "counts": rle["counts"].encode() if isinstance(rle["counts"], str) else rle["counts"]})
    return np.asarray(Image.fromarray(m * 255).resize(size, Image.NEAREST)) > 127


def gt_rles(g):
    for a in g["annotations"]:
        s = a["segmentation"]
        yield json.loads(s.replace("'", '"')) if isinstance(s, str) else s


wrap = lambda s, n=3: "\n".join(textwrap.wrap(s, 36)[:n]) + ("..." if len(textwrap.wrap(s, 36)) > n else "")
fig, axes = plt.subplots(2, 3, figsize=(5.5, 6.75))
texts = []
for ax, (name, i, n) in zip(axes.ravel(), picks):
    g, p = gt[i], pred[i]
    im = Image.open(next(DATA.glob(f"{i}.*[gG]"))).convert("RGB")
    k = 360 / max(im.size); size = (max(1, round(im.width * k)), max(1, round(im.height * k)))
    ax.imshow(im.resize(size))
    for r in p["masks"]:
        m = dec(r, size); ax.imshow(np.dstack([np.zeros(m.shape)] * 2 + [np.ones(m.shape), 0.5 * m]))
    for r in gt_rles(g):
        ax.contour(dec(r, size), levels=[0.5], colors="#e66101", linewidths=0.9)
    ax.set_axis_off()
    ax.set_title(name + "\n" + f"({g['task_domain']}, {g['task_type']})", fontsize=6.3, pad=2)
    texts.append((ax, g, ax.text(0, -0.03, f"Q: {wrap(g['question'], 2)}\nGT: {wrap(g['answer'], 2)}\nOurs: {wrap(p['text'], 3)}\n"
            f"T = {ver[i]}, S = {S[i]:.2f}, GROVE = {max(ver[i], .01) ** .5 * max(S[i], .01) ** .5:.2f}",
            transform=ax.transAxes, va="top", fontsize=5.8, family="DejaVu Sans")))
fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.235, wspace=0.08, hspace=0.98)
fig.canvas.draw()
for ax, g, t in texts:  # the comparison box goes 5 pt under the panel's text, whatever its length
    y = ax.transAxes.inverted().transform(t.get_window_extent())[0][1]
    ax.add_artist(AnnotationBbox(compare(g), (0, y), xycoords=ax.transAxes, xybox=(2, -5), boxcoords="offset points",
                                 box_alignment=(0, 1), pad=0.35, annotation_clip=False,
                                 bboxprops=dict(boxstyle="round,pad=0.35", fc="#eef4fd", ec="#a9c4ee", lw=0.5)))
(ROOT / "report").mkdir(exist_ok=True)
fig.savefig(ROOT / "report" / "qual.pdf", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("->", ROOT / "report" / "qual.pdf")
