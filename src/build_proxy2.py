"""Build proxy2: external multi-domain dev set (AV, robotics, science, math) in VistaQA JSON format. Deterministic (seed 0).

    python src/build_proxy2.py             # -> Competition Data/proxy2/<id>.json + image
    python src/build_proxy2.py --selftest
    python src/build_proxy2.py gtbox|gtmask  # AV: boxes out to SAM3, masks back into the JSONs

Proxy dev (COCO only) did not warn about false alarms outside COCO, over-masking or the count raise. VistaQA's own image
sources (SA-1B, NuScenes, DROID, ScienceQA; paper Sec. 3) are avoided on purpose. Fixed recipe, built once: it is never
tuned to look like VistaQA results. Answers and masks come from source annotations or the drawing code, never from a model.
Only public statistics inform the mix (six task types, ~27% hallucination).
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Competition Data" / "external" / "proxy2_src"
OUT = ROOT / "Competition Data" / "proxy2"
HALLUC = 0.27
TASKS = ["identification", "attribute", "ocr", "spatial", "counting", "comparison"]


def rle(mask):
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode()}


# ---------------- math: every element's mask is drawn from the same coordinates as the figure ----------------

COLORS = {"red": (220, 40, 40), "blue": (40, 90, 220), "green": (40, 160, 70), "orange": (245, 140, 30),
          "purple": (140, 60, 180), "yellow": (240, 210, 40), "pink": (240, 130, 180), "brown": (130, 85, 50)}
CATS = "Apples Bananas Cherries Dates Figs Grapes Kiwis Lemons Mangoes Oranges Pears Plums".split()
ABSENT_CATS = "Walnuts Peaches Melons Papayas".split()
DAYS = "Monday Tuesday Wednesday Thursday Friday".split()


class Fig:
    def __init__(self, w=800, h=600):
        self.w, self.h = w, h
        self.im = Image.new("RGB", (w, h), "white")
        self.d = ImageDraw.Draw(self.im)

    def shape(self, kind, xy, **style):
        """Draw and return the filled element's mask."""
        getattr(self.d, kind)(xy, **style)
        m = Image.new("L", (self.w, self.h), 0)
        extra = {k: v for k, v in style.items() if k in ("start", "end")}
        getattr(ImageDraw.Draw(m), kind)(xy, fill=255, **extra)
        return np.array(m) > 0

    def text(self, xy, s, size=20, fill="black", anchor="mm"):
        f = ImageFont.load_default(size)
        self.d.text(xy, s, fill=fill, font=f, anchor=anchor)
        x1, y1, x2, y2 = self.d.textbbox(xy, s, font=f, anchor=anchor)
        m = np.zeros((self.h, self.w), bool)
        m[max(0, int(y1) - 2):int(y2) + 3, max(0, int(x1) - 2):int(x2) + 3] = True
        return m


def absent(q):
    return (q[0], q[1], [])


def bar_chart(rng, task, halluc):
    n = rng.randint(4, 7)
    cats, vals = rng.sample(CATS, n), rng.sample(range(2, 21), n)
    colors = rng.sample(list(COLORS), n)
    f = Fig()
    x0, x1, y0, y1 = 80, 770, 520, 50
    for v in range(0, 25, 5):
        y = y0 - (y0 - y1) * v / 22
        f.d.line([(x0, y), (x1, y)], fill=(215, 215, 215))
        f.text((x0 - 8, y), str(v), 16, anchor="rm")
    f.d.line([(x0, y1), (x0, y0), (x1, y0)], fill="black", width=2)
    step, bars, labels = (x1 - x0) / n, {}, {}
    for i, (c, v, col) in enumerate(zip(cats, vals, colors)):
        bx0, bx1, by = x0 + step * i + step * 0.15, x0 + step * (i + 1) - step * 0.15, y0 - (y0 - y1) * v / 22
        bars[c] = f.shape("rectangle", [bx0, by, bx1, y0], fill=COLORS[col], outline="black")
        labels[c] = f.text(((bx0 + bx1) / 2, by - 12), str(v), 18)
        f.text(((bx0 + bx1) / 2, y0 + 16), c, 16)
    a, b = rng.sample(range(n), 2)
    ca, cb = cats[a], cats[b]
    if halluc:
        z = rng.choice(ABSENT_CATS)
        q = {"attribute": f"What color is the bar for {z}?", "ocr": f"What number is printed above the bar for {z}?",
             "spatial": f"Which bar is immediately to the right of the bar for {z}?", "counting": f"How many bars for {z} are in the chart?",
             "comparison": f"Is the bar for {z} taller than the bar for {ca}?", "identification": f"What value does the bar for {z} show?"}[task]
        return f, q, f"There is no bar for {z} in the chart.", []
    if task == "identification":
        hi = rng.random() < 0.5
        c = cats[(max if hi else min)(range(n), key=vals.__getitem__)]
        return f, f"Which category has the {'highest' if hi else 'lowest'} value in the bar chart?", f"{c} has the {'highest' if hi else 'lowest'} value.", [bars[c]]
    if task == "attribute":
        return f, f"What color is the bar for {ca}?", f"The bar for {ca} is {colors[a]}.", [bars[ca]]
    if task == "ocr":
        return f, f"What number is printed above the bar for {ca}?", f"The number printed above the bar for {ca} is {vals[a]}.", [labels[ca]]
    if task == "spatial":
        i = rng.randrange(n - 1)
        return f, f"Which category's bar is immediately to the right of the bar for {cats[i]}?", f"The bar for {cats[i + 1]} is immediately to the right of the bar for {cats[i]}.", [bars[cats[i + 1]]]
    if task == "counting":
        t = sorted(vals)[rng.randint(0, n - 2)]
        hit = [c for c, v in zip(cats, vals) if v > t]
        return f, f"How many bars have a value greater than {t}?", f"There are {len(hit)} bars with a value greater than {t}.", [bars[c] for c in hit]
    hi, lo = (ca, cb) if vals[a] > vals[b] else (cb, ca)
    return f, f"How much greater is the value of {hi} than the value of {lo}?", f"The value of {hi} is {abs(vals[a] - vals[b])} greater than the value of {lo}.", [bars[hi], bars[lo]]


def _polygon(cx, cy, r, sides):
    start = -math.pi / 2
    return [(cx + r * math.cos(start + 2 * math.pi * i / sides), cy + r * math.sin(start + 2 * math.pi * i / sides)) for i in range(sides)]


SHAPES = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6, "circle": 0}


def _draw_shape(f, kind, cx, cy, r, col):
    if kind == "circle":
        return f.shape("ellipse", [cx - r, cy - r, cx + r, cy + r], fill=COLORS[col], outline="black")
    if kind == "square":
        return f.shape("rectangle", [cx - r * .8, cy - r * .8, cx + r * .8, cy + r * .8], fill=COLORS[col], outline="black")
    return f.shape("polygon", _polygon(cx, cy, r, SHAPES[kind]), fill=COLORS[col], outline="black")


def shapes_scene(rng, task, halluc):
    cols, rows = 5, 4
    cells = rng.sample([(c, r) for c in range(cols) for r in range(rows)], rng.randint(7, 14))
    palette = rng.sample(list(COLORS)[:6], rng.randint(3, 5))
    f, items = Fig(), []
    for c, r in cells:
        kind, col = rng.choice(list(SHAPES)[:rng.choice([3, 5])]), rng.choice(palette)
        rad = rng.randint(30, 62)
        cx, cy = 80 + c * 160 + rng.randint(-8, 8), 75 + r * 150 + rng.randint(-8, 8)
        items.append(dict(kind=kind, col=col, r=rad, c=c, row=r, cx=cx, cy=cy, mask=_draw_shape(f, kind, cx, cy, rad, col)))
    kinds = sorted({i["kind"] for i in items})
    if halluc:
        miss_kind = next(k for k in rng.sample(list(SHAPES), len(SHAPES)) if k not in kinds) if len(kinds) < 5 else None
        miss_col = next(k for k in rng.sample(list(COLORS), len(COLORS)) if k not in palette)
        any_kind = miss_kind or rng.choice(kinds)
        obj = f"{miss_col} {any_kind}"
        q = {"attribute": f"How large is the {obj} compared with the others?", "ocr": f"What number is written inside the {obj}?",
             "spatial": f"What shape is directly above the {obj}?", "counting": f"How many {obj}s are in the figure?",
             "comparison": f"Is the {obj} bigger than the other shapes?", "identification": f"Which shape is next to the {obj}?"}[task]
        return f, q, f"There is no {obj} in the figure.", []
    by = lambda **kw: [i for i in items if all(i[k] == v for k, v in kw.items())]
    if task == "attribute":
        for k in rng.sample(kinds, len(kinds)):
            v = sorted(by(kind=k), key=lambda i: -i["r"])
            if len(v) >= 2 and v[0]["r"] - v[1]["r"] >= 8:
                return f, f"What color is the largest {k}?", f"The largest {k} is {v[0]['col']}.", [v[0]["mask"]]
        task = "counting"
    if task == "identification":
        solo = [c for c in palette if len(by(col=c)) == 1]
        if solo:
            c = rng.choice(solo)
            return f, f"What shape is the only {c} object in the figure?", f"The only {c} object is a {by(col=c)[0]['kind']}.", [by(col=c)[0]["mask"]]
        task = "counting"
    if task == "spatial":
        pairs = [(a, b) for a in items for b in items if a["c"] == b["c"] and a["row"] + 1 == b["row"]
                 and len(by(kind=b["kind"], col=b["col"])) == 1]
        if pairs:
            a, b = rng.choice(pairs)
            return f, f"What shape is directly above the {b['col']} {b['kind']}?", f"A {a['col']} {a['kind']} is directly above the {b['col']} {b['kind']}.", [a["mask"], b["mask"]]
        task = "counting"
    if task == "comparison":
        ks = sorted(kinds, key=lambda k: -len(by(kind=k)))
        if len(ks) >= 2 and len(by(kind=ks[0])) != len(by(kind=ks[1])):
            a, b = rng.sample(ks[:2], 2)
            more = ks[0]
            return f, f"Are there more {a}s or {b}s in the figure?", f"There are more {more}s than {ks[1]}s.", [i["mask"] for i in items if i["kind"] in (a, b)]
        task = "counting"
    if task == "ocr":
        i = rng.choice(items)
        lab = str(rng.randint(10, 99))
        return f, f"What number is written inside the {i['col']} {i['kind']} near column {i['c'] + 1}?", f"The number written inside it is {lab}.", [f.text((i["cx"], i["cy"]), lab, 22, fill="white")]
    k, c = rng.choice([(i["kind"], i["col"]) for i in items])
    v = by(kind=k, col=c)
    if rng.random() < 0.5:
        v, name = by(kind=k), f"{k}s"
    else:
        name = f"{c} {k}s"
    return f, f"How many {name} are in the figure?", f"There {'is' if len(v) == 1 else 'are'} {len(v)} {name if len(v) != 1 else name[:-1]}.", [i["mask"] for i in v]


def geometry(rng, task, halluc):
    f = Fig()
    a, b = rng.randint(3, 12), rng.randint(3, 12)
    while a == b:
        b = rng.randint(3, 12)
    labs = rng.sample("ABCDEFGHJKLMNPQRST", 4)
    x0, y0, s = 150, 120, 500 / max(a, b)
    x1, y1 = x0 + a * s, y0 + b * s * 0.7
    rect = f.shape("rectangle", [x0, y0, x1, y1], fill=(225, 235, 250), outline="black", width=3)
    pos = [(x0 - 18, y0 - 14), (x1 + 18, y0 - 14), (x1 + 18, y1 + 14), (x0 - 18, y1 + 14)]
    vm = {l: f.text(p, l, 24) for l, p in zip(labs, pos)}
    f.text(((x0 + x1) / 2, y0 - 16), str(a), 22)
    f.text((x1 + 30, (y0 + y1) / 2), str(b), 22)
    name = "".join(labs)
    if halluc:
        z = rng.choice([x for x in "UVWXYZ"])
        q = {"attribute": "What is the radius of the circle?", "ocr": f"What number is written next to vertex {z}?",
             "spatial": f"Which vertex is directly below vertex {z}?", "counting": "How many circles are drawn inside the rectangle?",
             "comparison": "Which has the larger area, the rectangle or the triangle?", "identification": "What is the area of the shaded triangle?"}[task]
        what = {"attribute": "circle", "counting": "circle", "comparison": "triangle", "identification": "triangle"}.get(task, f"vertex {z}")
        return f, q, f"There is no {what} in the figure.", []
    if task == "identification":
        return f, "What type of quadrilateral is drawn in the figure?", f"The figure shows a rectangle, {name}.", [rect]
    if task == "attribute":
        area = rng.random() < 0.5
        return f, f"What is the {'area' if area else 'perimeter'} of rectangle {name}?", f"The {'area' if area else 'perimeter'} of rectangle {name} is {a * b if area else 2 * (a + b)}.", [rect]
    if task == "ocr":
        i = rng.randrange(4)
        where = ["top-left", "top-right", "bottom-right", "bottom-left"][i]
        return f, f"What letter labels the {where} vertex of the rectangle?", f"The {where} vertex is labeled {labs[i]}.", [vm[labs[i]]]
    if task == "spatial":
        return f, f"Which vertex is directly below vertex {labs[0]}?", f"Vertex {labs[3]} is directly below vertex {labs[0]}.", [vm[labs[3]]]
    if task == "counting":
        return f, f"How many vertices of rectangle {name} are labeled in the figure?", f"There are 4 labeled vertices.", [vm[l] for l in labs]
    side = "longer" if a > b else "shorter"
    return f, "Is the horizontal side of the rectangle longer or shorter than its vertical side?", f"The horizontal side is {side} than the vertical side.", [rect]


def table(rng, task, halluc):
    f = Fig()
    rows = rng.sample(DAYS, rng.randint(3, 5))
    cols = rng.sample(["Math", "Art", "Music", "Sport"], rng.randint(3, 4))
    vals = [[rng.randint(10, 99) for _ in cols] for _ in rows]
    x0, y0, ch = 40, 60, 80
    cw = (f.w - 2 * x0) / (len(cols) + 1)
    cells = {}
    for j, c in enumerate(["", *cols]):
        f.text((x0 + cw * j + cw / 2, y0 + ch / 2), c, 22)
    for i, r in enumerate(rows):
        f.text((x0 + cw / 2, y0 + ch * (i + 1) + ch / 2), r, 22)
        for j, c in enumerate(cols):
            xy = [x0 + cw * (j + 1), y0 + ch * (i + 1), x0 + cw * (j + 2), y0 + ch * (i + 2)]
            m = f.shape("rectangle", xy, outline="black")
            f.text((xy[0] + cw / 2, xy[1] + ch / 2), str(vals[i][j]), 22)
            cells[r, c] = m
    for k in range(len(rows) + 2):
        f.d.line([(x0, y0 + ch * k), (x0 + cw * (len(cols) + 1), y0 + ch * k)], fill="black")
    for k in range(len(cols) + 2):
        f.d.line([(x0 + cw * k, y0), (x0 + cw * k, y0 + ch * (len(rows) + 1))], fill="black")
    r, c = rng.choice(rows), rng.choice(cols)
    v = lambda r_, c_: vals[rows.index(r_)][cols.index(c_)]
    if halluc:
        z = rng.choice(["Saturday", "Sunday"])
        q = {"attribute": f"What is the value for {z} under {c}?", "ocr": f"What number is in the {z} row of the {c} column?",
             "spatial": f"Which row is directly below the {z} row?", "counting": f"How many values in the {z} row are above 50?",
             "comparison": f"Is the {c} value for {z} greater than the one for {r}?", "identification": f"Which subject has the highest value on {z}?"}[task]
        return f, q, f"There is no {z} row in the table.", []
    if task in ("ocr", "attribute"):
        return f, f"What is the value in the {r} row of the {c} column?", f"The value in the {r} row of the {c} column is {v(r, c)}.", [cells[r, c]]
    if task == "identification":
        best = max(rows, key=lambda x: v(x, c))
        return f, f"Which day has the highest value in the {c} column?", f"{best} has the highest value in the {c} column.", [cells[best, c]]
    if task == "spatial":
        i = rng.randrange(len(rows) - 1)
        return f, f"What value is directly below the {rows[i]} value in the {c} column?", f"The value directly below it is {vals[i + 1][cols.index(c)]}.", [cells[rows[i + 1], c]]
    if task == "counting":
        t = sorted(vals[i][cols.index(c)] for i in range(len(rows)))[0]
        hit = [x for x in rows if v(x, c) > t]
        return f, f"How many values in the {c} column are greater than {t}?", f"There are {len(hit)} values greater than {t} in the {c} column.", [cells[x, c] for x in hit]
    r2 = rng.choice([x for x in rows if x != r and v(x, c) != v(r, c)] or [r])
    if r2 == r:
        return f, f"What is the value in the {r} row of the {c} column?", f"The value in the {r} row of the {c} column is {v(r, c)}.", [cells[r, c]]
    return f, f"Which is greater in the {c} column, the {r} value or the {r2} value?", f"The {max((r, r2), key=lambda x: v(x, c))} value is greater.", [cells[r, c], cells[r2, c]]


def coords(rng, task, halluc):
    f = Fig()
    x0, y0, s = 100, 540, 48
    for k in range(0, 11):
        f.d.line([(x0 + k * s, y0), (x0 + k * s, y0 - 10 * s)], fill=(225, 225, 225))
        f.d.line([(x0, y0 - k * s), (x0 + 10 * s, y0 - k * s)], fill=(225, 225, 225))
        f.text((x0 + k * s, y0 + 16), str(k), 15)
        f.text((x0 - 16, y0 - k * s), str(k), 15)
    f.d.line([(x0, y0 - 10 * s), (x0, y0), (x0 + 10 * s, y0)], fill="black", width=2)
    n = rng.randint(4, 7)
    labs = rng.sample("ABCDEFGHJK", n)
    xs, ys = rng.sample(range(1, 10), n), rng.sample(range(1, 10), n)
    pts = {}
    for l, x, y in zip(labs, xs, ys):
        cx, cy = x0 + x * s, y0 - y * s
        m = f.shape("ellipse", [cx - 7, cy - 7, cx + 7, cy + 7], fill=(30, 30, 200))
        pts[l] = m | f.text((cx + 16, cy - 16), l, 20)
    a, b = rng.sample(range(n), 2)
    if halluc:
        z = rng.choice("WXYZ")
        q = {"attribute": f"What color is point {z}?", "ocr": f"What are the coordinates of point {z}?", "spatial": f"Which point is directly to the left of point {z}?",
             "counting": f"How many points are above point {z}?", "comparison": f"Is point {z} higher than point {labs[a]}?", "identification": f"What is the x-coordinate of point {z}?"}[task]
        return f, q, f"There is no point {z} in the figure.", []
    if task in ("ocr", "identification"):
        return f, f"What are the coordinates of point {labs[a]}?", f"Point {labs[a]} is at ({xs[a]}, {ys[a]}).", [pts[labs[a]]]
    if task == "attribute":
        return f, f"What is the y-coordinate of point {labs[a]}?", f"The y-coordinate of point {labs[a]} is {ys[a]}.", [pts[labs[a]]]
    if task == "spatial":
        i = max(range(n), key=xs.__getitem__)
        return f, "Which point is farthest to the right?", f"Point {labs[i]} is farthest to the right.", [pts[labs[i]]]
    if task == "counting":
        t = sorted(xs)[rng.randint(0, n - 2)]
        hit = [l for l, x in zip(labs, xs) if x > t]
        return f, f"How many points have an x-coordinate greater than {t}?", f"There are {len(hit)} points with an x-coordinate greater than {t}.", [pts[l] for l in hit]
    hi = labs[a] if ys[a] > ys[b] else labs[b]
    return f, f"Which point is higher, {labs[a]} or {labs[b]}?", f"Point {hi} is higher.", [pts[labs[a]], pts[labs[b]]]


def pie(rng, task, halluc):
    f = Fig()
    n = rng.randint(3, 5)  # distinct shares of 20 parts: 1+2+..+6 > 20, so at most 5 slices
    parts = []
    while len(set(parts)) < n:
        cuts = sorted(rng.sample(range(1, 20), n - 1))
        parts = [b - a for a, b in zip([0, *cuts], [*cuts, 20])]
    pct = [p * 5 for p in parts]
    cats, colors = rng.sample(CATS, n), rng.sample(list(COLORS), n)
    box, start, wedges = [170, 70, 630, 530], -90.0, {}
    for c, p, col in zip(cats, pct, colors):
        end = start + 3.6 * p
        wedges[c] = f.shape("pieslice", box, start=start, end=end, fill=COLORS[col], outline="black")
        mid = math.radians((start + end) / 2)
        f.text((400 + 300 * math.cos(mid), 300 + 275 * math.sin(mid)), f"{c} {p}%", 18)
        start = end
    a, b = rng.sample(range(n), 2)
    if halluc:
        z = rng.choice(ABSENT_CATS)
        q = {"attribute": f"What color is the {z} slice?", "ocr": f"What percentage is shown for {z}?", "spatial": f"Which slice is next to the {z} slice?",
             "counting": f"How many slices are labeled {z}?", "comparison": f"Is the {z} slice larger than the {cats[a]} slice?", "identification": f"What share does {z} have?"}[task]
        return f, q, f"There is no {z} slice in the chart.", []
    if task == "identification":
        big = cats[max(range(n), key=pct.__getitem__)]
        return f, "Which category has the largest share of the pie chart?", f"{big} has the largest share.", [wedges[big]]
    if task == "attribute":
        return f, f"What color is the {cats[a]} slice?", f"The {cats[a]} slice is {colors[a]}.", [wedges[cats[a]]]
    if task == "ocr":
        return f, f"What percentage is shown for {cats[a]}?", f"{cats[a]} is shown as {pct[a]}%.", [wedges[cats[a]]]
    if task == "spatial":
        i = rng.randrange(n)
        j = (i + 1) % n
        return f, f"Going clockwise, which slice comes right after {cats[i]}?", f"{cats[j]} comes right after {cats[i]} going clockwise.", [wedges[cats[j]]]
    if task == "counting":
        t = sorted(pct)[rng.randint(0, n - 2)]
        hit = [c for c, p in zip(cats, pct) if p > t]
        return f, f"How many slices are larger than {t}%?", f"There are {len(hit)} slices larger than {t}%.", [wedges[c] for c in hit]
    hi = cats[a] if pct[a] > pct[b] else cats[b]
    return f, f"Which share is larger, {cats[a]} or {cats[b]}?", f"{hi} has the larger share.", [wedges[cats[a]], wedges[cats[b]]]


MATH = [bar_chart, shapes_scene, geometry, table, coords, pie]


def math_rows(rng, n):
    """Round-robin over figure kinds and tasks; about HALLUC of the samples ask about something the figure lacks."""
    rows = []
    for i in range(n):
        fn, task, h = MATH[i % len(MATH)], TASKS[(i // len(MATH) + i) % len(TASKS)], rng.random() < HALLUC
        f, q, a, masks = fn(rng, task, h)
        assert h or masks, (fn.__name__, task)
        rows.append(dict(domain="math", task=task, halluc=h, image=f.im, q=q, a=a, masks=masks, source={"kind": fn.__name__}, ext="png"))
    return rows


# ---------------- shared question templates for photos ----------------

PLURALS = {"pedestrian": "pedestrians", "bus": "buses", "traffic light": "traffic lights", "traffic sign": "traffic signs", "rider": "cyclists"}


def plural(n):
    return PLURALS.get(n, n + ("es" if n.endswith(("s", "x", "ch")) else "s"))


def article(n):
    return ("an " if n[0] in "aeiou" else "a ") + n


def absent_q(rng, x, ref):
    """A question that assumes x is in the image (POPE-style) -> (task, question, answer)."""
    t, q = rng.choice([("attribute", f"What color is the {x}?"), ("counting", f"How many {plural(x)} are there in the image?"),
                       ("spatial", f"Is the {x} to the left or to the right of the {ref}?"), ("ocr", f"What is written on the {x}?"),
                       ("comparison", f"Which is larger, the {x} or the {ref}?"), ("identification", f"What is the {x} next to?")])
    return t, q, f"There is no {x} in the image."


def take(rng, cands, want):
    """{task: n} from shuffled candidates, at most one sample per image across all tasks (hardest-to-fill tasks first)."""
    used, out = set(), []
    for task in sorted(want, key=lambda t: len(cands[t])):
        pool = list(cands[task]); rng.shuffle(pool)
        k = 0
        for c in pool:
            if k == want[task]:
                break
            if c["key"] not in used:
                used.add(c["key"]); out.append({**c, "task": c.get("task", task)}); k += 1
        print(f"  {task}: {k}/{want[task]} (pool {len(pool)})")
    return out


def box_mask(b, w, h):
    m = np.zeros((h, w), bool)
    m[int(b[1]):int(b[3]) + 1, int(b[0]):int(b[2]) + 1] = True
    return m


# ---------------- AV: BDD100K (dgural/bdd100k, 10k val images); masks = SAM3 on the annotated boxes, like VistaQA's AV masks ----------------

BDD_URL = "https://huggingface.co/datasets/dgural/bdd100k/resolve/main/"
AV_WANT = dict(counting=24, identification=24, spatial=24, comparison=20, attribute=18, halluc=40)
LIGHT = {"G": "green", "R": "red", "Y": "yellow"}


def av_cands(rng):
    from collections import defaultdict
    cands = defaultdict(list)
    samples = json.loads((SRC / "samples.json").read_text(encoding="utf-8"))["samples"]
    for s in sorted(samples, key=lambda x: x["filepath"]):
        W, H = s["metadata"]["width"], s["metadata"]["height"]
        dets = []
        for d in s["detections"]["detections"]:
            x, y, w, h = d["bounding_box"]
            box = [x * W, y * H, (x + w) * W, (y + h) * H]
            dets.append(dict(label=d["label"], box=box, area=w * h, color=d.get("trafficLightColor", "NA")))
        by = defaultdict(list)
        for d in dets:
            by[d["label"]].append(d)
        add = lambda task, q, a, ev: cands[task].append(dict(key=s["filepath"], url=BDD_URL + s["filepath"], W=W, H=H, q=q, a=a, boxes=[e["box"] for e in ev]))
        for c in ["pedestrian", "truck", "bus", "traffic light", "traffic sign", "bicycle", "motorcycle", "rider", "car"]:
            v = by[c]
            if 2 <= len(v) <= 9 and all(d["area"] >= 0.0015 for d in v):
                n = plural(c)
                add("counting", rng.choice([f"How many {n} are visible in this street scene?", f"Count the {n} in the image.", f"How many {n} can you see?"]), f"There are {len(v)} {n}.", v)
        heavy = by["bus"] + by["truck"] + by["train"]
        if len(heavy) == 1 and heavy[0]["area"] >= 0.01:
            add("identification", "What type of large vehicle is visible in the scene?", f"It is {article(heavy[0]['label'])}.", heavy)
        two = by["bicycle"] + by["motorcycle"]
        if len(two) == 1 and two[0]["area"] >= 0.003:
            add("identification", "What kind of two-wheeled vehicle is in the scene?", f"It is {article(two[0]['label'])}.", two)
        veh = sorted([d for c in ("car", "truck", "bus") for d in by[c] if d["area"] >= 0.003], key=lambda d: d["box"][0])
        if len(veh) >= 3 and len({d["label"] for d in veh}) >= 2 and veh[1]["box"][0] - veh[0]["box"][0] >= 0.05 * W:
            add("identification", "What type of vehicle is furthest to the left in the image?", f"The vehicle furthest to the left is {article(veh[0]['label'])}.", veh[:1])
        singles = {c: v[0] for c, v in by.items() if len(v) == 1 and v[0]["area"] >= 0.002
                   and c in ("bus", "truck", "pedestrian", "rider", "bicycle", "motorcycle", "traffic light", "traffic sign")}
        names = sorted(singles)
        for i, a in enumerate(names):
            da = singles[a]
            cx = (da["box"][0] + da["box"][2]) / 2 / W
            if cx < 0.4 or cx > 0.6:
                add("spatial", f"Is the {a} on the left or the right side of the image?", f"The {a} is on the {'left' if cx < 0.5 else 'right'} side of the image.", [da])
            for b in names[i + 1:]:
                db = singles[b]
                if da["box"][2] < db["box"][0] or db["box"][2] < da["box"][0]:
                    side = "left" if da["box"][2] < db["box"][0] else "right"
                    add("spatial", f"Is the {a} to the left or to the right of the {b}?", f"The {a} is to the {side} of the {b}.", [da, db])
                r = da["area"] / db["area"]
                if r >= 2.5 or r <= 0.4:
                    add("comparison", f"Which appears larger in the image, the {a} or the {b}?", f"The {a if r > 1 else b} appears larger.", [da, db])
        cs = [c for c in ("car", "pedestrian", "truck", "traffic light", "traffic sign") if 1 <= len(by[c]) <= 8 and all(d["area"] >= 0.0015 for d in by[c])]
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                if abs(len(by[a]) - len(by[b])) >= 2:
                    more, less = (a, b) if len(by[a]) > len(by[b]) else (b, a)
                    add("comparison", f"Are there more {plural(a)} or {plural(b)} in the scene?", f"There are more {plural(more)} than {plural(less)}.", by[a] + by[b])
        lights = by["traffic light"]
        if 1 <= len(lights) <= 4 and all(d["area"] >= 0.0006 for d in lights) and len({d["color"] for d in lights}) == 1 and lights[0]["color"] in LIGHT:
            col = LIGHT[lights[0]["color"]]
            add("attribute", "What color is the traffic light showing?" if len(lights) == 1 else "What color are the traffic lights showing?",
                f"The traffic light{'' if len(lights) == 1 else 's'} {'is' if len(lights) == 1 else 'are'} showing {col}.", lights)
        missing = [c for c in ("bus", "truck", "motorcycle", "bicycle", "train", "traffic light") if not by[c]]
        if missing and singles:
            t, q, a = absent_q(rng, rng.choice(missing), rng.choice(names))
            cands["halluc"].append(dict(key=s["filepath"], url=BDD_URL + s["filepath"], W=W, H=H, q=q, a=a, boxes=[], task=t))
    return cands


def av_rows(rng):
    """Masks are filled in by `gtmask` (SAM3 on the annotated boxes); `boxes` stays in the JSON until then."""
    import io
    import urllib.request
    rows = []
    for c in take(rng, av_cands(rng), AV_WANT):
        im = Image.open(io.BytesIO(urllib.request.urlopen(c["url"], timeout=60).read())).convert("RGB")
        rows.append(dict(domain="av", task=c["task"], halluc=not c["boxes"], image=im, q=c["q"], a=c["a"], masks=[], boxes=c["boxes"],
                         source={"bdd100k": c["key"]}, ext="jpg"))
    return rows


# ---------------- robotics: YCB-V + LM-O (BOP test sets, real visible masks), ARMBench totes ----------------

YCBV = dict(enumerate(["master chef can", "cracker box", "sugar box", "tomato soup can", "mustard bottle", "tuna fish can", "pudding box", "gelatin box",
                       "potted meat can", "banana", "pitcher", "bleach bottle", "bowl", "mug", "power drill", "wood block", "scissors", "marker",
                       "large clamp", "extra large clamp", "foam brick"], 1))
LMO = {1: "toy ape", 5: "watering can", 6: "toy cat", 8: "power drill", 9: "toy duck", 10: "egg carton", 11: "glue bottle", 12: "hole punch"}
CANS = {"master chef can", "tomato soup can", "tuna fish can", "potted meat can"}
BOXES = {"cracker box", "sugar box", "pudding box", "gelatin box"}
FIXED_COLOR = {"cracker box": "red", "mustard bottle": "yellow", "banana": "yellow", "mug": "red", "bowl": "red", "foam brick": "red", "pitcher": "blue"}
ROBOT_WANT = dict(counting=20, identification=24, spatial=24, comparison=20, attribute=22, halluc=40)


def bop_cands(rng, zip_name, names, complete):
    """complete: every object on the table is annotated (YCB-V), so counts and 'leftmost' are well defined; LM-O has unlabeled clutter."""
    import zipfile
    from collections import defaultdict
    cands = defaultdict(list)
    zf = zipfile.ZipFile(SRC / zip_name)
    for sg in sorted(n for n in zf.namelist() if n.endswith("scene_gt.json")):
        scene = sg.rsplit("/", 1)[0]
        gt, info = json.loads(zf.read(sg)), json.loads(zf.read(sg.replace("scene_gt.json", "scene_gt_info.json")))
        for img in sorted(gt, key=int):
            objs = [dict(name=names[g["obj_id"]], k=k, z=g["cam_t_m2c"][2], box=i["bbox_visib"], px=i["px_count_visib"], vis=i["visib_fract"])
                    for k, (g, i) in enumerate(zip(gt[img], info[img])) if g["obj_id"] in names]
            ok = [o for o in objs if o["vis"] >= 0.5 and o["px"] >= 500]
            add = lambda pool, q, a, ev, **kw: cands[pool].append(dict(key=(zip_name, scene, img), zip=zip_name, scene=scene, img=int(img), q=q, a=a, ev=[e["k"] for e in ev], **kw))
            if complete and objs and all(o["vis"] >= 0.3 for o in objs):
                add("counting", rng.choice(["How many objects are on the table?", "Count the objects on the table."]), f"There are {len(objs)} objects on the table.", objs)
                for group, word in ((CANS, "cans"), (BOXES, "boxes")):
                    v = [o for o in objs if o["name"] in group]
                    if len(v) >= 2:
                        add("counting", f"How many {word} are on the table?", f"There are {len(v)} {word} on the table.", v)
            for o in ok:
                if o["name"] in FIXED_COLOR and sum(x["name"] == o["name"] for x in objs) == 1:
                    add("attribute", f"What color is the {o['name']}?", f"The {o['name']} is {FIXED_COLOR[o['name']]}.", [o])
            if complete and len(ok) >= 3 and len(ok) == len(objs):
                left = sorted(ok, key=lambda o: o["box"][0])
                if left[1]["box"][0] - left[0]["box"][0] >= 0.05 * 640:
                    add("identification", "What object is furthest to the left in the image?", f"The object furthest to the left is the {left[0]['name']}.", left[:1])
                near = sorted(ok, key=lambda o: o["z"])
                if near[1]["z"] - near[0]["z"] >= 80:
                    add("identification", "Which object is closest to the camera?", f"The {near[0]['name']} is closest to the camera.", near[:1])
            for i, a in enumerate(ok):
                for b in ok[i + 1:]:
                    if a["name"] == b["name"] or "clamp" in a["name"] and "clamp" in b["name"]:
                        continue
                    if a["box"][0] + a["box"][2] < b["box"][0] or b["box"][0] + b["box"][2] < a["box"][0]:
                        side = "left" if a["box"][0] < b["box"][0] else "right"
                        add("spatial", f"Is the {a['name']} to the left or to the right of the {b['name']}?", f"The {a['name']} is to the {side} of the {b['name']}.", [a, b])
                    r = a["px"] / b["px"]
                    if r >= 2.5 or r <= 0.4:
                        add("comparison", f"Which appears larger in the image, the {a['name']} or the {b['name']}?", f"The {a['name'] if r > 1 else b['name']} appears larger.", [a, b])
            missing = sorted(set(names.values()) - {o["name"] for o in objs})
            if missing and ok:
                t, q, a = absent_q(rng, rng.choice(missing), rng.choice(ok)["name"])
                add("halluc", q, a, [], task=t)
    return cands


def robotics_rows(rng):
    import io
    import zipfile
    from collections import defaultdict
    ycb, lmo = bop_cands(rng, "ycbv_test_bop19.zip", YCBV, True), bop_cands(rng, "lmo_test_bop19.zip", LMO, False)
    cands = defaultdict(list)
    for t in ROBOT_WANT:
        cands[t] = ycb[t] + lmo[t]
    rows, zips = [], {}
    for c in take(rng, cands, ROBOT_WANT):
        zf = zips.setdefault(c["zip"], zipfile.ZipFile(SRC / c["zip"]))
        im = Image.open(io.BytesIO(zf.read(f"{c['scene']}/rgb/{c['img']:06d}.png"))).convert("RGB")
        masks = [np.array(Image.open(io.BytesIO(zf.read(f"{c['scene']}/mask_visib/{c['img']:06d}_{k:06d}.png")))) > 0 for k in c["ev"]]
        rows.append(dict(domain="robotics", task=c["task"], halluc=not c["ev"], image=im, q=c["q"], a=c["a"], masks=masks,
                         source={"bop": c["zip"], "scene": c["scene"], "img": c["img"]}, ext="jpg"))
    return rows


# ---------------- science: Data Science Bowl 2018 nuclei (real per-nucleus masks) ----------------

SCI_WANT = dict(counting=30, spatial=25, comparison=25, attribute=20, halluc=30)
SCI_ABSENT = ["scale bar", "text label", "red blood cell", "flagellum", "arrow", "ruler"]


def science_rows(rng):
    import io
    import zipfile
    from collections import defaultdict
    zf = zipfile.ZipFile(SRC / "stage1_train.zip")
    per = defaultdict(list)
    for n in zf.namelist():
        if "/masks/" in n and n.endswith(".png"):
            per[n.split("/")[0]].append(n)
    cands = defaultdict(list)
    for key in sorted(per):
        if not 2 <= len(per[key]) <= 30:
            continue
        ms = sorted(per[key])
        masks = [np.array(Image.open(io.BytesIO(zf.read(n)))) > 0 for n in ms]
        im = Image.open(io.BytesIO(zf.read(f"{key}/images/{key}.png"))).convert("RGB")
        W, H = im.size
        area = [int(m.sum()) for m in masks]
        if min(area) < 25:  # specks are not countable nuclei
            continue
        add = lambda pool, q, a, ev, **kw: cands[pool].append(dict(key=key, q=q, a=a, ev=ev, **kw))
        add("counting", rng.choice(["How many nuclei are visible in this microscope image?", "Count the cell nuclei in the image."]), f"There are {len(masks)} nuclei.", list(range(len(masks))))
        xs = [np.nonzero(m)[1].mean() / W for m in masks]
        order = sorted(range(len(masks)), key=lambda i: -area[i])
        big = order[0]
        if area[big] >= 1.5 * area[order[1]] and (xs[big] < 0.42 or xs[big] > 0.58):
            add("spatial", "Is the largest nucleus in the left half or the right half of the image?", f"The largest nucleus is in the {'left' if xs[big] < 0.5 else 'right'} half of the image.", [big])
        lh = sum(x < 0.5 for x in xs)
        if abs(2 * lh - len(xs)) >= 2 and len(xs) <= 20:
            add("comparison", "Which half of the image contains more nuclei, the left or the right?", f"The {'left' if 2 * lh > len(xs) else 'right'} half contains more nuclei.", list(range(len(masks))))
        arr = np.array(im.convert("L"), dtype=float)
        fg = arr[np.any(masks, axis=0)].mean()
        bg = arr[~np.any(masks, axis=0)].mean()
        if abs(fg - bg) >= 30 and len(masks) <= 15:
            add("attribute", "Are the nuclei brighter or darker than the background?", f"The nuclei are {'brighter' if fg > bg else 'darker'} than the background.", list(range(len(masks))))
        x = rng.choice(SCI_ABSENT)
        t, q = rng.choice([("attribute", f"What color is the {x}?"), ("ocr", f"What does the {x} say?"), ("counting", f"How many {plural(x)} are in the image?"), ("spatial", f"Where is the {x} located in the image?")])
        cands["halluc"].append(dict(key=key, q=q, a=f"There is no {x} in the image.", ev=[], task=t))
    rows = []
    for c in take(rng, cands, SCI_WANT):
        ms = sorted(per[c["key"]])
        im = Image.open(io.BytesIO(zf.read(f"{c['key']}/images/{c['key']}.png"))).convert("RGB")
        masks = [np.array(Image.open(io.BytesIO(zf.read(ms[k])))) > 0 for k in c["ev"]]
        rows.append(dict(domain="science", task=c["task"], halluc=not c["ev"], image=im, q=c["q"], a=c["a"], masks=masks, source={"dsb2018": c["key"]}, ext="png"))
    return rows


# ---------------- output ----------------

def write(rows, start=1):
    OUT.mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(rows, start):
        im = r["image"].convert("RGB")
        name = f"{k}.{r['ext']}"
        im.save(OUT / name, **({"quality": 95} if r["ext"] == "jpg" else {}))
        anns = [{"segmentation": rle(m)} for m in r["masks"]]
        (OUT / f"{k}.json").write_text(json.dumps({
            "image": {"image_id": k, "width": im.width, "height": im.height, "file_name": name}, "question": r["q"], "answer": r["a"],
            "task_type": r["task"], "task_domain": r["domain"], "num_instances": len(anns) or len(r.get("boxes", [])), "hallucination": int(r["halluc"]),
            "annotations": anns, "source": r["source"], **({"boxes": r["boxes"]} if r.get("boxes") else {})}), encoding="utf-8")
    return start + len(rows)


def selftest():
    rng = random.Random(1)
    rows = math_rows(rng, 240)
    assert all(m.any() for r in rows for m in r["masks"]) and all(not r["masks"] for r in rows if r["halluc"])
    assert all(r["q"].endswith("?") and r["a"].endswith(".") for r in rows)
    assert {r["task"] for r in rows} == set(TASKS)
    f = Fig(100, 100)
    m = f.shape("rectangle", [10, 10, 19, 19], fill="red")
    assert m.sum() == 100 and f.im.getpixel((15, 15))[0] > 200
    assert plural("bus") == "buses" and plural("mug") == "mugs" and article("egg carton") == "an egg carton"
    assert box_mask([1, 2, 3, 4], 6, 6).sum() == 9
    print("selftest ok", len(rows), "hallucination", sum(r["halluc"] for r in rows))


def build():
    """Fixed order and seed, so ids and samples never change. Only written once; rerunning reproduces the same set."""
    rows = math_rows(random.Random(10), 150)
    rows += robotics_rows(random.Random(20))
    rows += science_rows(random.Random(30))
    rows += av_rows(random.Random(40))
    write(rows)
    from collections import Counter
    print(Counter(r["domain"] for r in rows), Counter(r["task"] for r in rows), "hallucination", sum(r["halluc"] for r in rows), "total", len(rows))


def gtbox():
    """AV annotated boxes -> runs/p2-gtbox/raw.jsonl, for `fleet.py push p2-gtmask --boxes p2-gtbox` (SAM3 masks them)."""
    rs = [json.loads(p.read_text(encoding="utf-8")) for p in OUT.glob("*.json")]
    d = ROOT / "runs" / "p2-gtbox"; d.mkdir(parents=True, exist_ok=True)
    rows = [{"image_id": r["image"]["image_id"], "boxes": r["boxes"]} for r in rs if r.get("boxes")]
    (d / "raw.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    print(len(rows), "samples ids", sorted(r["image_id"] for r in rows)[:3], "...")


def gtmask():
    """Masks of p2-gtmask -> AV annotations (BDD100K has boxes only). These measure which objects were found, not mask quality."""
    preds = {r["image"]["image_id"]: r["prediction"]["masks"] for r in map(json.loads, (ROOT / "runs" / "p2-gtmask" / "preds.jsonl").read_text(encoding="utf-8").splitlines())}
    n = 0
    for p in OUT.glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("boxes"):
            assert preds.get(r["image"]["image_id"]), f"no masks for {p.name}"
            r["annotations"], r["num_instances"] = [{"segmentation": m} for m in preds[r["image"]["image_id"]]], len(preds[r["image"]["image_id"]])
            p.write_text(json.dumps(r), encoding="utf-8"); n += 1
    print("filled", n)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); sys.exit()
    {"gtbox": gtbox, "gtmask": gtmask}.get(sys.argv[1] if len(sys.argv) > 1 else "", build)()
