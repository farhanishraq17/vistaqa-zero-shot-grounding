"""VistaQA zero-shot pipeline: VLM (answer + per-instance boxes) -> SAM (box -> mask) -> official RLE format.

Runs inside a Kaggle kernel (T4 x2). Scripts/fleet.py prepends `CFG = {...}` to this file.
Only `image` meta and `question` are read from a sample; answers / masks / task_type are never touched.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULTS = dict(
    vlm="Qwen/Qwen3-VL-8B-Instruct",
    seg="facebook/sam2.1-hiera-large",
    data="hf:mozhina/VistaQA",          # or a local dir of <id>.json + images (same format)
    shard=0, num_shards=1, limit=0,
    max_pixels=1280 * 1280,
    max_new_tokens=2048,
    thinking=False,
    prompt="v0",
    out="/kaggle/working",
)

PROMPTS = {
    "v0": """You are given an image and a question about it.
1. Answer the question accurately and concisely, as one short complete sentence.
2. Localize the visual evidence: every object or region in the image that your answer is based on, one entry per object instance, each with a bounding box [x1, y1, x2, y2] in coordinates normalized to 0-1000.
- If you count objects, give one box per counted object; the number of boxes must equal your count.
- For questions about text, box the text region you read.
- If the object or entity the question asks about does not exist in the image, or the question cannot be answered from the image, say so in the answer and return an empty evidence list.

Question: {question}

Return only JSON: {{"answer": "...", "evidence": [{{"label": "...", "bbox_2d": [x1, y1, x2, y2]}}]}}""",
    "v1": """You are given an image and a question about it.
1. Answer the question accurately and concisely, as one short complete sentence.
2. Localize the visual evidence: only what directly shows the answer, one entry per object instance, each with a bounding box [x1, y1, x2, y2] in coordinates normalized to 0-1000.
- Box the answer element itself, not what you read to find it: the bar or slice rather than its label or the legend, the table cell holding the value rather than its row and column headers, the shape rather than its side lengths.
- If the question compares or relates objects, box each of them.
- If you count objects, give one box per counted object; the number of boxes must equal your count.
- For questions about text, box the text region you read.
- If the object or entity the question asks about does not exist in the image, or the question cannot be answered from the image, say so in the answer and return an empty evidence list.

Question: {question}

Return only JSON: {{"answer": "...", "evidence": [{{"label": "...", "bbox_2d": [x1, y1, x2, y2]}}]}}""",
    "reason_v1": """Look at the image and answer the question accurately and concisely, as one short complete sentence.
Then list the visual evidence your answer is based on: a short referring expression for each distinct object (or group of identical objects) in the image that supports the answer, specific enough to find it in the image.
- For counting, give one expression that covers exactly the counted objects.
- For questions about text, refer to the text region you read.
- If the object or entity the question asks about does not exist in the image, or the question cannot be answered from the image, say so in the answer (e.g. "There is no ... in the image.") and give an empty list.

Question: {question}

Return only JSON: {{"answer": "...", "evidence": ["...", "..."]}}""",
}
GROUND = 'Locate every object that matches the description "{target}" in the image and output their bbox coordinates in JSON format.'
EXPAND = """Question: {question}
Short answer: {answer}
Rewrite the short answer as one complete sentence that restates what the question asks about. Do not add or change any information. Return only the sentence."""
VERIFY = "Is there {target} in the image? Answer yes or no."
PREMISE_VERIFY = {  # POPE-style confirm of the premise check's "missing" thing ("no" keeps the override)
    "v1": "Is there any {miss} in the image? Answer yes or no.",
    "v2": "Look carefully at the whole image, including small, distant or partially visible objects. "
          "Is there any {miss} in the image? Answer yes or no.",
}
ZOOM = """This is a zoomed-in crop of the image region relevant to the question.
Question: {question}
Read any text in the crop exactly as written (keep its spelling and case), then answer the question in one short complete sentence. Return only the sentence."""
READ_Q = re.compile(r"\b(text|written|writing|says?|read|words?|letters?|spell\w*)\b", re.I)
ROUTE = ("Is this image a chart, plot, table, diagram or geometric figure, rather than a photograph or a rendering of a real scene? "
         "Answer yes or no.")  # figure router: which questions take prompt v1's evidence boxes
REASON = ("Question about this image: {question}\nDo not answer it. Does answering it need several steps of calculation or "
          "reasoning (arithmetic, algebra, geometry, physics, a logic puzzle), rather than reading a value, label or count directly "
          "from the image? Answer yes or no.")  # reasoning router (cfg route: "reason"): which figure questions take a thinking answer
POT = ("Question about this image: {question}\nWrite a short Python program that computes the answer. First read the values you "
       "need from the image and assign them to variables, with a comment saying where each comes from; then compute; then print "
       "the final answer exactly as it should be given: a number (with its unit if the question asks for one) or the text of "
       "the correct option. Use only arithmetic and the math module. Reply with the program in a ```python block.")  # cfg pot

# model-written code runs in a separate interpreter: no imports but math, no _names, no files, 2 GB, 10 s
_RUNNER = r"""
import math, sys
try:
    import resource; resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
except Exception:
    pass
ok = {k: getattr(__builtins__, k) for k in ("abs", "round", "min", "max", "sum", "len", "range", "int", "float", "str", "sorted",
      "print", "pow", "enumerate", "zip", "list", "tuple", "dict", "set", "bool", "divmod", "map", "reversed", "any", "all")}
ok["__import__"] = lambda name, *a, **k: math if name == "math" else None
exec(compile(sys.stdin.read(), "<pot>", "exec"), {"__builtins__": ok, "math": math})
"""


def run_program(text, timeout=10):
    """VLM reply with a ```python block -> last printed line (<= 200 chars), or None if unsafe, failing, silent or slow."""
    import ast
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    src = m.group(1) if m else text
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for n in ast.walk(tree):
        if (isinstance(n, (ast.Import, ast.ImportFrom)) and any(a.name != "math" for a in n.names)) or \
                isinstance(n, ast.ImportFrom) or isinstance(n, (ast.ClassDef, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith,
                                                                ast.AsyncFunctionDef, ast.Await)) or \
                (isinstance(n, ast.Attribute) and n.attr.startswith("_")) or (isinstance(n, ast.Name) and n.id.startswith("_")):
            return None
    try:
        r = subprocess.run([sys.executable, "-I", "-c", _RUNNER], input=src, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    return lines[-1][:200] if r.returncode == 0 and lines else None


SELECT = ("Question about this image: {question}\nAnswer A: {a}\nAnswer B: {b}\nLook at the image carefully. Which answer is "
          "correct? If both or neither are, pick the more accurate one. Reply with the letter A or B only.")  # answer selector (cfg pairs)
MARKS = ("Question about this image: {question}\nAnswer: {answer}\nThe image shows {n} numbered boxes. Which numbered boxes mark "
         "visual evidence required to support this answer? Reply with a JSON list of box numbers only.")  # mask check (cfg marks)

PREMISE = """Question about this image: {question}
Do not answer the question. Check its premise: does the question assume an object, person, text or detail that is not actually visible in the image? Look carefully at the whole image.
If something it assumes is missing, name it in a few words; otherwise answer none.
Return only JSON: {{"missing": "..."}}"""
ABSENT = re.compile(r"\b(there (is|are) no|no \w+ (is|are) (visible|present)|not (present|visible)|does not (exist|appear)|"
                    r"cannot be (determined|answered)|isn't any|aren't any)\b", re.I)
NUMWORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()
_NUM = re.compile(r"\b(\d+|" + "|".join(NUMWORDS) + r")\b", re.I)
COUNT_Q = re.compile(r"\b(how many|count|number of)\b", re.I)
# counting with arithmetic / conditions / statistics ("Subtract all X. How many are left?", "mode of the number of")
_REASON_Q = re.compile(r"\b(subtract\w*|left|remain\w*|more|less|fewer|than|at least|at most|most|least|after|before|"
                       r"born|values?|sum|difference|average|mean|median|mode|range|ratio|percent|exceed\w*|between|"
                       r"per|each|fit)\b|%", re.I)
_BOX = re.compile(r"\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]")


# ---------------- pure helpers (tested in selftest) ----------------

def parse_output(text):
    """Pull {"answer", "evidence"} out of raw VLM text; tolerant of fences, think blocks, bare box lists."""
    text = re.sub(r"(?s)<think>.*?</think>", "", text).strip()
    text = re.sub(r"```(?:json)?", "", text)
    obj = None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[m.start():])
            if isinstance(obj, dict) and "answer" in obj:
                break
            obj = None
        except json.JSONDecodeError:
            continue
    if obj is None:  # malformed JSON (e.g. repeated keys in one object): salvage answer + every box
        ans = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        boxes, labels = all_boxes(text), re.findall(r'"label"\s*:\s*"([^"]*)"', text)
        labels = labels if len(labels) == len(boxes) else [""] * len(boxes)
        return {"answer": ans.group(1) if ans else text.strip()[:500],
                "evidence": [{"label": l, "bbox_2d": b} for l, b in zip(labels, boxes)], "parse_ok": False}
    ev = []
    for e in obj.get("evidence") or []:
        b = e.get("bbox_2d") if isinstance(e, dict) else e
        if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            ev.append({"label": str(e.get("label", "")) if isinstance(e, dict) else "", "bbox_2d": [float(v) for v in b]})
    return {"answer": str(obj.get("answer", "")).strip(), "evidence": ev, "parse_ok": True}


def all_boxes(text):
    """Every [x1, y1, x2, y2] number quad in the text (grounding output is a JSON list, often fenced/malformed)."""
    text = re.sub(r"(?s)<think>.*?</think>", "", text)
    return [[float(v) for v in b] for b in _BOX.findall(text)]


def parse_reason(text):
    """reason_v1 output -> {"answer", "targets": [str], "parse_ok"}."""
    text = re.sub(r"(?s)<think>.*?</think>", "", text)
    text = re.sub(r"```(?:json)?", "", text)
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "answer" in obj:
            ev = obj.get("evidence") or []
            ts = [str(e.get("label", "") if isinstance(e, dict) else e).strip() for e in ev] if isinstance(ev, list) else []
            return {"answer": str(obj["answer"]).strip(), "targets": [t for t in ts if t][:8], "parse_ok": True}
    ans = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    return {"answer": ans.group(1) if ans else text.strip()[:500], "targets": [], "parse_ok": False}


def parse_missing(text):
    """PREMISE output -> the missing thing without its article, or None when the premise holds."""
    m = re.search(r'"missing"\s*:\s*"([^"]*)"', text)
    v = re.sub(r"^(a|an|the|any)\s+", "", (m.group(1) if m else "").strip().rstrip("."), flags=re.I)
    return None if v.lower() in ("", "none", "nothing", "n/a", "null") else v


def answer_count(text):
    m = _NUM.search(text)
    return None if not m else int(m.group(1)) if m.group(1).isdigit() else NUMWORDS.index(m.group(1).lower())


def plain_count(question):
    """A plain instance count (one short sentence, no arithmetic/condition): the only case where more visible
    instances means a larger answer, so the only case merge_count may raise (MathVista count probe)."""
    q = question.strip()
    return (bool(COUNT_Q.search(q)) and not _REASON_Q.search(q) and len(q.split()) <= 12
            and not re.search(r"[.?!]\s+\S", q))


def merge_count(answer, masks, others):
    """Counting fix: other counters (confident SAM3 concept instances, a second VLM) may only raise the VLM's count
    (every counter here under-counts); the largest wins. others = [(count, masks)], earlier wins ties.
    -> (answer with the number replaced, masks)."""
    m, v = _NUM.search(answer), answer_count(answer)
    n, best = max(others, key=lambda o: o[0], default=(0, masks))
    if m is None or n <= v:
        return answer, masks
    head, tail = answer[:m.start()], answer[m.end():]
    if v == 1:  # the judge rejects "There is 2 chair": fix the verb and pluralise the counted noun
        head = re.sub(r"\bis(\s+(?:only\s+)?)$", r"are\1", head, flags=re.I)
        tail = re.sub(r"^(\s+)([A-Za-z]+?)(?<!s)\b", r"\1\2s", tail)
    return head + str(n) + tail, best


def dedup_masks(masks, iou=0.5):
    """COCO RLE masks in the VLM's evidence order; drop one whose IoU with an already kept mask is >= iou (near duplicates)."""
    from pycocotools import mask as mu
    enc = [{"size": m["size"], "counts": m["counts"].encode()} for m in masks]
    keep = []
    for k, m in enumerate(enc):
        if not keep or mu.iou([m], [enc[j] for j in keep], [0] * len(keep)).max() < iou:
            keep.append(k)
    return [masks[k] for k in keep]


def clean_masks(masks, mode):
    """Round R7: per COCO RLE mask, "fill" closes interior holes, "lcc" keeps the largest 8-connected component, "both"
    does both. An empty mask stays as it is."""
    import numpy as np
    from pycocotools import mask as mu
    from scipy import ndimage as nd
    out = []
    for m in masks:
        a = mu.decode({"size": m["size"], "counts": m["counts"].encode()}).astype(bool)
        if a.any() and mode in ("lcc", "both"):
            lab, n = nd.label(a, structure=np.ones((3, 3)))
            a = lab == 1 + np.bincount(lab.ravel())[1:].argmax() if n > 1 else a
        if mode in ("fill", "both"):
            a = nd.binary_fill_holes(a)
        r = mu.encode(np.asfortranarray(a.astype(np.uint8)))
        out.append({"size": [int(x) for x in r["size"]], "counts": r["counts"].decode()})
    return out


def prefer_b(raw):
    """Selector raw {"ab": reply with (A, B) shown, "ba": reply with them swapped}: B only if picked in both orders."""
    return raw["ab"].strip().upper().startswith("B") and raw["ba"].strip().upper().startswith("A")


def draw_marks(image, boxes):
    """Set-of-Mark: a copy of the image with each 0-1000 box outlined and numbered 1..n (mask check, cfg marks)."""
    from PIL import ImageDraw, ImageFont
    im, (w, h) = image.copy(), image.size
    d, lw, fs = ImageDraw.Draw(im), max(2, round(max(w, h) / 320)), max(14, round(max(w, h) / 36))
    try:
        font = ImageFont.load_default(size=fs)
    except TypeError:  # Pillow < 10.1: fixed small bitmap font
        font = ImageFont.load_default()
    colors = ["red", "lime", "yellow", "cyan", "magenta", "orange", "deepskyblue", "white"]
    for k, b in enumerate(boxes):
        x1, y1, x2, y2, c = b[0] * w / 1000, b[1] * h / 1000, b[2] * w / 1000, b[3] * h / 1000, colors[k % len(colors)]
        d.rectangle([x1, y1, x2, y2], outline=c, width=lw)
        l, t, r, bt = d.textbbox((x1 + lw, y1 + lw), str(k + 1), font=font)
        d.rectangle([l - 2, t - 2, r + 2, bt + 2], fill=c)
        d.text((x1 + lw, y1 + lw), str(k + 1), fill="black", font=font)
    return im


def parse_keep(text, n):
    """Mask check reply -> sorted 0-based indices of the boxes it lists; None (keep all) if no list, empty or no valid number."""
    m = re.search(r"\[([^\[\]]*)\]", text)
    return sorted({int(x) - 1 for x in re.findall(r"\d+", m.group(1)) if 1 <= int(x) <= n}) or None if m else None


def zoom_crop(image, boxes, margin=0.25, side=1024, max_scale=4.0):
    """Union of 0-1000 boxes + margin, cropped and upscaled so its long side reaches `side` (<= max_scale x)."""
    w, h = image.size
    px = [b for b in (to_pixels(b, w, h) for b in boxes) if b]
    if not px:
        return None
    x1, y1 = min(b[0] for b in px), min(b[1] for b in px)
    x2, y2 = max(b[2] for b in px), max(b[3] for b in px)
    mx, my = max(16, margin * (x2 - x1)), max(16, margin * (y2 - y1))
    x1, y1, x2, y2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    s = min(max_scale, side / max(x2 - x1, y2 - y1))
    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
    return crop.resize((max(28, int(crop.width * s)), max(28, int(crop.height * s)))) if s > 1 else crop


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def to_pixels(b, w, h):
    """0-1000 normalized box -> clipped pixel box; None if degenerate."""
    x1, y1, x2, y2 = b
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, x2 = np.clip([x1 * w / 1000, x2 * w / 1000], 0, w)
    y1, y2 = np.clip([y1 * h / 1000, y2 * h / 1000], 0, h)
    return [float(x1), float(y1), float(x2), float(y2)] if x2 - x1 >= 1 and y2 - y1 >= 1 else None


def rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode()}


def selftest():
    p = parse_output('<think>x</think>```json\n{"answer": "Two cats.", "evidence": [{"label": "cat", "bbox_2d": [0, 0, 500, 500]}, {"bbox_2d": [1,2,3]}]}\n```')
    assert p["answer"] == "Two cats." and len(p["evidence"]) == 1 and p["parse_ok"]
    assert parse_output("no json here")["parse_ok"] is False
    bad = parse_output('{"answer": "Two.", "evidence": [{"label": "a", "bbox_2d": [1, 2, 3, 4], "label": "b", "bbox_2d": [5, 6, 7, 8], "lab')
    assert bad["answer"] == "Two." and [e["bbox_2d"] for e in bad["evidence"]] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert parse_output('{"answer": "No dog.", "evidence": []}')["evidence"] == []
    r = parse_reason('```json\n{"answer": "There are 3 cups.", "evidence": ["cups on the table", ""]}\n```')
    assert r["answer"] == "There are 3 cups." and r["targets"] == ["cups on the table"] and r["parse_ok"]
    assert parse_reason('{"answer": "There is no cat in the image.", "evidence": []}')["targets"] == []
    assert parse_missing('```json\n{"missing": "The chair."}\n```') == "chair" and parse_missing('{"missing": "none"}') is None
    assert parse_missing("garbage") is None
    xyz = (3, ["x", "y", "z"])
    assert merge_count("There are two cats.", ["a", "b"], [xyz]) == ("There are 3 cats.", ["x", "y", "z"])
    assert merge_count("There are 3 cats.", ["a"], [(2, ["x", "y"]), (3, ["p", "q", "r"])]) == ("There are 3 cats.", ["a"])
    assert merge_count("There are 2 cats.", ["a"], [(3, ["x"]), (4, ["p"]), (4, ["q"])]) == ("There are 4 cats.", ["p"])
    assert merge_count("There is no cat.", [], [(1, ["x"])]) == ("There is no cat.", [])
    assert merge_count("There is 1 chair in the image.", ["a"], [(2, [])])[0] == "There are 2 chairs in the image."
    assert merge_count("There is one bus.", ["a"], [(2, [])])[0] == "There are 2 bus."  # already ends in s
    assert merge_count("I can see 2 people.", ["a"], [xyz])[0] == "I can see 3 people." and answer_count("no") is None
    assert COUNT_Q.search("How many dogs?") and not COUNT_Q.search("What color is the dog?")
    assert plain_count("Count the bananas in the picture.") and plain_count("How many Triangles do you see?")
    assert not plain_count("Subtract all red things. How many objects are left?")
    assert not plain_count("How many bars have values smaller than 6?") and not plain_count("What color is it?")
    from PIL import Image
    z = zoom_crop(Image.new("RGB", (640, 480)), [[500, 500, 600, 550]])  # 64x24 px box + 16 px margins -> 96x56, x4
    assert z.size == (384, 224) and zoom_crop(Image.new("RGB", (640, 480)), []) is None
    assert parse_keep("[2, 3]", 3) == [1, 2] and parse_keep("```json\n[1, 1]\n```", 2) == [0]
    assert parse_keep("[]", 3) is None and parse_keep("[7]", 3) is None and parse_keep("Boxes 1 and 2.", 3) is None
    mk = draw_marks(Image.new("RGB", (200, 100)), [[0, 0, 500, 500], [500, 500, 1000, 1000]])  # boxes [0,0,100,50], [100,50,200,100]
    assert mk.size == (200, 100) and mk.getpixel((0, 40)) == (255, 0, 0) and mk.getpixel((150, 30)) == (0, 0, 0)
    assert READ_Q.search("What does the sign say?") and not READ_Q.search("What color is the bus?")
    assert all_boxes('```json\n[{"bbox_2d": [1, 2, 30, 40], "label": "a"}, {"bbox_2d": [5,6,7,8]}]```') == [[1, 2, 30, 40], [5, 6, 7, 8]]
    assert parse_output('{"answer": "2", "evidence": [{"label": "a", "bbox_2d": [1, 2, 3, 4]], {"label": "b", "bbox_2d": [5, 6, 7, 8]]}')["evidence"][1]["label"] == "b"
    assert abs(box_iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9 and box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0
    assert ABSENT.search("There is no dog in the image.") and not ABSENT.search("The dog is to the left of the cat.")
    assert to_pixels([0, 0, 500, 1000], 200, 100) == [0.0, 0.0, 100.0, 100.0]
    assert to_pixels([10, 10, 10, 500], 200, 100) is None
    m = np.zeros((7, 9), bool); m[2:5, 3:8] = True
    from pycocotools import mask as mu
    r = rle(m)
    assert r["size"] == [7, 9] and (mu.decode({"size": r["size"], "counts": r["counts"].encode()}) == m).all()
    m2 = m.copy(); m2[2:5, 7] = False  # IoU 12/15 with m: a near duplicate
    m3 = np.zeros((7, 9), bool); m3[5:7, 0:3] = True
    assert dedup_masks([r, rle(m2), rle(m3)]) == [r, rle(m3)] and dedup_masks([]) == []
    ring = np.zeros((7, 9), bool); ring[1:6, 1:6] = True; ring[3, 3] = False; ring[0, 8] = True  # hole + stray pixel
    dec = lambda x: mu.decode({"size": x["size"], "counts": x["counts"].encode()}).astype(bool)
    f, l, b = (clean_masks([rle(ring)], k)[0] for k in ("fill", "lcc", "both"))
    assert dec(f).sum() == 26 and dec(l).sum() == 24 and dec(b).sum() == 25 and not dec(b)[0, 8] and dec(b)[3, 3]
    assert clean_masks([rle(np.zeros((7, 9), bool))], "both")[0] == rle(np.zeros((7, 9), bool))
    assert prefer_b({"ab": "B", "ba": "A."}) and not prefer_b({"ab": "B", "ba": "B"}) and not prefer_b({"ab": "A", "ba": "B"})
    assert run_program("```python\nimport math\nr = 2  # radius\nprint(round(math.pi * r ** 2, 2))\n```") == "12.57"
    assert run_program("```python\nv = []\nv.append(3)\nprint(f'{sum(v) * 2} cm')\n```") == "6 cm"
    for bad in ("import os\nprint(os.getcwd())", "print(open('x').read())", "print(().__class__)", "while True: pass",
                "from math import pi\nprint(pi)", "x = 1", "print(1/0)"):
        assert run_program(bad, timeout=3) is None, bad
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:  # WSP edits: amplify direction e1 x3 in layer 1's writers (centered at 0.5), steer layer 0 along e2
        import base64
        enc = lambda a: base64.b64encode(np.asarray(a, np.float32).tobytes()).decode()
        lay = torch.nn.ModuleList()
        for _ in range(2):
            L = torch.nn.Module(); L.self_attn, L.mlp = torch.nn.Module(), torch.nn.Module()
            L.self_attn.o_proj, L.mlp.down_proj = torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(3, 2, bias=False)
            torch.nn.init.ones_(L.self_attn.o_proj.weight); torch.nn.init.ones_(L.mlp.down_proj.weight); lay.append(L)
        apply_edit(lay, [{"kind": "amplify", "layers": [1, 1], "src": enc([1, 0]), "beta": 2.0, "center": 0.5},
                         {"kind": "steer", "layer": 0, "d": enc([0, 1]), "alpha": 3.0}])
        assert lay[1].mlp.down_proj.weight.tolist() == [[3, 3, 3], [1, 1, 1]] and lay[1].self_attn.o_proj.weight.tolist() == [[3, 3], [1, 1]]
        assert lay[1].mlp.down_proj.bias.tolist() == [-1, 0] and lay[0].mlp.down_proj(torch.zeros(1, 3)).tolist() == [[0, 3]]
        assert lay[0].self_attn.o_proj.weight.tolist() == [[1, 1], [1, 1]] and lay[0].self_attn.o_proj.bias is None
    print("selftest ok")


# ---------------- data ----------------

def list_samples(cfg):
    """[(image_id, json_path_or_hf_name_or_sample, image_name_or_url)] sorted by id, sharded."""
    if cfg["data"] == "inline":  # fleet.py --inline: [{"image", "question", "url"}], no GT fields
        items = [(s["image"]["image_id"], s, s["url"]) for s in cfg["samples"]]
    elif cfg["data"].startswith("hf:"):
        from huggingface_hub import list_repo_files
        files = list_repo_files(cfg["data"][3:], repo_type="dataset")
        imgs = {Path(f).stem: f for f in files if f.startswith("VistaQA/") and f.lower().endswith((".jpg", ".png", ".jpeg"))}
        items = [(int(Path(f).stem), f, imgs.get(Path(f).stem)) for f in files if f.startswith("VistaQA/") and f.endswith(".json")]
    else:
        d = Path(cfg["data"])
        if not d.exists():  # Kaggle dataset mount path varies; find the folder by name
            d = next(p for p in Path("/kaggle/input").rglob(d.name) if p.is_dir())
        imgs = {p.stem: p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")}
        items = [(int(p.stem), p, imgs.get(p.stem)) for p in d.glob("*.json") if p.stem.isdigit()]
    keep = set(cfg["ids"]) if cfg.get("ids") else None  # subset ablations (e.g. only OCR-like questions)
    items = [it for it in sorted(items) if it[0] >= cfg.get("ids_from", 0) and (keep is None or it[0] in keep)]
    items = items[cfg["shard"]::cfg["num_shards"]]
    return items[: cfg["limit"]] if cfg["limit"] else items


def _parquet_image(repo, file, row):
    """Image bytes from an HF dataset parquet ("hfpq:repo|file|row" urls: MathVista / TextVQA probes)."""
    import functools
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    if not hasattr(_parquet_image, "t"):
        _parquet_image.t = functools.lru_cache(4)(lambda r, f: pq.read_table(hf_hub_download(r, f, repo_type="dataset")))
    t = _parquet_image.t(repo, file)
    cell = t.column("decoded_image" if "decoded_image" in t.column_names else "image")[int(row)].as_py()
    return cell["bytes"]


def load_sample(cfg, item):
    from PIL import Image
    _, js, im = item
    if cfg["data"] == "inline":
        import io
        import urllib.request
        data = _parquet_image(*im[5:].split("|")) if im.startswith("hfpq:") else urllib.request.urlopen(im, timeout=60).read()
        return {"image": js["image"], "question": js["question"]}, Image.open(io.BytesIO(data)).convert("RGB")
    if cfg["data"].startswith("hf:"):
        from huggingface_hub import hf_hub_download
        js = hf_hub_download(cfg["data"][3:], js, repo_type="dataset")
        im = hf_hub_download(cfg["data"][3:], im, repo_type="dataset")
    rec = json.loads(Path(js).read_text(encoding="utf-8"))
    sample = {"image": {k: rec["image"][k] for k in ("image_id", "width", "height")}, "question": rec["question"]}
    del rec  # GT fields never leave this function
    return sample, Image.open(im).convert("RGB")


# ---------------- models ----------------

class VLM:
    def __init__(self, cfg):
        import torch
        import transformers as T
        self.cfg, self.torch = cfg, torch
        self.proc = T.AutoProcessor.from_pretrained(cfg["vlm"])
        # leave ~4 GiB on the last GPU for SAM + activations (quantized 27B+ needs nearly all of it -> seg_device "cpu")
        n = torch.cuda.device_count()
        cap = cfg.get("max_memory") or ["13GiB"] * (n - 1) + ["9GiB"]
        kw = dict(dtype=torch.float16, device_map="auto", max_memory=dict(enumerate(cap[:n])) or None)
        if cfg.get("quant") == "nf4":  # 27B+ on T4x2; needs cfg pip: ["bitsandbytes"]
            kw["quantization_config"] = T.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                             bnb_4bit_compute_dtype=torch.float16)
        try:
            self.model = T.AutoModelForImageTextToText.from_pretrained(cfg["vlm"], **kw)
        except ValueError:
            self.model = T.AutoModelForMultimodalLM.from_pretrained(cfg["vlm"], **kw)
        self.model.eval()
        if cfg.get("edit"):  # weight-space programming test (WSP): fixed edits written by the editor, applied once at load
            apply_edit(decoder_layers(self.model), cfg["edit"])

    def __call__(self, image, text, max_new_tokens=None, thinking=None):
        w, h = image.size
        s = min(1.0, (self.cfg["max_pixels"] / (w * h)) ** 0.5)
        img = image.resize((max(28, int(w * s)), max(28, int(h * s)))) if s < 1 else image
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}]
        think = self.cfg["thinking"] if thinking is None else thinking
        kw = {} if "Qwen3-VL" in self.cfg["vlm"] else {"enable_thinking": think}
        inputs = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", **kw).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.cfg["max_new_tokens"], do_sample=False)
        return self.proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def decoder_layers(model):
    return next(m for n, m in model.named_modules() if n.endswith("language_model.layers"))


def apply_edit(layers, edits):
    """WSP edits on decoder layers (no gradients: the editor chose every direction and scale; vectors are base64 float32):
    steer   {"layer", "d", "alpha"}: down_proj bias += alpha d (a zero bias is added first; the 8B has none)
    amplify {"layers": [a, b], "src", "beta", "center"}: every writer W (o_proj, down_proj) of layers a..b: W += beta src (src^T W)
    wire    same with "dst": W += beta dst (src^T W);  ablate: W -= gamma src (src^T W);  "center" m adds -beta m dst to b's bias."""
    import base64
    import torch
    vec = lambda s: torch.from_numpy(np.frombuffer(base64.b64decode(s), dtype=np.float32).copy())

    def add_bias(lin, b):
        if lin.bias is None:
            lin.bias = torch.nn.Parameter(torch.zeros_like(lin.weight[:, 0]), requires_grad=False)
        lin.bias += b.to(lin.weight.device, lin.weight.dtype)

    with torch.no_grad():
        for e in edits:
            if e["kind"] == "steer":
                add_bias(layers[e["layer"]].mlp.down_proj, e["alpha"] * vec(e["d"]))
                continue
            (a, b), src = e["layers"], vec(e["src"])
            dst, k = vec(e.get("dst", e["src"])), -e.get("gamma", 1.0) if e["kind"] == "ablate" else e["beta"]
            for L in range(a, b + 1):
                for m in ("self_attn.o_proj", "mlp.down_proj"):
                    W = layers[L].get_submodule(m).weight
                    W += (k * torch.outer(dst.to(W.device), src.to(W.device) @ W.float())).to(W.dtype)
            if e.get("center"):
                add_bias(layers[b].mlp.down_proj, -k * e["center"] * dst)


def resolve_weights(name):
    """"kaggle:owner/model/framework/variation" (Kaggle Models mirror) -> local dir holding config.json; HF ids pass through."""
    if not name.startswith("kaggle:"):
        return name
    import kagglehub
    d = kagglehub.model_download(name[len("kaggle:"):])
    return next(r for r, _, fs in os.walk(d) if "config.json" in fs)


class BoxSegmenter:
    """SAM2.1 or SAM3 (tracker head = SAM2-style box prompts on the SAM3 backbone): one mask per box."""
    def __init__(self, cfg):
        import torch
        import transformers as T
        self.torch = torch
        self.dev = cfg.get("seg_device") or (f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.is_available() else "cpu")
        M, P = (T.Sam3TrackerModel, T.Sam3TrackerProcessor) if "sam3" in cfg["seg"].lower() else (T.Sam2Model, T.Sam2Processor)
        src = resolve_weights(cfg["seg"])
        self.proc = P.from_pretrained(src)
        self.model = M.from_pretrained(src).to(self.dev).eval()
        self.multimask = cfg.get("multimask", False)  # 3 candidates per box, keep the best predicted IoU

    def __call__(self, image, boxes, chunk=8):
        out_masks = []
        for i in range(0, len(boxes), chunk):  # many-box samples (counting) OOM the decoder in one batch
            inputs = self.proc(images=image, input_boxes=[boxes[i:i + chunk]], return_tensors="pt").to(self.dev)
            with self.torch.inference_mode():
                out = self.model(**inputs, multimask_output=self.multimask)
            masks = self.proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]  # (n, k, H, W)
            best = out.iou_scores[0].argmax(-1).tolist() if self.multimask else [0] * len(masks)
            out_masks += [m[k].numpy().astype(bool) for m, k in zip(masks, best)]
        return out_masks


class ConceptSegmenter:
    """SAM3 text prompt -> every matching instance as [(mask, score)] (counting ablation, fleet --concepts)."""
    def __init__(self, cfg):
        import torch
        import transformers as T
        self.torch, self.dev = torch, cfg.get("seg_device") or "cuda:0"
        src = resolve_weights(cfg["seg"])
        self.proc = T.Sam3Processor.from_pretrained(src)
        self.model = T.Sam3Model.from_pretrained(src).to(self.dev).eval()

    def __call__(self, image, phrase, threshold=0.3):
        inputs = self.proc(images=image, text=phrase, return_tensors="pt").to(self.dev)
        with self.torch.inference_mode():
            out = self.model(**inputs)
        r = self.proc.post_process_instance_segmentation(out, threshold=threshold, mask_threshold=0.5,
                                                         target_sizes=inputs["original_sizes"].tolist())[0]
        return [(m.cpu().numpy().astype(bool), float(sc)) for m, sc in zip(r["masks"], r["scores"])]


# ---------------- per-sample logic ----------------

def predict(vlm, image, question, cfg):
    # premise check (false-premise questions are 27% of VistaQA): "only" = run just this pass, for offline combination
    if cfg.get("route"):  # router only: the yes/no answer, merged offline (fleet.py remask --route)
        r = vlm(image, REASON.format(question=question) if cfg["route"] == "reason" else ROUTE, max_new_tokens=4, thinking=False).strip()
        return {"answer": r, "boxes": [], "parse_ok": True, "raw": {"route": r}}
    if cfg.get("premise"):
        raw = {"premise": vlm(image, PREMISE.format(question=question), max_new_tokens=64, thinking=False)}
        miss = parse_missing(raw["premise"])
        if miss:  # second opinion, POPE-style; "confirm" keeps the miss only if this says no
            raw["verify"] = vlm(image, PREMISE_VERIFY[cfg.get("verify_prompt", "v1")].format(miss=miss), max_new_tokens=8, thinking=False)
            if cfg["premise"] == "confirm" and not raw["verify"].strip().lower().startswith("no"):
                miss = None
        if miss or cfg["premise"] == "only":
            return {"answer": f"There is no {miss} in the image." if miss else "", "boxes": [], "parse_ok": True, "raw": raw}
    if cfg.get("pot"):  # calculator tool: the VLM writes a program, run_program runs it; failure -> "" (merged offline, fleet think)
        code = vlm(image, POT.format(question=question), max_new_tokens=cfg.get("pot_tokens", 1024), thinking=False)
        out = run_program(code)
        p = {"answer": out or "", "boxes": [], "parse_ok": bool(out), "raw": {"code": code, "out": out, "ok": bool(out)}}
    else:
        p = _predict(vlm, image, question, cfg)
    # small text: re-read on an upscaled crop around the evidence boxes
    if cfg.get("zoom") and READ_Q.search(question) and p["boxes"] and not ABSENT.search(p["answer"]):
        crop = zoom_crop(image, p["boxes"])
        if crop is not None:
            p["first_answer"] = p["answer"]
            p["answer"] = vlm(crop, ZOOM.format(question=question), max_new_tokens=128, thinking=False).strip()
    # the official judge rejects "incomplete" answers ("right", "bed", bare "STOP"): restate terse answers as a sentence
    if cfg.get("expand_short") and 0 < len(p["answer"].split()) <= 3:
        p["short_answer"] = p["answer"]
        p["answer"] = vlm(image, EXPAND.format(question=question, answer=p["answer"]), max_new_tokens=64, thinking=False).strip()
    return p


def _predict(vlm, image, question, cfg):
    """-> {"answer", "boxes" (0-1000), "parse_ok", "raw"}. mode "single" = prompt v0; "two_stage" = reason, then ground."""
    if cfg.get("mode", "single") == "single":
        raw = vlm(image, PROMPTS[cfg["prompt"]].format(question=question))
        p = parse_output(raw)
        return {"answer": p["answer"], "boxes": [e["bbox_2d"] for e in p["evidence"]], "parse_ok": p["parse_ok"], "raw": raw}
    raw = {"reason": vlm(image, PROMPTS["reason_v1"].format(question=question))}
    r = parse_reason(raw["reason"])
    answer, targets = r["answer"], r["targets"]
    if ABSENT.search(answer):
        targets = []
    if targets and cfg.get("verify"):  # POPE-style existence check per target
        raw["verify"] = [vlm(image, VERIFY.format(target=t), max_new_tokens=8, thinking=False) for t in targets]
        keep = [t for t, v in zip(targets, raw["verify"]) if not v.strip().lower().startswith("no")]
        if not keep:
            answer = f"There is no {targets[0]} in the image."
        targets = keep
    boxes, raw["ground"] = [], []
    for t in targets:
        g = vlm(image, GROUND.format(target=t), max_new_tokens=cfg["max_new_tokens"], thinking=False)
        raw["ground"].append(g)
        boxes += [b for b in all_boxes(g) if all(box_iou(b, o) < 0.8 for o in boxes)]
    return {"answer": answer, "boxes": boxes, "parse_ok": r["parse_ok"], "raw": raw}


# ---------------- main loop ----------------

def run(cfg):
    out_dir = Path(cfg["out"]); out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"pred_shard{cfg['shard']}.jsonl"
    raw_path = out_dir / f"raw_shard{cfg['shard']}.jsonl"
    done = {json.loads(l)["image"]["image_id"] for l in pred_path.open()} if pred_path.exists() else set()
    only = cfg.get("concepts", cfg.get("pairs", cfg.get("marks")))  # concept / selector / mask-check passes touch only their own ids
    items = [it for it in list_samples(cfg) if it[0] not in done and (only is None or it[0] in only)]
    print(f"shard {cfg['shard']}/{cfg['num_shards']}: {len(items)} to do, {len(done)} done", flush=True)
    # segmenter ablations: pixel boxes (fleet --boxes) or SAM3 text phrases (fleet --concepts) of an earlier run
    vlm = None if "boxes" in cfg or "concepts" in cfg else VLM(cfg)
    # seg "" = VLM pass only (boxes kept in raw; masks come from a separate --boxes pass, e.g. SAM3 next to a 27B)
    seg = ConceptSegmenter(cfg) if "concepts" in cfg else BoxSegmenter(cfg) if cfg["seg"] else None
    t0 = time.time()
    with pred_path.open("a") as fp, raw_path.open("a") as fr:
        for n, item in enumerate(items, 1):
            ts = time.time()
            sample, image = load_sample(cfg, item)
            if "concepts" in cfg and cfg.get("count_only", True) and not COUNT_Q.search(sample["question"]):
                continue  # evidence-label concepts only serve the counting fix (premise-veto concepts: count_only False)
            if cfg.get("plain_only") and not plain_count(sample["question"]):
                continue  # a second VLM's answers only feed countfix, which only touches plain counting Qs
            w, h = image.size
            import torch
            try:
                if "concepts" in cfg:
                    found = seg(image, cfg["concepts"][item[0]])
                    p, raw, boxes = {"answer": "", "parse_ok": True}, {"scores": [sc for _, sc in found]}, []
                    masks = [rle(m) for m, _ in found]
                elif "pairs" in cfg:  # answer selector: two candidate answers shown in both orders (fleet.py pick)
                    ask = lambda x, y: vlm(image, SELECT.format(question=sample["question"], a=x, b=y), max_new_tokens=2, thinking=False)
                    ca, cb = cfg["pairs"][item[0]]
                    p, raw, boxes, masks = {"answer": "", "parse_ok": True}, {"ab": ask(ca, cb), "ba": ask(cb, ca)}, [], []
                elif "marks" in cfg:  # mask check (fleet.py verify): an earlier run's masks as numbered boxes, keep the listed ones
                    mk = cfg["marks"][item[0]]
                    reply = vlm(draw_marks(image, mk["boxes"]), MARKS.format(question=sample["question"], answer=mk["answer"],
                                n=len(mk["boxes"])), max_new_tokens=96, thinking=False)
                    p, raw, boxes, masks = {"answer": "", "parse_ok": True}, {"marks": reply, "keep": parse_keep(reply, len(mk["boxes"]))}, [], []
                else:
                    if vlm is None:
                        p, raw, boxes = {"answer": "", "parse_ok": True}, "", cfg["boxes"].get(item[0], [])
                    else:
                        p = predict(vlm, image, sample["question"], cfg)
                        raw, boxes = p["raw"], [b for b in (to_pixels(bb, w, h) for bb in p["boxes"]) if b]
                    masks = [rle(m) for m in seg(image, boxes) if m.any()] if seg else []
            except torch.OutOfMemoryError:  # one bad sample must not kill the shard
                torch.cuda.empty_cache()
                raw, p, boxes, masks = "OOM", {"answer": "", "parse_ok": False}, [], []
            torch.cuda.empty_cache()
            fp.write(json.dumps({"image": sample["image"], "question": sample["question"],
                                 "prediction": {"text": p["answer"], "masks": masks}}) + "\n"); fp.flush()
            fr.write(json.dumps({"image_id": item[0], "raw": raw, "parse_ok": p.get("parse_ok", False), "boxes": boxes,
                                 "sec": round(time.time() - ts, 2)}) + "\n"); fr.flush()
            if n % 10 == 0 or n == len(items):
                print(f"{n}/{len(items)} {(time.time() - t0) / n:.1f}s/it", flush=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); sys.exit()
    cfg = {**DEFAULTS, **globals().get("CFG", {})}
    print("CFG", cfg, flush=True)
    for pkg in cfg.get("pip", []):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
    selftest()
    run(cfg)
