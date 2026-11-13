"""wsp.py: the weight-space programming loop on Qwen3-VL-8B (EXPERIMENTS.md, "WSP"). Runs on a shared PC next to pipeline.py.

    python wsp.py collect                # split A: hidden states where the answer starts + unedited answers -> out/acts_A.npz
    python wsp.py eval SPEC.json [...]   # split B answers under each macro spec -> out/eval_<name>.json (registered score)
    python wsp.py resolve SPEC.json      # macro spec with directions inlined -> out/edit_<name>.json, the pipeline cfg "edit"

A spec is {"name", "edits": [...]} in pipeline.apply_edit's format, written by the editor (Claude) from activations and weights,
with directions given as "file.npz:key" in out/ (normalized here). No gradients anywhere.
"""
import base64
import contextlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import pipeline as P

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE.parent / "data" / "wsp_practice", HERE / "out"  # on the PC: <ws>/wsp/wsp.py, <ws>/data/wsp_practice
PREFIX = '{"answer": "'
WRITERS = ("self_attn.o_proj", "mlp.down_proj")
UNSURE = re.compile(r"\b(cannot|can't|could not|unable to|not possible to) (be )?(determined?|answered?|tell|identif\w+)|not discernible", re.I)


def denies(answer, obj):
    """premise denial of the queried object, or "cannot be determined" (which the judge accepts on false-premise questions);
    "There is no object on top of the clock" about a present clock is an answer, not a denial."""
    o = r"(?:people|persons?)" if obj == "person" else re.escape(obj) + r"(?:e?s)?"
    return bool(re.search(rf"\bno (?:visible |such )?{o}\b|\b{o} (?:is|are) not (?:clearly )?(?:visible|present|shown|seen|"
                          rf"in (?:the|this) (?:image|picture|photo)|discernible|identifiable|depicted)|(?:does not|doesn't|do not) "
                          rf"(?:contain|show|include|have|depict|feature) (?:a |an |any )?{o}\b|\b{o} cannot be (?:seen|found|identified|located)",
                          answer, re.I) or UNSURE.search(answer))


enc = lambda a: base64.b64encode(np.asarray(a, np.float32).tobytes()).decode()


def load():
    vlm = P.VLM({**P.DEFAULTS, "vlm": "Qwen/Qwen3-VL-8B-Instruct", "max_memory": ["20GiB"]})
    return vlm, P.decoder_layers(vlm.model)


def rows(split):
    return [r for r in json.loads((DATA / "practice.json").read_text()) if r["split"] == split]


def inputs(vlm, image, question):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": P.PROMPTS["v0"].format(question=question)}]}]
    text = vlm.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + PREFIX
    return vlm.proc(text=[text], images=[image], return_tensors="pt").to(vlm.model.device)


def answer(vlm, image, question, obj, max_new=96):
    x = inputs(vlm, image, question)
    with torch.inference_mode():
        out = vlm.model.generate(**x, max_new_tokens=max_new, do_sample=False)
    raw = PREFIX + vlm.proc.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True)
    p = P.parse_output(raw)
    return {"answer": p["answer"], "deny": denies(p["answer"], obj), "noev": not p["evidence"], "raw": raw}


def image(r):
    return Image.open(DATA / f"{r['coco_id']}.jpg").convert("RGB")


def collect():
    vlm, _ = load()
    H, ans = [], []
    for k, r in enumerate(rows("A")):
        im = image(r)
        with torch.inference_mode():
            o = vlm.model(**inputs(vlm, im, r["question"]), output_hidden_states=True)
        H.append(torch.stack([h[0, -1] for h in o.hidden_states]).float().cpu().numpy().astype(np.float16))
        ans.append({**answer(vlm, im, r["question"], r["object"]), "absent": r["absent"], "question": r["question"], "coco_id": r["coco_id"]})
        if k % 50 == 0:
            print(k, ans[-1]["answer"][:70], flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "acts_A.npz", h=np.stack(H), absent=np.array([a["absent"] for a in ans]), deny=np.array([a["deny"] for a in ans]))
    (OUT / "ans_A.json").write_text(json.dumps(ans, indent=0))
    summarize("A-unedited", ans)


def resolve(spec):
    """direction references -> unit vectors as base64 float32 (pipeline.apply_edit's format)."""
    def vec(ref):
        f, k = ref.split(":")
        d = np.load(OUT / f)[k].astype(np.float32)
        return enc(d / np.linalg.norm(d))
    return [{k: vec(v) if k in ("d", "src", "dst") else v for k, v in e.items()} for e in spec["edits"]]


@contextlib.contextmanager
def edited(layers, edits):
    mods = []
    for e in edits:
        if e["kind"] == "steer":
            mods.append(layers[e["layer"]].mlp.down_proj)
        else:
            mods += [layers[L].get_submodule(m) for L in range(e["layers"][0], e["layers"][1] + 1) for m in WRITERS]
    saved = {id(m): (m, m.weight.detach().cpu().clone(), None if m.bias is None else m.bias.detach().clone()) for m in mods}
    P.apply_edit(layers, edits)
    try:
        yield
    finally:
        with torch.no_grad():
            for m, W, bias in saved.values():
                m.weight.copy_(W.to(m.weight.device))
                m.bias = None if bias is None else torch.nn.Parameter(bias, requires_grad=False)


def summarize(name, res):
    ab = [r for r in res if r["absent"]]; pr = [r for r in res if not r["absent"]]
    hit, fa = np.mean([r["deny"] for r in ab]), np.mean([r["deny"] for r in pr])
    s = {"name": name, "hit": round(hit, 3), "fa": round(fa, 3), "score": round(hit - 1.35 * fa, 3),
         "hit_noev": round(np.mean([r["deny"] and r["noev"] for r in ab]), 3), "n": len(res)}
    print(json.dumps(s), flush=True)
    return s


def evaluate(paths, split="B"):
    vlm, layers = load()
    base = {r["question"] + str(r["coco_id"]): r for r in json.loads((OUT / f"ans_{split}.json").read_text())} \
        if (OUT / f"ans_{split}.json").exists() else {}
    for p in paths:
        spec = json.loads(Path(p).read_text())
        t0, res = time.time(), []
        with edited(layers, resolve(spec)):
            for r in rows(split):
                a = answer(vlm, image(r), r["question"], r["object"])
                res.append({**a, "absent": r["absent"], "same": base.get(r["question"] + str(r["coco_id"]), {}).get("raw") == a["raw"]})
        s = summarize(spec["name"], res)
        s.update(changed=sum(not r["same"] for r in res), sec=round(time.time() - t0))
        tag = "" if split == "B" else "_" + split  # split A results never overwrite the feedback split's
        (OUT / f"eval_{spec['name']}{tag}.json").write_text(json.dumps({"summary": s, "spec": spec, "split": split, "res": res}, indent=0))
        if spec["name"] == "none":
            (OUT / f"ans_{split}.json").write_text(json.dumps([{**r, "question": q["question"], "coco_id": q["coco_id"]}
                                                                for r, q in zip(res, rows(split))], indent=0))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "collect":
        collect()
    elif cmd == "eval":
        evaluate([a for a in args if not a.startswith("--")], "A" if "--A" in args else "B")
    elif cmd == "rescore":  # saved answers -> summaries with the current denial detector (no forward passes)
        for p in args:
            e = json.loads(Path(p).read_text()); q = rows(e.get("split", "B"))
            for r, x in zip(e["res"], q):
                r["deny"] = denies(r["answer"], x["object"])
            s = summarize(e["spec"]["name"], e["res"]); s["changed"] = sum(not r["same"] for r in e["res"])
            print("   changed vs none:", s["changed"], flush=True)
            e["summary"] = s; Path(p).write_text(json.dumps(e, indent=0))
    elif cmd == "resolve":
        for p in args:
            spec = json.loads(Path(p).read_text())
            (OUT / f"edit_{spec['name']}.json").write_text(json.dumps(resolve(spec)))
