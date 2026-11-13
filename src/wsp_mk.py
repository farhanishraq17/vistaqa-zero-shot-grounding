"""wsp_mk.py ROUND: write that round's candidate edit specs (out/specs_<round>/*.json) with scales set from split A's activations.

Scales are in units of the absent-present gap along the edit's direction, so a candidate reads "push by half a gap" etc.
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
z, D = np.load(OUT / "acts_A.npz"), np.load(OUT / "dirs.npz")
h, ab = z["h"].astype(np.float32), z["absent"].astype(bool)


def gap(L, key=None):
    p = h[:, L] @ D[key or f"abs{L}"]
    return float(p[ab].mean() - p[~ab].mean())


def writes(src, a, b):
    """src-projection of what layers a..b write (h[b+1] - h[a]): class gap and the midpoint between the class means."""
    c = (h[:, b + 1] - h[:, a]) @ D[src]
    return float(c[ab].mean() - c[~ab].mean()), float((c[ab].mean() + c[~ab].mean()) / 2)


def steer(name, L, key, k):  # bias on layer L's down_proj: the residual entering L + 1 moves k gaps along key
    return {"name": name, "edits": [{"kind": "steer", "layer": L, "d": f"dirs.npz:{key}", "alpha": k * gap(L + 1, key)}]}


def cond(name, kind, src, a, b, k, dst=None, at=None):
    """amplify / wire with centering: the injected component's absent-present gap is k gaps of dst at layer `at`."""
    g, m = writes(src, a, b)
    beta = k * gap(at, dst) / g if kind == "wire" else k
    e = {"kind": kind, "layers": [a, b], "src": f"dirs.npz:{src}", "beta": beta, "center": m}
    if dst:
        e["dst"] = f"dirs.npz:{dst}"
    return {"name": name, "edits": [e]}


ROUNDS = {
    "r1": lambda: [
        {"name": "none", "edits": []},
        steer("s21-0.5", 20, "abs21", 0.5), steer("s21-1", 20, "abs21", 1.0), steer("s21-1.5", 20, "abs21", 1.5),
        cond("w18to22-0.5", "wire", "abs18", 14, 17, 0.5, dst="abs22", at=22),
        cond("w18to22-1", "wire", "abs18", 14, 17, 1.0, dst="abs22", at=22),
        cond("a18-1", "amplify", "abs18", 14, 17, 1.0),
        cond("a21-1", "amplify", "abs21", 18, 20, 1.0),
    ],
    # r1 (object-aware detector): steer 0.5 best (0.762) but rewrites 25 present answers; amplify 18-20 0.746 with FA unchanged
    "r2": lambda: [
        cond("a21-2", "amplify", "abs21", 18, 20, 2.0), cond("a21-3", "amplify", "abs21", 18, 20, 3.0),
        cond("a22-1", "amplify", "abs22", 18, 21, 1.0), cond("a22-2", "amplify", "abs22", 18, 21, 2.0),
        both("a18-1+a21-1", cond("", "amplify", "abs18", 14, 17, 1.0), cond("", "amplify", "abs21", 18, 20, 1.0)),
        both("a18-1+a21-2", cond("", "amplify", "abs18", 14, 17, 1.0), cond("", "amplify", "abs21", 18, 20, 2.0)),
        steer("s21-0.3", 20, "abs21", 0.3),
        both("s21-0.3+a21-1", steer("", 20, "abs21", 0.3), cond("", "amplify", "abs21", 18, 20, 1.0)),
    ],
    # r2: amplifiers level off at hit .80-.81 with FA .03-.04; steer 0.3 + amplify 18-20 best (.85 / .05 / .782)
    "r3": lambda: [
        both("s21-0.2+a21-1", steer("", 20, "abs21", 0.2), cond("", "amplify", "abs21", 18, 20, 1.0)),
        both("s21-0.4+a21-1", steer("", 20, "abs21", 0.4), cond("", "amplify", "abs21", 18, 20, 1.0)),
        both("s21-0.5+a21-1", steer("", 20, "abs21", 0.5), cond("", "amplify", "abs21", 18, 20, 1.0)),
        both("s21-0.3+a21-2", steer("", 20, "abs21", 0.3), cond("", "amplify", "abs21", 18, 20, 2.0)),
        both("s21-0.3+a18-1+a21-1", steer("", 20, "abs21", 0.3), cond("", "amplify", "abs18", 14, 17, 1.0),
             cond("", "amplify", "abs21", 18, 20, 1.0)),
        both("s21-0.3+a22-2", steer("", 20, "abs21", 0.3), cond("", "amplify", "abs22", 18, 21, 2.0)),
    ],
}


def both(name, *specs):
    return {"name": name, "edits": [e for s in specs for e in s["edits"]]}

if __name__ == "__main__":
    r = sys.argv[1]
    d = OUT / f"specs_{r}"; d.mkdir(exist_ok=True)
    for L in (18, 21, 22):
        print(f"gap{L} {gap(L):.2f}", end="  ")
    print("| writes 14-17 on abs18 (gap, mid):", [round(x, 2) for x in writes("abs18", 14, 17)],
          "| 18-20 on abs21:", [round(x, 2) for x in writes("abs21", 18, 20)])
    for s in ROUNDS[r]():
        (d / f"{s['name']}.json").write_text(json.dumps(s))
        print(s["name"], [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()} for e in s["edits"]])
