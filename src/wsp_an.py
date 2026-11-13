"""wsp_an.py: read the 8B's hidden states on practice split A (out/acts_A.npz from wsp.py collect) -> out/dirs.npz + a table.

Per layer: absent-minus-present direction (pairs share image and template), its cross-fitted pair accuracy and effect size;
among absent questions, the denied-minus-accepted direction and how well it separates them; logit lens of both directions.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def auc(pos, neg):
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def lens():
    """final RMSNorm weight and unembedding of the 8B, read from its safetensors (no model load)."""
    from safetensors import safe_open
    from transformers import AutoTokenizer
    snap = next((OUT.parent.parent / "hf" / "hub" / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots").iterdir())
    idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    get = lambda k: safe_open(str(snap / idx[k]), "pt").get_tensor(k).float().numpy()  # bf16 checkpoint
    U = get("lm_head.weight" if "lm_head.weight" in idx else "model.language_model.embed_tokens.weight")
    return get("model.language_model.norm.weight"), U, AutoTokenizer.from_pretrained(str(snap))


def main():
    z = np.load(OUT / "acts_A.npz")
    h, ab, dn = z["h"].astype(np.float32), z["absent"].astype(bool), z["deny"].astype(bool)
    n = len(ab); half = (np.arange(n) // 2) % 2 == 0  # pairs (2k, 2k+1) stay together
    print(f"{n} questions: absent denied {dn[ab].sum()}/{ab.sum()}, present denied {dn[~ab].sum()}/{(~ab).sum()}")
    dirs, rows = {}, []
    for L in range(h.shape[1] - 1):  # the last entry is after the final norm
        x = h[:, L]
        d = unit(x[ab].mean(0) - x[~ab].mean(0))
        acc = []
        for tr in (half, ~half):  # direction from one half of the pairs, tested on the other
            dd = unit(x[tr & ab].mean(0) - x[tr & ~ab].mean(0))
            p = x[~tr] @ dd
            acc.append(np.mean(p[ab[~tr]] > p[~ab[~tr]]))  # pair accuracy: absent above its own present twin
        p = x @ d
        eff = (p[ab].mean() - p[~ab].mean()) / np.sqrt((p[ab].var() + p[~ab].var()) / 2)
        a_ok, a_no = ab & dn, ab & ~dn
        dny = unit(x[a_ok].mean(0) - x[a_no].mean(0))
        cv = []
        for tr in (half, ~half):
            dd = unit(x[tr & a_ok].mean(0) - x[tr & a_no].mean(0))
            q = x @ dd
            cv.append(auc(q[a_ok & ~tr], q[a_no & ~tr]))
        dirs[f"abs{L}"], dirs[f"deny{L}"] = d, dny
        dirs[f"absmid{L}"] = np.float32((p[ab].mean() + p[~ab].mean()) / 2)  # midpoint of the two class means along d
        rows.append((L, np.mean(acc), eff, np.mean(cv), float(d @ dny), float(np.linalg.norm(x, axis=1).mean()),
                     float(p[ab].mean() - p[~ab].mean()), auc(p[ab], p[~ab])))
    np.savez(OUT / "dirs.npz", **dirs)
    print("layer pairacc effsize denyAUC cos(abs,deny) |h| gap absAUC(unpaired, in-sample)")
    for r in rows:
        print(f"{r[0]:5d} {r[1]:.3f} {r[2]:6.2f} {r[3]:.3f} {r[4]:6.3f} {r[5]:8.1f} {r[6]:7.2f} {r[7]:.3f}")
    g, U, tok = lens()
    for name in ("abs", "deny"):
        for L in (12, 16, 20, 24, 28, 32, 35):
            lg = U @ (g * dirs[f"{name}{L}"])
            top, bot = lg.argsort()[-12:][::-1], lg.argsort()[:8]
            print(f"{name}{L} + {[tok.decode([t]) for t in top]}  - {[tok.decode([t]) for t in bot]}")


if __name__ == "__main__":
    main()
