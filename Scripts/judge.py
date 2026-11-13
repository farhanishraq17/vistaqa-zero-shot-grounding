"""GROVE evaluation: LLM-judge on Kaggle (src/judge_kernel.py), masks + aggregation locally with the official code.

    python Scripts/judge.py oracle RUN --gt "Competition Data/VistaQA"     # GT-as-pred harness check (must be 100)
    python Scripts/judge.py push   RUN --gt DIR --account K01 [--subset]
    python Scripts/judge.py status RUN
    python Scripts/judge.py score  RUN --gt DIR [--subset]                 # -> runs/RUN/evaluation_results.json
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "Competition Data" / "VistaQA-Evaluation" / "src"))
import kgl  # noqa: E402
from data_loader import load_and_merge_samples, load_records_from_path  # noqa: E402
from scorer import SAEScorer  # noqa: E402
from mask_evaluator import MaskEvaluator  # noqa: E402

JUDGE = "Qwen/Qwen2.5-14B-Instruct-AWQ"


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def gt_for(a):
    """GT records restricted to predicted ids (only with --subset); written to runs/RUN/gt.jsonl for the official loader."""
    run_dir = ROOT / "runs" / a.run
    gt = {r["image"]["image_id"]: r for r in load_records_from_path(a.gt)}
    ids = {p["image"]["image_id"] for p in read_jsonl(run_dir / "preds.jsonl")}
    if ids != set(gt):
        if not a.subset:
            sys.exit(f"preds cover {len(ids & set(gt))}/{len(gt)} GT ids; pass --subset to score a subset")
        gt = {i: gt[i] for i in sorted(ids & set(gt))}
    path = run_dir / "gt.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in gt.values()), encoding="utf-8")
    return gt, path


def oracle(a):
    run_dir = ROOT / "runs" / a.run; run_dir.mkdir(parents=True, exist_ok=True)
    recs = load_records_from_path(a.gt)
    with (run_dir / "preds.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(recs):  # --shift: negative control, answers taken from the next sample
            f.write(json.dumps({"image": r["image"], "prediction": {
                "text": recs[(i + a.shift) % len(recs)]["answer"], "masks": [x["segmentation"] for x in r["annotations"]]}}) + "\n")
    print(f"wrote {len(recs)} oracle preds (shift={a.shift})")


def push(a):
    run_dir = ROOT / "runs" / a.run
    gt, _ = gt_for(a)
    preds = {p["image"]["image_id"]: p for p in read_jsonl(run_dir / "preds.jsonl")}
    items = [{"image_id": i, "question": g["question"], "gt_answer": g["answer"],
              "pred_answer": preds[i]["prediction"]["text"], "task_type": g["task_type"],
              "domain": g["task_domain"]} for i, g in gt.items()]
    kdir = run_dir / "judge"; shutil.rmtree(kdir, ignore_errors=True); kdir.mkdir(parents=True)
    cfg = {"judge": a.judge, **json.loads(a.jcfg)}
    (kdir / "main.py").write_text(f"CFG = {cfg!r}\nITEMS = {json.dumps(items)}\n" +
                                  (ROOT / "src" / "judge_kernel.py").read_text(encoding="utf-8"), encoding="utf-8")
    user = next(l.split(":", 1)[1].strip() for l in kgl.run(a.account, ["config", "view"], capture=True).stdout.splitlines() if "username:" in l)
    slug = re.sub(r"[^a-z0-9-]", "-", f"vlm4rwd-judge-{a.run}".lower())[:50]
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug, "code_file": "main.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=1))
    r = kgl.run(a.account, ["kernels", "push", "-p", str(kdir)], capture=True)
    print((r.stdout + r.stderr).strip().splitlines()[-1])
    (run_dir / "judge.json").write_text(json.dumps({"alias": a.account.upper(), "kernel": f"{user}/{slug}", "judge": a.judge}))


def status(a):
    j = json.loads((ROOT / "runs" / a.run / "judge.json").read_text())
    r = kgl.run(j["alias"], ["kernels", "status", j["kernel"]], capture=True)
    print((r.stdout + r.stderr).strip().splitlines()[-1])


class Verdicts:
    def __init__(self, path):
        self.v = {d["image_id"]: d for d in read_jsonl(path)}

    def evaluate(self, image_id, **_):
        return self.v[image_id]


def score(a):
    run_dir = ROOT / "runs" / a.run
    j = json.loads((run_dir / "judge.json").read_text())
    od = run_dir / "judge_out"; od.mkdir(exist_ok=True)
    if not j.get("cached"):
        kgl.run(j["alias"], ["kernels", "output", j["kernel"], "-p", str(od), "-o"], capture=True)
    _, gt_path = gt_for(a)
    samples = load_and_merge_samples(str(gt_path), str(run_dir / "preds.jsonl"))
    scorer = SAEScorer(epsilon=0.01)
    per = scorer.score_samples(samples, Verdicts(od / "verdicts.jsonl"), j["judge"])
    rep = scorer.build_standard_average_report(per)
    verd = Verdicts(od / "verdicts.jsonl").v
    me = MaskEvaluator()
    text_acc = sum(verd[s.image_id]["correct"] for s in samples) / len(samples)
    miou = sum(me.score(s.gt_masks, s.pred_masks) for s in samples) / len(samples)
    out = {"metadata": {"epsilon": 0.01, "judge_model": j["judge"], "total_samples": len(samples)},
           "overall": {**rep["overall"], "text_acc": text_acc, "mask_miou": miou}, "breakdown": rep["breakdown"]}
    (run_dir / "evaluation_results.json").write_text(json.dumps(out, indent=2))
    (run_dir / "per_sample.jsonl").write_text("".join(json.dumps(p) + "\n" for p in per))
    print(f"GROVE {100 * rep['overall']['standard_average']:.2f} | text acc {100 * text_acc:.2f} | mIoU {100 * miou:.2f} | n={len(samples)}")
    for k, rows in rep["breakdown"].items():
        print(f"  {k}: " + ", ".join(f"{r['group']} {100 * r['standard_average']:.1f}" for r in rows))


def cache(a):
    """judge.py cache RUN [--prefix dev]: reuse verdicts of earlier judged runs for identical (id, question, answer)
    (the greedy judge rarely flips on unchanged answers: 0 on proxy dev, 3/802 across kernels on VistaQA; if runs disagree
    the last one globbed wins). Complete -> judge_out/verdicts.jsonl + judge.json, then score.
    Proxy dev and VistaQA ids overlap: --prefix picks which runs feed the cache, never mix the two. Only runs whose judge.json
    names --judge feed it (AWQ by default; full-judge runs need --judge Qwen/Qwen2.5-14B-Instruct)."""
    seen = {}
    for vf in (ROOT / "runs").glob("*/judge_out/verdicts.jsonl"):
        d = vf.parent.parent
        if d.name == a.run or not (d / "preds.jsonl").exists() or not d.name.startswith(tuple(a.prefix)):
            continue
        if json.loads((d / "judge.json").read_text()).get("judge") != a.judge:  # never mix AWQ and full-judge verdicts
            continue
        ver = {v["image_id"]: v for v in read_jsonl(vf)}
        for p in read_jsonl(d / "preds.jsonl"):
            i = p["image"]["image_id"]
            if i in ver:
                seen[(i, p.get("question", ""), p["prediction"]["text"])] = ver[i]
    run_dir = ROOT / "runs" / a.run
    preds = read_jsonl(run_dir / "preds.jsonl")
    hit = [seen.get((p["image"]["image_id"], p.get("question", ""), p["prediction"]["text"])) for p in preds]
    miss = [p["image"]["image_id"] for p, h in zip(preds, hit) if h is None]
    if miss:
        sys.exit(f"{len(miss)} answers never judged (e.g. ids {miss[:8]}) -> judge.py push")
    (run_dir / "judge_out").mkdir(exist_ok=True)
    (run_dir / "judge_out" / "verdicts.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hit), encoding="utf-8")
    (run_dir / "judge.json").write_text(json.dumps({"judge": a.judge, "cached": True}))
    print(f"all {len(preds)} verdicts cached -> judge.py score {a.run}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["oracle", "push", "status", "score", "cache"])
    p.add_argument("--prefix", nargs="*", default=["dev", "pc", "cf", "s3"], help="cache: run-name prefixes (proxy dev runs)")
    p.add_argument("run")
    p.add_argument("--gt", default=str(ROOT / "Competition Data" / "VistaQA"))
    p.add_argument("--account", default="K01")
    p.add_argument("--judge", default=JUDGE)
    p.add_argument("--jcfg", default="{}", help='push: extra kernel CFG, e.g. {"llm": {"gpu_memory_utilization": 0.98}}')
    p.add_argument("--subset", action="store_true")
    p.add_argument("--shift", type=int, default=0)
    a = p.parse_args()
    {"oracle": oracle, "push": push, "status": status, "score": score, "cache": cache}[a.cmd](a)
