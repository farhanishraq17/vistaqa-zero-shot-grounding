"""Shard a pipeline run across Kaggle accounts.

    python Scripts/fleet.py push  RUN --accounts K01,K02 --cfg '{"vlm": "Qwen/Qwen3-VL-8B-Instruct"}' [--script src/pipeline.py]
    python Scripts/fleet.py status RUN
    python Scripts/fleet.py collect RUN        # -> runs/RUN/preds.jsonl (+ raw.jsonl)

One shard per account (shard k = image ids k::N). State lives in runs/RUN/manifest.json.
An account can also be a shared GPU PC, "PC1:1+0" = PC1's GPUs 1 and 0 in that order (--need 13 6: free GiB each must have).
There, cfg values starting with "@/" resolve inside the PC's workspace, e.g. {"data": "@/data/proxy2"}.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import kgl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def whoami(alias):
    out = kgl.run(alias, ["config", "view"], capture=True).stdout
    return next(l.split(":", 1)[1].strip() for l in out.splitlines() if "username:" in l)


def dl_hosts():
    """Shared GPU PCs from the git-ignored SuperComp.env: {alias: (ssh target, our folder in its workspace)}."""
    f = ROOT / "SuperComp.env"
    if not f.exists():
        return {}
    env = dict(l.split("=", 1) for l in f.read_text().splitlines() if "=" in l and not l.startswith("#"))
    return {k: (f"{env[k + '_USER']}@{env[k + '_HOST']}", env[k + "_WORKSPACE"] + "/vlm4rwd") for k in ("PC1", "PC2")}


def ssh(host, cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, cmd], capture_output=True, text=True)


def dl_push(a, k, alias, gpus, kdir, cfg, src):
    """One shard as a detached, niced process on a shared PC, only if its GPUs have the free memory asked for. No credentials go there."""
    host, ws = dl_hosts()[alias]
    ids, rd = gpus.replace("+", ","), f"{ws}/runs/{a.run}/shard{k}"
    free = [int(x) / 1024 for x in ssh(host, f"nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i {ids}").stdout.split()]
    print(alias, f"GPUs {ids} free GiB", [round(x, 1) for x in free])
    if a.need and (len(free) != len(a.need) or any(f < n for f, n in zip(free, a.need))):
        sys.exit(f"{alias}: need {a.need} GiB free on GPUs {ids}; not starting")
    cfg = {**{c: (ws + v[1:] if isinstance(v, str) and v.startswith("@/") else v) for c, v in cfg.items()}, "out": rd}
    (kdir / "main.py").write_text(f"CFG = {cfg!r}\n" + src, encoding="utf-8")
    ssh(host, f"mkdir -p {rd}")
    subprocess.run(["scp", "-q", str((kdir / "main.py").relative_to(ROOT)), f"{host}:{rd}/main.py"], cwd=ROOT, check=True)
    env = f"CUDA_VISIBLE_DEVICES={ids} HF_HOME={ws}/hf HF_HUB_DISABLE_XET=1 KAGGLEHUB_CACHE={ws}/models/kagglehub"
    print(alias, ssh(host, f"cd {rd} && {{ {env} setsid nohup nice -n 10 {ws}/env/bin/python -u {rd}/main.py > log.txt 2>&1 < /dev/null & }} ; echo started {rd}").stdout.strip())


def dl_status(run, s):
    host, ws = dl_hosts()[s["alias"]]
    rd = f"{ws}/runs/{run}/shard{s['shard']}"
    return ssh(host, f"pgrep -f '{rd}/[m]ain.py' >/dev/null && echo -n 'RUNNING ' || echo -n 'STOPPED '; cat {rd}/pred_shard{s['shard']}.jsonl 2>/dev/null | wc -l | tr '\\n' ' '; "
                     f"tail -1 {rd}/log.txt | cut -c1-110").stdout.strip()


def push(a):
    run_dir = ROOT / "runs" / a.run
    accounts = [x.strip().upper() for x in a.accounts.split(",")]
    cfg = json.loads(Path(a.cfg[1:]).read_text(encoding="utf-8") if a.cfg.startswith("@") else a.cfg)  # @file: long cfgs (WSP edits)
    src = (ROOT / a.script).read_text(encoding="utf-8")
    samples = []
    if a.inline:  # ship only image meta + question + public URL; GT stays local
        for p in sorted(Path(a.inline).glob("*.json"), key=lambda p: int(p.stem)):
            r = json.loads(p.read_text(encoding="utf-8"))
            samples.append({"image": {k: r["image"][k] for k in ("image_id", "width", "height")},
                            "question": r["question"], "url": r["source"]["url"]})
        cfg["data"] = "inline"
    if a.concepts:  # SAM3 text prompts from an earlier run: answer run -> top evidence label (kernel keeps counting
        sys.path.insert(0, str(ROOT / "src"))  # Qs only); premise run -> the "missing" phrase (presence veto, all Qs)
        from collections import Counter
        from pipeline import parse_missing, parse_output
        cfg["concepts"] = {}
        for l in (ROOT / "runs" / a.concepts / "raw.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(l) if l.strip() else None
            if r and isinstance(r["raw"], dict) and "premise" in r["raw"]:
                cfg["count_only"] = False
                labs = [m for m in [parse_missing(r["raw"]["premise"])] if m]
            else:
                labs = [e["label"] for e in parse_output(r["raw"])["evidence"] if e["label"]] if r and isinstance(r["raw"], str) else []
            if labs:
                cfg["concepts"][r["image_id"]] = Counter(labs).most_common(1)[0][0]
        print(len(cfg["concepts"]), "concept prompts")
    if a.pairs:  # answer selector: the two runs' answers (model output, not GT) where their normalized text differs
        rd = lambda r: {json.loads(l)["image"]["image_id"]: json.loads(l)["prediction"]["text"] for l in
                        (ROOT / "runs" / r / "preds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        ta, tb = map(rd, a.pairs)
        norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
        cfg["pairs"] = {i: [ta[i], tb[i]] for i in ta if i in tb and norm(ta[i]) != norm(tb[i])}
        print(len(cfg["pairs"]), "pairs differ of", len(ta))
    if a.marks:
        cfg["marks"] = marks_from(a.marks)
        print(len(cfg["marks"]), "predictions to check")
    if a.boxes:  # re-segment an earlier run's VLM boxes (model output, not GT) -> segmenter ablation without the VLM
        cfg["boxes"] = {json.loads(l)["image_id"]: json.loads(l)["boxes"] for l in
                        (ROOT / "runs" / a.boxes / "raw.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    manifest = []
    for k, acct in enumerate(accounts):
        alias, _, gpus = acct.partition(":")
        kdir = run_dir / "kernels" / f"shard{k}"
        shutil.rmtree(kdir, ignore_errors=True); kdir.mkdir(parents=True)
        shard_cfg = {**cfg, "shard": k, "num_shards": len(accounts), **({"samples": samples} if samples else {})}
        cfg_log = {k2: v for k2, v in cfg.items() if k2 not in ("boxes", "concepts", "pairs", "marks")}
        if alias in dl_hosts():
            manifest.append({"shard": k, "alias": alias, "gpus": gpus})
            if not a.only or k in a.only:
                dl_push(a, k, alias, gpus, kdir, shard_cfg, src)
            continue
        user = whoami(alias)
        slug = re.sub(r"[^a-z0-9-]", "-", f"vlm4rwd-{a.run}-s{k}".lower())[:50]
        (kdir / "main.py").write_text(f"CFG = {shard_cfg!r}\n" + src, encoding="utf-8")
        (kdir / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug}", "title": slug, "code_file": "main.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": a.machine, "dataset_sources": [d.format(user=user) for d in a.datasets], "competition_sources": [],
            "kernel_sources": [], "model_sources": a.models}, indent=1))
        manifest.append({"shard": k, "alias": alias, "kernel": f"{user}/{slug}"})
        if a.only and k not in a.only:
            continue
        r = kgl.run(alias, ["kernels", "push", "-p", str(kdir)], capture=True)
        print(alias, user, slug, (r.stdout + r.stderr).strip().splitlines()[-1])
    (run_dir / "manifest.json").write_text(json.dumps({"cfg": cfg_log, **({"boxes_from": a.boxes} if a.boxes else {}),
                                                       **({"concepts_from": a.concepts} if a.concepts else {}),
                                                       **({"marks_from": a.marks} if a.marks else {}),
                                                       "shards": manifest}, indent=1))


def marks_from(run):
    """Mask check input: RUN's predictions (model output, not GT) with >= 2 masks on non-counting questions ->
    {id: {"answer", "boxes": each mask's bounding box, 0-1000 of the mask's own [H, W]}}."""
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline import COUNT_Q
    from pycocotools import mask as mu
    out = {}
    for l in (ROOT / "runs" / run / "preds.jsonl").read_text(encoding="utf-8").splitlines():
        p = json.loads(l) if l.strip() else None
        if p and len(p["prediction"]["masks"]) >= 2 and not COUNT_Q.search(p["question"]):
            bx = [(m["size"], mu.toBbox({"size": m["size"], "counts": m["counts"].encode()}).tolist()) for m in p["prediction"]["masks"]]
            out[p["image"]["image_id"]] = {"answer": p["prediction"]["text"], "boxes": [
                [round(1000 * v / s, 1) for v, s in zip((x, y, x + w, y + h), (W, H, W, H))] for (H, W), (x, y, w, h) in bx]}
    return out


def status(a):
    for s in json.loads((ROOT / "runs" / a.run / "manifest.json").read_text())["shards"]:
        if "kernel" not in s:
            print(f"shard{s['shard']} {s['alias']:12s} {dl_status(a.run, s)}")
            continue
        r = kgl.run(s["alias"], ["kernels", "status", s["kernel"]], capture=True)
        print(f"shard{s['shard']} {s['alias']:12s} {(r.stdout + r.stderr).strip().splitlines()[-1]}")


def collect(a):
    run_dir = ROOT / "runs" / a.run
    preds, raws = [], []
    for s in json.loads((run_dir / "manifest.json").read_text())["shards"]:
        od = run_dir / "out" / f"shard{s['shard']}"
        od.mkdir(parents=True, exist_ok=True)
        if "kernel" in s:
            kgl.run(s["alias"], ["kernels", "output", s["kernel"], "-p", str(od), "-o"], capture=True)
        else:
            if dl_status(a.run, s).startswith("RUNNING"):
                print(f"WARNING shard{s['shard']} on {s['alias']} is still running: partial output")
            host, ws = dl_hosts()[s["alias"]]
            rd, n = f"{ws}/runs/{a.run}/shard{s['shard']}", s["shard"]
            subprocess.run(["scp", "-q", f"{host}:{rd}/pred_shard{n}.jsonl", f"{host}:{rd}/raw_shard{n}.jsonl", f"{host}:{rd}/log.txt",
                            str(od.relative_to(ROOT))], cwd=ROOT, check=True)
        for f in od.glob("pred_shard*.jsonl"):
            preds += f.read_text(encoding="utf-8").splitlines()
        for f in od.glob("raw_shard*.jsonl"):
            raws += f.read_text(encoding="utf-8").splitlines()
    (run_dir / "preds.jsonl").write_text("\n".join(preds) + "\n", encoding="utf-8")
    (run_dir / "raw.jsonl").write_text("\n".join(raws) + "\n", encoding="utf-8")
    print(f"{len(preds)} predictions -> {run_dir / 'preds.jsonl'}")


def merge(a):
    """fleet.py merge NEW --parts RUN1 RUN2: concatenate collected preds/raw (e.g. dev 1-420 + dev2 421-560)."""
    out = ROOT / "runs" / a.run; out.mkdir(parents=True, exist_ok=True)
    for name in ("preds.jsonl", "raw.jsonl"):
        lines = [l for r in a.parts for l in (ROOT / "runs" / r / name).read_text(encoding="utf-8").splitlines() if l.strip()]
        (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} records -> {out}")


def remask(a):
    """fleet.py remask NEW --parts BASE RESEG: BASE answers (and its judge verdicts, if any) with RESEG's masks.
    System order: remask (VLM pass + seg pass) -> override (premise) -> countfix (concept)."""
    rd = lambda r: [json.loads(l) for l in (ROOT / "runs" / r / "preds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    base, reseg = a.parts
    masks = {p["image"]["image_id"]: p["prediction"]["masks"] for p in rd(reseg)}
    pick = lambda p: masks[p["image"]["image_id"]]
    if a.route:  # only images the router called a figure take RESEG's masks; the rest keep BASE's
        yes = {p["image"]["image_id"] for p in rd(a.route) if p["prediction"]["text"].strip().lower().startswith("yes")}
        pick = lambda p: masks[p["image"]["image_id"]] if p["image"]["image_id"] in yes else p["prediction"]["masks"]
    preds = [{**p, "prediction": {**p["prediction"], "masks": pick(p)}} for p in rd(base)]
    save_same_text(a.run, base, preds)
    print(f"{len(preds)} predictions with {reseg} masks -> runs/{a.run} (score with judge.py score; text verdicts reused)")


def save_same_text(run, base, preds):
    """Write NEW preds whose answers are BASE's, with BASE's raw output and judge verdicts (no new judging)."""
    out = ROOT / "runs" / run; out.mkdir(parents=True, exist_ok=True)
    (out / "preds.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    for f in ("raw.jsonl", "judge.json"):
        if (ROOT / "runs" / base / f).exists():
            shutil.copy(ROOT / "runs" / base / f, out / f)
    if (ROOT / "runs" / base / "judge_out").exists():  # a cached base (judge.py cache) has no kernel to re-download from
        shutil.copytree(ROOT / "runs" / base / "judge_out", out / "judge_out", dirs_exist_ok=True)


def verify(a):
    """fleet.py verify NEW --parts BASE VER: BASE predictions keeping only the masks VER's mask check listed (push --marks
    BASE; no valid list = keep all). Answers are unchanged, so BASE's verdicts are reused."""
    base, ver = a.parts
    raws = [json.loads(l) for l in (ROOT / "runs" / ver / "raw.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = {r["image_id"]: r["raw"]["keep"] for r in raws if isinstance(r["raw"], dict) and r["raw"].get("keep")}
    preds = [json.loads(l) for l in (ROOT / "runs" / base / "preds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for p in preds:
        k = keep.get(p["image"]["image_id"])
        if k:
            p["prediction"]["masks"] = [m for j, m in enumerate(p["prediction"]["masks"]) if j in k]
    save_same_text(a.run, base, preds)
    n = sum(len(k) for k in keep.values())
    print(f"{len(keep)} checked predictions kept {n} masks -> runs/{a.run} (score with judge.py score; text verdicts reused)")


def override(a):
    """fleet.py override NEW --parts BASE PREMISE [VETO] [--confirm] [--veto-tau 0.7]: BASE predictions, replaced by the
    premise-only run's "There is no X" answers (no masks) where it found a missing premise (--confirm: only if its
    yes/no check said no; VETO: a SAM3 concept run on the missing phrase -> keep BASE if SAM3 finds X >= veto-tau)."""
    rd = lambda r, n: [json.loads(l) for l in (ROOT / "runs" / r / n).read_text(encoding="utf-8").splitlines() if l.strip()]
    base, prem, *veto = a.parts
    ver = {x["image_id"]: x["raw"].get("verify", "") for x in rd(prem, "raw.jsonl") if isinstance(x["raw"], dict)}
    found = {x["image_id"]: max(x["raw"]["scores"], default=0) for v in veto for x in rd(v, "raw.jsonl")}
    said_no = lambda v: v.strip().lstrip("*_\"'`").lower().startswith("no")  # "**No**" too (no 8B reply has markdown)
    flagged = lambda run: {p["image"]["image_id"] for p in rd(run, "preds.jsonl") if p["prediction"]["text"]} - (
        {x["image_id"] for x in rd(run, "raw.jsonl") if isinstance(x["raw"], dict)
         and not said_no(x["raw"].get("verify", ""))} if a.confirm else set())
    also = flagged(a.and_prem) if a.and_prem else None  # --and-prem: a second premise run must flag it too
    over = {p["image"]["image_id"]: p for p in rd(prem, "preds.jsonl") if p["prediction"]["text"]
            and not (a.confirm and not said_no(ver.get(p["image"]["image_id"], "")))
            and found.get(p["image"]["image_id"], 0) < a.veto_tau and (also is None or p["image"]["image_id"] in also)}
    preds = [over.get(p["image"]["image_id"], p) for p in rd(base, "preds.jsonl")]
    out = ROOT / "runs" / a.run; out.mkdir(parents=True, exist_ok=True)
    (out / "preds.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    shutil.copy(ROOT / "runs" / base / "raw.jsonl", out / "raw.jsonl")
    print(f"{len(over)} of {len(preds)} answers overridden -> {out}")


def countfix(a):
    """fleet.py countfix NEW --parts BASE CONCEPT [VLM2 ...] [--tau 0.7]: pipeline.merge_count on BASE's masked answers
    to counting questions (= ids in the CONCEPT run): SAM3 instances with score >= tau and the counts stated by other
    VLM runs may raise the count, on plain counting questions only (pipeline.plain_count; --no-guard: all).
    Text changes -> judge NEW again."""
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline import answer_count, merge_count, plain_count
    rd = lambda r, n: [json.loads(l) for l in (ROOT / "runs" / r / n).read_text(encoding="utf-8").splitlines() if l.strip()]
    base, cpt, *vlms = a.parts
    sc = {x["image_id"]: x["raw"]["scores"] for x in rd(cpt, "raw.jsonl")}
    cm = {p["image"]["image_id"]: p["prediction"]["masks"] for p in rd(cpt, "preds.jsonl")}
    extra = [{p["image"]["image_id"]: p["prediction"] for p in rd(r, "preds.jsonl")} for r in vlms]
    preds, n = [], 0
    for p in rd(base, "preds.jsonl"):
        i, pr = p["image"]["image_id"], p["prediction"]
        if i in cm and pr["masks"] and (a.no_guard or plain_count(p["question"])):
            keep = [m for m, s in zip(cm[i], sc[i]) if s >= a.tau]
            others = [(len(keep), keep)] + [(answer_count(e[i]["text"]) or 0, e[i]["masks"]) for e in extra if i in e]
            text, masks = merge_count(pr["text"], pr["masks"], others)
            if a.keep_masks and not masks:  # the winning counter has none (count-only VLM run): keep the larger set we have
                masks = max(keep, pr["masks"], key=len)
            n += text != pr["text"]
            p = {**p, "prediction": {"text": text, "masks": masks}}
        preds.append(p)
    out = ROOT / "runs" / a.run; out.mkdir(parents=True, exist_ok=True)
    (out / "preds.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    shutil.copy(ROOT / "runs" / base / "raw.jsonl", out / "raw.jsonl")
    print(f"{n} counts raised -> {out}")


def dedup(a):
    """fleet.py dedup NEW --parts BASE [--iou 0.5]: BASE predictions with near-duplicate masks dropped (pipeline.dedup_masks,
    VLM evidence order). Text unchanged -> judge.py cache NEW reuses the verdicts."""
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline import dedup_masks
    rd = lambda r, n: [json.loads(l) for l in (ROOT / "runs" / r / n).read_text(encoding="utf-8").splitlines() if l.strip()]
    preds = [{**p, "prediction": {**p["prediction"], "masks": dedup_masks(p["prediction"]["masks"], a.iou)}} for p in rd(a.parts[0], "preds.jsonl")]
    n = sum(len(p["prediction"]["masks"]) for p in rd(a.parts[0], "preds.jsonl")) - sum(len(p["prediction"]["masks"]) for p in preds)
    out = ROOT / "runs" / a.run; out.mkdir(parents=True, exist_ok=True)
    (out / "preds.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    shutil.copy(ROOT / "runs" / a.parts[0] / "raw.jsonl", out / "raw.jsonl")
    print(f"{n} masks dropped -> {out}")


def clean(a):
    """fleet.py clean NEW --parts BASE --mode fill|lcc|both: BASE predictions with every mask cleaned (pipeline.clean_masks,
    round R7). Text unchanged -> reuse BASE's verdicts."""
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline import clean_masks
    rd = lambda r, n: [json.loads(l) for l in (ROOT / "runs" / r / n).read_text(encoding="utf-8").splitlines() if l.strip()]
    preds = [{**p, "prediction": {**p["prediction"], "masks": clean_masks(p["prediction"]["masks"], a.mode)}} for p in rd(a.parts[0], "preds.jsonl")]
    out = ROOT / "runs" / a.run; out.mkdir(parents=True, exist_ok=True)
    (out / "preds.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    shutil.copy(ROOT / "runs" / a.parts[0] / "raw.jsonl", out / "raw.jsonl")
    print(f"masks cleaned ({a.mode}) -> {out}")


def splice(a):
    """fleet.py splice NEW --parts BASE PATCH [--scope RUN]: on the ids of RUN's preds (default PATCH's), PATCH's preds and
    raw lines replace BASE's (an id PATCH lacks there is dropped); elsewhere BASE's (round R6: high-res passes on the
    large images; a concept run has lines only for its counting questions, so its scope is the answer run's ids)."""
    rd = lambda r, n: [json.loads(l) for l in (ROOT / "runs" / r / n).read_text(encoding="utf-8").splitlines() if l.strip()]
    key = lambda x: x["image"]["image_id"] if "image" in x else x["image_id"]
    scope = {key(x) for x in rd(a.scope or a.parts[1], "preds.jsonl")}
    out = ROOT / "runs" / a.run; out.mkdir(parents=True, exist_ok=True)
    for n in ("preds.jsonl", "raw.jsonl"):
        new = [x for x in rd(a.parts[1], n) if key(x) in scope]
        rows = [x for x in rd(a.parts[0], n) if key(x) not in scope] + new
        (out / n).write_text("".join(json.dumps(x) + "\n" for x in sorted(rows, key=key)), encoding="utf-8")
    print(f"{len(new)} lines from {a.parts[1]} on {len(scope)} scoped ids, {len(rows)} in all -> {out}")


def pick(a):
    """fleet.py pick NEW --parts A B SEL: per sample A's prediction, or B's (text and masks) where the selector run SEL
    (cfg pairs from `push --pairs A B`) chose B's answer in both orders (pipeline.prefer_b). Text changes -> judge NEW."""
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline import prefer_b
    rd = lambda r, n: [json.loads(l) for l in (ROOT / "runs" / r / n).read_text(encoding="utf-8").splitlines() if l.strip()]
    ra, rb, sel = a.parts
    b = {p["image"]["image_id"]: p for p in rd(rb, "preds.jsonl")}
    use = {x["image_id"] for x in rd(sel, "raw.jsonl") if isinstance(x["raw"], dict) and prefer_b(x["raw"])}
    preds = [b[p["image"]["image_id"]] if p["image"]["image_id"] in use else p for p in rd(ra, "preds.jsonl")]
    out = ROOT / "runs" / a.run; out.mkdir(parents=True, exist_ok=True)
    (out / "preds.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    shutil.copy(ROOT / "runs" / ra / "raw.jsonl", out / "raw.jsonl")
    print(f"{len(use)} of {len(preds)} predictions from {rb} -> {out}")


def think(a):
    """fleet.py think NEW --parts BASE THINK: BASE predictions, with THINK's answer text (a `thinking: true` run) where its
    trace closed. An unclosed trace hit the token limit: keep BASE's no-think answer (what a deployed system would re-run;
    greedy, so identical). Masks stay BASE's. Text changes -> judge NEW again."""
    rd = lambda r, n: [json.loads(l) for l in (ROOT / "runs" / r / n).read_text(encoding="utf-8").splitlines() if l.strip()]
    base, th = a.parts
    closed = {x["image_id"] for x in rd(th, "raw.jsonl")  # thinking trace closed, or (cfg pot) the program ran
              if "</think>" in str(x["raw"]) or (isinstance(x["raw"], dict) and x["raw"].get("ok"))}
    text = {p["image"]["image_id"]: p["prediction"]["text"] for p in rd(th, "preds.jsonl")}
    use = {i for i in closed if text.get(i)}
    preds = [{**p, "prediction": {**p["prediction"], "text": text[p["image"]["image_id"]]}} if p["image"]["image_id"] in use else p
             for p in rd(base, "preds.jsonl")]
    out = ROOT / "runs" / a.run; out.mkdir(parents=True, exist_ok=True)
    (out / "preds.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    shutil.copy(ROOT / "runs" / base / "raw.jsonl", out / "raw.jsonl")
    print(f"{len(use)} thinking answers, {len(text) - len(use)} unclosed kept no-think -> {out}")


def image_meta(d):
    """{image_id: [H, W]} as the samples declare it; reads image meta only, never answers/annotations."""
    ims = (json.loads(f.read_text(encoding="utf-8"))["image"] for f in Path(d).glob("*.json"))
    return {im["image_id"]: [im["height"], im["width"]] for im in ims}


def fit(a):
    """fleet.py fit RUN --inline DIR: resize masks (nearest) to the [H, W] the sample declares. Some VistaQA image
    files are smaller than their JSON says (ids 138, 496); the official scorer IoUs against masks of the declared size."""
    import numpy as np
    from PIL import Image
    from pycocotools import mask as mu
    meta, n = image_meta(a.inline), 0
    path = ROOT / "runs" / a.run / "preds.jsonl"
    preds = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for p in preds:
        h, w = meta[p["image"]["image_id"]]
        for m in p["prediction"]["masks"]:
            if m["size"] != [h, w]:
                d = Image.fromarray(mu.decode({"size": m["size"], "counts": m["counts"].encode()}))
                m["counts"] = mu.encode(np.asfortranarray(np.asarray(d.resize((w, h), Image.NEAREST), dtype=np.uint8)))["counts"].decode()
                m["size"], n = [h, w], n + 1
    path.write_text("".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    print(f"{n} masks resized -> {path}")


def check(a):
    """fleet.py check RUN --inline DIR: submission format only (reads image meta, never answers/annotations):
    every image_id exactly once, masks decodable RLE of size [H, W], non-empty."""
    from pycocotools import mask as mu
    meta = image_meta(a.inline)
    preds = [json.loads(l) for l in (ROOT / "runs" / a.run / "preds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [p["image"]["image_id"] for p in preds]
    bad = sum(m["size"] != meta.get(p["image"]["image_id"]) or not mu.decode({"size": m["size"], "counts": m["counts"].encode()}).any()
              for p in preds for m in p["prediction"]["masks"])
    print(f"{len(ids)} preds | unique {len(set(ids))} | missing {len(set(meta) - set(ids))} | unknown {len(set(ids) - set(meta))} | "
          f"bad masks {bad} | empty answers {sum(not p['prediction']['text'] for p in preds)} | masks/sample {sum(len(p['prediction']['masks']) for p in preds) / max(1, len(preds)):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["push", "status", "collect", "merge", "override", "remask", "countfix", "think", "dedup", "pick", "clean", "splice", "fit", "check", "verify"])
    p.add_argument("--tau", type=float, default=0.7, help="countfix: SAM3 concept score threshold")
    p.add_argument("--no-guard", action="store_true", help="countfix: also raise on non-plain counting questions")
    p.add_argument("--route", help="remask: router run (cfg route); only ids it answered yes take RESEG's masks")
    p.add_argument("--keep-masks", action="store_true", help="countfix: a raise by a count-only run keeps masks (the entry's chain lost them)")
    p.add_argument("--veto-tau", type=float, default=0.7, help="override: SAM3 presence score that vetoes a premise override")
    p.add_argument("run")
    p.add_argument("--parts", nargs="*", default=[])
    p.add_argument("--accounts", default="K01")
    p.add_argument("--cfg", default="{}")
    p.add_argument("--script", default="src/pipeline.py")
    p.add_argument("--machine", default="NvidiaTeslaT4")
    p.add_argument("--datasets", nargs="*", default=[], help="{user} = the shard's account (private per-account copies)")
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--inline", help="local GT dir; ship questions + image URLs inside the kernel")
    p.add_argument("--confirm", action="store_true", help="override: require the premise run's yes/no check to say no")
    p.add_argument("--concepts", help="push: run whose evidence labels become SAM3 text prompts (counting Qs, no VLM)")
    p.add_argument("--boxes", help="push: run whose raw.jsonl pixel boxes are re-segmented (no VLM)")
    p.add_argument("--marks", help="push: run whose predictions (>= 2 masks, not counting) get the numbered-box mask check (cfg marks)")
    p.add_argument("--pairs", nargs=2, help="push: answer-selector pass over runs A B where their answers differ (cfg pairs)")
    p.add_argument("--and-prem", help="override: a second premise run that must flag the question too (its --confirm check applies)")
    p.add_argument("--iou", type=float, default=0.5, help="dedup: mask IoU at which a later mask is a duplicate")
    p.add_argument("--mode", choices=["fill", "lcc", "both"], help="clean: mask cleanup variant (round R7)")
    p.add_argument("--scope", help="splice: run whose pred ids PATCH replaces (default PATCH)")
    p.add_argument("--only", type=int, nargs="*", help="push only these shard indices (re-push after a failure)")
    p.add_argument("--need", type=float, nargs="*", help="push to a shared PC: free GiB each of its GPUs must have, else don't start")
    a = p.parse_args()
    {"push": push, "status": status, "collect": collect, "merge": merge, "override": override, "remask": remask, "countfix": countfix, "think": think, "dedup": dedup, "pick": pick, "clean": clean, "splice": splice, "fit": fit, "check": check, "verify": verify}[a.cmd](a)
