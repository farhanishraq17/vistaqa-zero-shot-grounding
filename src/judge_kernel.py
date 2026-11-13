"""Kaggle kernel: official VistaQA LLM-judge (prompt + parser from VistaQA-Evaluation) served by vLLM.

Scripts/judge.py prepends `CFG = {...}` and `ITEMS = [...]` (image_id, question, gt_answer, pred_answer, task_type, domain).
Writes /kaggle/working/verdicts.jsonl.
"""
import json
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", *CFG.get("pip", ["vllm"])], check=True)  # noqa: F821
# vllm brings its own torch build; the image's torchaudio no longer matches it and transformers imports it
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/VistaQA/VistaQA-Evaluation", "/tmp/ev"], check=True)
sys.path.insert(0, "/tmp/ev/src")
from config import JUDGE_PROMPT_TEMPLATE  # noqa: E402
from text_evaluator import TextEvaluator  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

llm = LLM(model=CFG["judge"], **{"tensor_parallel_size": 2, "dtype": "half", "max_model_len": 4096,  # noqa: F821
                                 "gpu_memory_utilization": 0.9, "enforce_eager": CFG.get("eager", False),  # noqa: F821
                                 **CFG.get("llm", {})})  # noqa: F821
msgs = [[{"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(
    question=it["question"], gt_answer=it["gt_answer"], pred_answer=it["pred_answer"],
    task_type=it["task_type"], domain=it["domain"])}] for it in ITEMS]  # noqa: F821
outs = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=256))
with open("/kaggle/working/verdicts.jsonl", "w") as f:
    for it, o in zip(ITEMS, outs):  # noqa: F821
        raw = o.outputs[0].text
        try:
            correct, reason = TextEvaluator._parse_judge_response(None, raw)
        except (TypeError, ValueError):  # official parser crashes on e.g. "correct": null -> wrong, like unparseable JSON
            correct, reason = 0, f"Judge verdict cannot be parsed: {raw[:200]}"
        f.write(json.dumps({"image_id": it["image_id"], "correct": correct, "reason": reason, "raw": raw}) + "\n")
print("judged", len(ITEMS), "accuracy", sum(json.loads(l)["correct"] for l in open("/kaggle/working/verdicts.jsonl")) / len(ITEMS))
