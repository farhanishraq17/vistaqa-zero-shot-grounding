# PLAN — VistaQA Zero-Shot Grounding (Track 1, paper track)

## Status and roadmap (updated 2026-11-07)
**Entry `final` = 48.40 GROVE (48.14 unquantized judge), fixed.** The zero-shot rule forbids picking a system by its VistaQA score,
and our registration says no later row replaces `final`. Post-freeze rows are analysis: v2 48.01, v3 48.21, v4 48.37, v5 48.03 (v3–v5 level with each other); post-hoc `final-km`
48.47, `final-v` 49.03 (not submittable). Results: RESULTS.md; every run and rule: EXPERIMENTS.md.

Rounds (rule written before dev results, decided on proxy dev + proxy2 only; adopted rounds go into one registered VistaQA row `v4`):
| Round | Idea | Status |
|---|---|---|
| R1 | answer selector between the Qwen3.6 and Qwen3.8 answers (both orders) | not adopted: position bias; proxy2 56.85 < 60.04 |
| R2 | drop near-duplicate masks | not adopted: dev has almost none |
| R3 | premise override also needs Qwen3.8's premise check | not adopted: proxy2 +0.35, proxy dev −0.40 |
| R4 | calculator tool for figure arithmetic (math gap 26.1 vs 41.5) | adopted as fallback for unfinished thinking (probe 146 vs 143) → `v4` registered |
| R5 | larger answerer on PC1 (2×24 GB) if a newer open model > 27B fits | dropped (Nov 7): newer ones are 108–321B (60–180 GB at 4-bit); the only fit, InternVL3.5-38B (Sep 2025), is older than our 27Bs |
| R6 | answerer input cap 1280² → 2048² (536 VistaQA images are larger; dev sets have none) | adopted: V*Bench text 176 → 192/238, box hit 66 → 72% → `v5` 48.03 [47.34]: level with v4, did not transfer |
| R7 | mask cleanup: fill holes / largest component | not adopted: fill +0.43 proxy dev but −0.05 proxy2; lcc hurts both |

| Dates | Step |
|---|---|
| Nov 7–8 | R1–R3 decided; R4 built and tested on the MathVista probe |
| Nov 9 | R5 if a candidate exists |
| Nov 7 | v4 registered, run, judged both ways: 48.37 [47.37], level with v3 (done early) |
| Nov 7 | development closed (R1–R7 done; v4, v5 level with v3); qualitative appendix figure made after the close |
| Nov 7 | public copy pushed to anonymized-repository-link (`4947a78`; MIT, no-reply author); private until the user makes it public |
| Nov 7 | weight-space programming test (user request, analysis only): Claude edits the 8B's weights from practice data; edited vs unedited 8B +2.57 pooled dev, +1.70 VistaQA; with the premise check +0.28 / +0.56 (not significant); report appendix |
| Nov 8 | mask verifier test (user request, analysis only): Set-of-Mark check of the entry's masks, new dev set proxy3 (RefCOCOg); 27B check −0.38 pooled dev, +0.21 VistaQA (`final-mv` 48.61, not significant); report appendix |
| Nov 8 | model swaps (user request, analysis only): Qwen3-VL-32B answerer +0.05 pooled dev, −5.02 VistaQA (`bb32` 43.38); Gemma 4 31B premise checker −1.80 pooled dev, +1.35 VistaQA (`final-gp` 49.76, not significant); report appendix |
| Nov 13 | final PDF; `python Scripts/make_release.py` then `git -C public_release push` updates the public repo |
| Nov 14 | user submits on OpenReview (deadline Nov 15 AoE) |

Report still needs: team name, authors, affiliation, email (the public link is in). The repo must be public before submission.

## Original plan (written Oct 27; phases P0–P4 are done)

Target: beat the published hybrid baselines (Qwen3-VL-4B+SAM3 **43.30**, GPT-5.4-T+SAM3 **45.53** GROVE) with an
open-weight, Kaggle-runnable, fully zero-shot system, and submit a reproducible 4-page report by **Nov 15, 2026 AoE**.

## Timeline
| Dates | Phase | Exit criterion |
|---|---|---|
| Oct 27–28 | P0 infra | data local, 17 accounts verified, pipeline + fleet + judge kernels run end-to-end |
| **Oct 31** | **Team registration (user)** | form submitted |
| Oct 28–30 | P1 proxy dev set | `Competition Data/proxy_dev/` (~500 samples, VistaQA format), judged baseline on it |
| Oct 30–Nov 7 | P2 system development (proxy dev only) | frozen config + ablation table |
| Nov 7–9 | P3 final VistaQA run (pre-registered) | `runs/final/preds.jsonl` + official `evaluation_results.json` |
| Nov 9–14 | P4 report + release | PDF on OpenReview, code + preds + eval public |

## P0 — Infrastructure (in progress)
- [x] Read challenge site, VistaQA site, paper, eval repo.
- [x] `Scripts/kgl.py` multi-account Kaggle CLI (17/18 accounts OK); `Scripts/fleet.py` shard/push/status/collect.
- [x] Probe: Kaggle = T4×2 fp16, 31 GB RAM, internet, transformers 5.16.1.
- [x] `src/pipeline.py` v0: VLM JSON {answer, evidence boxes 0–1000} → SAM2.1-L box prompts → RLE.
- [ ] Download `mozhina/VistaQA` → `Competition Data/VistaQA/` (1,157 json + images).
- [ ] Smoke-test kernels (12 samples) for Qwen3-VL-8B and Qwen3.5-9B: runtime, parse rate, format only.
- [ ] Judge kernel: official `VistaQA-Evaluation` with Qwen2.5-14B-Instruct served locally
      (vLLM on T4×2 with AWQ/GPTQ, fallback transformers + monkeypatched `_call_openai`). Validate on a
      synthetic pred file (GT-as-pred must score 100).
- [ ] SAM3 access: user HF token with `facebook/sam3` approval → private Kaggle dataset of weights, or token in kernel.

## P1 — External proxy dev set (the only thing we tune on)
Same JSON schema as VistaQA so the pipeline/eval run unchanged. Mirror the published *task/statistics* (not samples):
~27% hallucination, mostly single-instance, counting long tail.
| Task | Source (external) | Template / construction | GT masks |
|---|---|---|---|
| counting | COCO val2017 instances | "How many {cat}s are there?" (2–15 non-crowd) | all instances |
| identification | COCO val2017 | "What kind of {supercat} is in the image?" (exactly one instance in supercat) | that instance |
| spatial | COCO val2017 | "Is the {A} to the left or right of the {B}?" (unique A, B, clear gap) | A and B |
| comparison | COCO val2017 | "Which looks larger, the {A} or the {B}?" (area ratio > 2) | A and B |
| attribute | PACO-LVIS val (color/material attrs) | "What color is the {cat}?" | the object |
| ocr | TextOCR / COCO-Text v2 | "What does the text on the {obj} say?" | word region(s) |
| hallucination | COCO val2017 (POPE-style co-occurring absent cats) | count/attribute/where questions on absent {cat} | none |
Build script: `src/build_proxy_dev.py` (deterministic seed). Domains beyond indoor/outdoor (AV, robotics, science, math)
are a known gap — add only if a free, login-less source is found (e.g. nuImages/Cityscapes need accounts).

## P2 — System development (all on proxy dev; log every run in EXPERIMENTS.md)
Baseline = Qwen3-VL-8B + SAM2.1-L boxes, prompt v0. Then, in order of expected GROVE gain:
1. **Backbone**: Qwen3-VL-8B vs **Qwen3.5-9B** (RefCOCO 89.7, CountBench 97.2, OCRBench 89.2) vs Qwen3.5-27B /
   Qwen3.6-27B / Qwen3.6-35B-A3B in 4-bit (fit T4×2; slower → shard over more accounts).
2. **Segmenter**: SAM2.1-L box vs SAM3 box vs SAM3 text-concept + VLM-box filtering (helps dense counting).
3. **Hallucination calibration** (27% of score, 1.0 each when right): explicit existence check pass;
   agree VLM "absent" with SAM3 presence score; answer template "There is no X in the image." + empty masks.
4. **Answer–evidence consistency**: count in answer == #masks; dedupe overlapping masks (IoU>0.7);
   drop tiny/degenerate masks; never emit masks when answer says absent.
5. **Evidence policy** for spatial/comparison (box all referenced objects) and OCR (box text lines).
6. **Thinking mode / self-consistency**: Qwen3.5 thinking on, k-sample answer vote + box fusion (costly → more shards).
7. **Zoom-in second pass** for small text/objects: crop around boxes, re-ask, re-box.
8. Answer phrasing (short complete sentence) — check judge sensitivity on proxy dev only.
Freeze: best proxy-dev GROVE config, written into EXPERIMENTS.md as `FINAL` **before** any VistaQA scoring.

## P3 — Final VistaQA evaluation
- `fleet.py push final --accounts <8 accounts> --cfg <FINAL>`; collect; validate 1,157 unique ids; run official eval
  with Qwen2.5-14B-Instruct judge; save `evaluation_results.json` + per-task/domain/hallucination breakdown.
- Also run (as reported analysis, not selection): paper-style baseline (Qwen3-VL-4B + SAM) for a direct comparison.

## P4 — Report (≤4 pages, workshop format, refs/appendix excluded)
Team/authors · architecture · external models/data (all disclosed) · prompting/tools · zero-shot protocol
(proxy dev, pre-registration) · VistaQA GROVE + breakdowns + text-acc/mIoU · ablations on proxy dev ·
links: GitHub repo, predictions + eval json · reproducibility (Kaggle T4×2, exact model revisions, seeds).

## Escalation (ask user before spending)
If open models on T4 plateau below ~45: options are API MLLMs (GPT-5.x-thinking / Gemini 3) + SAM3 — the paper's
best — or rented A100/H100 for 27B–122B open models in bf16 with thinking.

## Re-download commands
```bash
python -c "from huggingface_hub import snapshot_download as s; s('mozhina/VistaQA', repo_type='dataset', local_dir='Competition Data')"
git clone https://github.com/VistaQA/VistaQA-Evaluation "Competition Data/VistaQA-Evaluation"
```
