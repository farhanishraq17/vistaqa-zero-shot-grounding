# EXPERIMENTS — run log

Rules: CLAUDE.md "Zero-shot rule". Dev decisions come only from **proxy dev** (`Competition Data/proxy_dev`, 560
samples, external). VistaQA rows are smoke (format only), harness checks, or pre-registered final runs.

Proxy dev: v1 = ids 1–420 (COCO val2017: counting 100, identification 70, spatial 70, comparison 60, hallucination 120);
v2 adds ids 421–560 (attribute 60 PACO-LVIS, OCR 50 COCO-Text v2, hallucination +30). Total hallucination 150 (26.8%).
Built by `python src/build_proxy_dev.py` (seed 0). `devall-X` = merge of `dev-X` (1–420) + `dev2-X` (421–560).

## Harness checks
| Run | What | Result |
|---|---|---|
| smoke-q3vl8b / smoke-q35-9b | VistaQA 12 samples, format only | parse 12/12 and 11/12, masks full-res. Not scored. |
| smoke-prem/ans/seg/cpt → smoke-final | VistaQA first 12 samples through all 4 candidate passes + merge | format only: 12 unique ids, 0 bad masks, 0 empty answers; 27B 33 s/it, premise 8.6 s/it, SAM3 box 2.7 s/it. **Not scored.** |
| oracle | VistaQA GT-as-pred, judge Qwen2.5-14B-Instruct-AWQ (vLLM, T4×2) | **GROVE 100.00**, text 100, mIoU 100 ✔ |
| oracle-shift | VistaQA, each answer replaced by the next sample's | text acc **1.21%**, GROVE 11.09 (≈ √0.01 floor) ✔ judge rejects wrong answers |

## Proxy dev runs
| Run | Config | Accounts | GROVE | text | mIoU | notes |
|---|---|---|---|---|---|---|
| devall-q3vl8b | Qwen3-VL-8B + SAM2.1-L, single-pass prompt v0 | K01,K02,K09 | **58.60** | 66.4 | 64.3 | 6.3 s/it. halluc text 52% (answers false-premise Qs: "The person is next to a dog"); halluc-with-masks 58/150; counting answer#==GT 64%, box#==GT 38% |
| devall-q3vl4b | Qwen3-VL-4B + SAM2.1-L, v0 | K07,K08,K11 | **39.02** | 55.7 | 50.3 | 4.8 s/it. terse answers ("right", "bed") judged incomplete → spatial text 20%, comparison 22%; 135 grounded samples without mask |
| dev-q35-9b | Qwen3.5-9B (no think) + SAM2.1-L, v0 (v2 run covered all 560) | K03,K04,K10 | **62.14** | 74.5 | 62.8 | ~50 s/it (slower than the 27B on T4); halluc 76.7 (text 77%), counting 47.6, comparison 50.8 → not used |
| dev-q36-27b | Qwen3.6-27B nf4 + SAM2.1-L (CPU), v0 | K12,K13,K14,K15 | **64.91** | 71.2 | 68.6 | 31 s/it (v1 SAM OOM, v2 load fail, v3 ok). spatial 84.8 (text 97%), comparison 74.4, ocr 52.9; halluc 58.0 (text 59%, 63/150 with masks); parse fail 14 |
| dev-ts-q3vl8b | Qwen3-VL-8B, **two_stage** (reason_v1 → native GROUND per target) | K01,K02,K09 | 58.56 | 65.0 | 67.6 | 38 s/it. halluc 53.3→64.2, comparison mIoU 50→71 (boxes both objects), but spatial text 74→34% (terse "right"), ocr mIoU 65→46 |
| dev-tsv-q3vl8b | two_stage + **verify** (yes/no existence per target) | K07,K08,K11 | 57.41 | 64.8 | 65.9 | verify ≈ no-op on halluc (model says yes to its own targets), hurts counting |
| dev-tsv-q3vl4b | Qwen3-VL-4B two_stage + verify | K16,K17 | 50.20 | 58.0 | 64.7 | 8.5 s/it. halluc 64.5 (text 72%); comparison/spatial text 17% (terse) |

| dev-es-q3vl8b | 8B v0 + expand_short | K01,K02,K09 | **62.31** | 71.6 | 64.3 | +3.7 over v0 58.60 (296/560 answers were ≤3 words) |
| prem-q3vl8b | 8B premise check only (`premise: "only"`) | K07,K08 | — | | | halluc caught 148/150 but flags 105/410 grounded; **+ yes/no confirm: 138/150, 17/410 false** |
| prem-q36-27b | 27B premise check only | K12..K15 | — | | | 142/150, 57 false; + confirm 135/150, 36 false (27B says "no" too easily) |
| pc8-q3vl8b | devall-q3vl8b answers, overridden by prem-q3vl8b confirmed misses (`fleet.py override --confirm`) | K05 (judge) | **67.32** | 75.0 | 71.3 | +8.7 over 58.60; halluc 53.3→91.5 |
| pc8-q36-27b | dev-q36-27b answers + prem-q3vl8b confirmed misses | K06 (judge) | **71.74** | 77.0 | 76.0 | +6.8 over 64.91; halluc 58.0→91.5; spatial 84.8→73.6 (11 false alarms on small objects) |
| pc27-q36-27b | dev-q36-27b answers + its own prem-q36-27b confirmed misses | K16 (judge) | 69.60 | 74.3 | 74.0 | single-model variant; 36 false alarms → spatial 66.6, comparison 61.1. 8B premise wins (+2.1) |
| pc8-es-q3vl8b | 8B v0 + **expand_short** (dev-es-q3vl8b, 296 answers rewritten) + 8B premise confirm | K06 (judge) | **70.00** | 78.8 | 71.3 | expand_short +2.7 over pc8-q3vl8b (spatial 58→69, ocr 45→59, identification 73→76); 8B is 5× faster than 27B |
| pc8-es-q36-27b | 27B v0 + expand_short (dev-es-q36-27b: identical to dev-q36-27b except the 29 short answers → deterministic) + 8B premise confirm | K05 (judge) | **72.29** | 77.7 | 76.0 | **best so far**; +0.55 over pc8-q36-27b, ocr 52.9→58.9 |
| cpt-sam3-q36-27b | SAM3 **text-prompt** (concept) masks for counting Qs (regex how many/count/number of; prompt = 27B's most common evidence label), scores kept | K17 | — | | | 98 counting Qs: exact count 27B 60, SAM3 τ0.6 65, **max(27B, SAM3 τ0.7) 67**; S 0.673 → 0.687 |
| cf-pc8-es-q36-27b | pc8-es-q36-27b + `countfix` τ=0.7 (`pipeline.merge_count`: confident SAM3 instances may only *raise* the count; number in answer rewritten, SAM3 masks) | K06 (judge) | **73.04** | 78.6 | 76.3 | 25 counts raised, correct 7→12; counting 56.7→61.0; 0 judge flips on unchanged answers. τ picked from 5 values on these 98 Qs |
| s3cf-pc8-es-q36-27b | same, but box masks from SAM3 too (seg-sam3-q36-27b) | — (verdicts reused) | **73.05** | 78.6 | 76.4 | = all-SAM3 system; ties SAM2.1 → keep one segmenter |
| feas-q36-35b | Qwen3.6-35B-A3B nf4, 24 samples (feasibility) | K12 | — | | | **load fails**: bnb does not quantize the fused MoE experts → ~70 GB, "modules dispatched on CPU". Dropped (only FP8/GPTQ releases; no T4 path) |
| cf2-pc8-es-q36-27b / cf2s3-es-q36-27b | countfix + **grammar fix** when raising from 1 ("There is 2 chair" → "There are 2 chairs"; 2 right numbers were judged wrong) | K05 (judge) / cache | **73.29 / 73.31** | 78.9 | 76.4 | counting 61.0→62.4; cf2s3 = all-SAM3, system order, **best** |
| cf3-s3-es-q36-27b | countfix with **3 counters**: max(27B, 8B answer count from dev-es-q3vl8b, SAM3 instances ≥ 0.8) | K06 (judge) | **73.81** | 79.8 | 76.5 | **best**; counting 62.8→65.7 (text 68→73%). Exact counts on 98 Qs: 27B 60, 8B 63, max(27B,SAM3.7) 67, max3@.8 72. τ from {0.6,0.7,0.8} on these Qs. Needs an 8B answer pass on VistaQA (~6 s/it) |
| veto-sam3 → cfvt0.5/0.6/0.7 | SAM3 concept on the premise's "missing" phrase; keep the 27B answer if SAM3 finds it ≥ τ | K08 / cache | 73.16 / 73.18 / 73.18 | 79.1 | 76.2 | confirm 138 hit/17 FA → τ0.7 136/13, τ0.5 133/12; text +0.2 but GROVE −0.13 → dropped |
| prem2-q3vl8b | 8B premise, confirm prompt v2 ("look carefully … small, distant or partially visible"), 253 flagged ids | K07 | — | | | 133 hit / 16 FA vs v1 138 / 17 → keep v1 |
| OCR filled-box masks (offline) | rectangles from the 27B boxes instead of SAM3 masks on text questions | — | — | | | S 0.649 vs SAM3 0.696 on 49 OCR Qs → keep SAM3 |
| `judge.py cache` | verdict reuse for identical (id, question, answer) from judged proxy runs | — | | | | reproduces s3cf 73.05 exactly ✔ (judge greedy, 0 flips seen on proxy dev; 3/802 across kernels on VistaQA) |
| seg-ctl-q36-27b → pcctl | SAM2.1 re-run on dev-q36-27b boxes (`--boxes`, no VLM), remasked onto pc8-q36-27b | K17 | 71.74 | | 76.0 | control: reproduces 71.74 (548/560 identical; rest = GPU vs CPU SAM) ✔ tooling |
| seg-mm-q36-27b → pcmm | same, SAM2.1 `multimask` (best of 3 by predicted IoU) | K10 | 71.53 | | 75.4 | worse than single-mask → keep single |
| seg-sam3-q36-27b → pcsam3 | **SAM3** (tracker head, box prompts; Kaggle mirror `chonlanawawa/sam3` = facebook/sam3, SHA-256 of model.safetensors and sam3.pt verified identical) on dev-q36-27b boxes | K08 | 71.72 | | 76.1 | tie with SAM2.1 (71.74): ocr/comparison/counting +0.5–1.3, attribute/identification −0.7–1.0. 3.4 s/it. COCO polygon GT too coarse to separate them |
| speed-q35-9b-fla | Qwen3.5-9B + flash-linear-attention, 24 samples | K10 | — | | | **142 s/it** (vs 11 s/it 8B on same ids): fla Triton kernels slower on T4 → Qwen3.5-9B dropped |
| speed-q36-27b-think | 27B thinking, 16 counting samples | K11 | — | | | **311 s/it** median (max 1027, 10× no-think); exact count 8/16 vs 9/16 without thinking → thinking dropped |
| probe-prem-q3vl8b | **premise probe** (MathVista testmini: 200 own-image Qs + 100 Qs moved to another context's image; `src/build_premise_probe.py`) — 8B premise + confirm | K07 | — | | | moved Qs caught **86/100**; false alarms **21/200**: charts/diagrams/tables/docs 6/155 (≈4%, same as COCO 17/410), natural-image "age gap" 9/15 (unobservable attribute), CLEVR "subtract all X" 6/15. Premise check transfers to math/science figures |
| zoom2-q36-27b → cfz-s3-es-q36-27b | 27B **OCR zoom re-read** (text Qs: crop of the pass-2 boxes, ×≤4, "read exactly as written"; zoom before expand_short) on the cf3 chain | 4 accts / K06 (judge) | 73.96 | 80.0 | 76.5 | +0.15 = **1 net sample**: 42 answers changed, 4 right / 3 wrong (one only lost quote marks → judge flip), ocr 59.8→61.4. Noise level, extra 27B pass → not adopted |
| cp-* (count probe) | **count probe** (`src/build_count_probe.py`: 168 MathVista testmini free-form integer Qs matching COUNT_Q — CLEVR "subtract all", tables, charts, plots, photos; text GT only): 27B pass 2 + SAM3 box + SAM3 concept + 8B, same merge chain | K12,K13,K14 / K01 / K08 / K17 / K04, K05 (judge) | — | | | judge text acc: 27B 75.00, cf3 rule 70.83, guarded 75.60 (cache). exact count: 27B 114, 8B 88; **countfix max-of-3 (cf3 rule) 108**: 34 raised, 11 fixed, 17 broken — all on reasoning counts ("Subtract all X. How many are left?", "mode of the number of …") where visible instances ≠ answer. 8B: 84 s/it here (long generations) |
| cf3 + `plain_count` guard | countfix only on plain counts (one sentence ≤12 words, COUNT_Q, no arithmetic/condition/statistics words) | — | **73.81** | | | COCO: all 119 counting Qs plain → cf3 preds byte-identical. Probe: 2 raised, **115** (vs 108 unguarded, 114 none) → **adopted** |
| op-* (OCR probe) | **OCR probe** (`src/build_ocr_probe.py`: 225 TextVQA val reading Qs = READ_Q (the zoom trigger), top answer ≥ 5/10 votes; text GT only): 27B pass 2 with / without `zoom` | K12,K13,K14 / K15,K16,K10 / K04, K05 (judge) | — | | | judge text acc: **no zoom 76.44, zoom 75.11** (GT-substring 196 vs 190). Zoom rewrote 135/225, mostly case; real changes misread ("tua"→"cua") as the crop loses context → **zoom dropped** (COCO-Text +0.15 was noise) |
| dev-th-q36-27b → cf3-thf | 27B **thinking** (max_new_tokens 4096) as pass 2; 34/560 traces never closed → no-think fallback (greedy = dev-es-q36-27b lines); SAM3 boxes (seg-sam3-th), same premise override + guarded countfix | 15 accts / K08,K17 / K06 (judge) | 73.64 | 80.0 | 75.9 | vs cf3 73.81: text +1 sample, mIoU −0.6. comparison 72.5→76.1, spatial 73.1→75.4, halluc 91.5→92.2 but counting 65.7→62.5, ocr 59.8→54.7. Median 102 s/it, 29.6 GPU-h (3× no-think) → **dropped** |
| concept-mask swap (offline) | on plain counting Qs whose final count = #SAM3 concept instances ≥ τ, use those masks instead of box masks | — | +0.05…+0.16 | | | τ 0.5–0.8: 43–55 swapped, S better ≈ worse (e.g. 18/24 at .8) → box and concept masks equally good; dropped (and box→concept snapping skipped) |
| premise × answer agreement (offline, cached verdicts) | override only if the independent 8B / 27B answers also say absent | — | | | | flagged & both answer normally: 43 halluc vs 16 grounded → requiring agreement loses 43 hits to save 16 FAs; 11 unflagged halluc answered normally by both → no signal left |
| think vs no-think oracle (offline) | per-sample correctness cf3 vs cf3-thf | — | | | | both right 432, both wrong 97, only no-think 15, only think 16 — spread over all tasks, no selector signal → no ensemble |
| op-prem-q3vl8b | 8B premise + confirm on the OCR probe (all premises valid) | K07 | — | | | false alarms **8/225 (3.6%)** ≈ COCO 17/410 → no change |
| pcb → cfb-s3-es-q36-27b | premise = 8B ∧ 27B confirmed (both must flag) on the cf2s3 recipe | cache | 73.39 | 79.1 | 76.4 | vs cf2s3 73.31 (+0.08): halluc 91.5→90.2, spatial 73.1→76.5. Noise, extra 27B premise pass → dropped |

## Candidate final system (proxy dev 73.81, cf3-s3-es-q36-27b) — not yet frozen
Passes (each a fleet run, merged locally in this order; `remask → override → countfix` reproduces cf3 exactly):
1. **premise**: Qwen3-VL-8B, `premise: "only"` (missing-premise JSON + POPE yes/no confirm v1).
2. **answer**: Qwen3.6-27B nf4, prompt v0, `expand_short`, `seg: ""` (boxes only, max_memory 14+14 GiB).
3. **seg**: SAM3 tracker on pass-2 boxes (`--boxes`), Kaggle mirror of facebook/sam3.
4. **concept**: SAM3 text prompt (pass-2 top evidence label) on counting questions (`--concepts`).
5. **count**: Qwen3-VL-8B answer pass, prompt v0, `expand_short` (only its counts are used).
Merge: `fleet.py remask` (2+3) → `override --confirm` (1) → `countfix --tau 0.8` (4, 5; plain counting Qs only).

## FINAL — pre-registered 2026-11-02, before any VistaQA scoring
Frozen on proxy dev at **73.81** (cf3-s3-es-q36-27b + `plain_count` guard; byte-identical preds). Code = the commit that adds
this section. The leaderboard entry is run **`final`**, whatever it scores.

Passes on VistaQA v1.1 (`data: hf:mozhina/VistaQA`; the kernel reads image + question only):
| Run | Config |
|---|---|
| final-prem | `{"vlm": "Qwen/Qwen3-VL-8B-Instruct", "premise": "only"}` |
| final-ans | `{"vlm": "Qwen/Qwen3.6-27B", "quant": "nf4", "pip": ["bitsandbytes"], "max_memory": ["14GiB", "14GiB"], "seg": "", "expand_short": true}` |
| final-cnt | `{"vlm": "Qwen/Qwen3-VL-8B-Instruct", "seg": "", "expand_short": true, "plain_only": true}` (only plain counting Qs: its other answers are never used, per-sample outputs unchanged) |
| final-seg | `--boxes final-ans --models chonlanawawa/sam3/pyTorch/default/1`, `{"seg": "kaggle:chonlanawawa/sam3/pyTorch/default"}` |
| final-cpt | `--concepts final-ans --models chonlanawawa/sam3/pyTorch/default/1`, same cfg |

Merge:
```
fleet.py remask   final-s3 --parts final-ans final-seg
fleet.py override final-pc --parts final-s3 final-prem --confirm
fleet.py countfix final    --parts final-pc final-cpt final-cnt --tau 0.8
fleet.py check    final    --inline "Competition Data/VistaQA"
```
Scoring: `judge.py push final --account X`, `judge.py score final` (official mask scorer and judge prompt/parser;
Qwen2.5-14B-Instruct-AWQ on vLLM). Per-task / per-domain breakdowns are reported as they come.

Reported as analysis only, never to pick the entry: `final-s3` (no premise check, no count raise) and `final-pc` (no count
raise), scored the same way, for the ablation table. Paper baselines are quoted from the paper, not re-run.

Rules: re-runs only for infrastructure failures (kernel crash, OOM, missing shard), same config. No prompt, threshold or
merge change after this point. VistaQA failures are not inspected for development. Any later VistaQA run gets its own
pre-registered row and cannot replace `final`.

Amendment 2026-11-02, before any scoring (format fix, no prompt/threshold/merge change): `fleet.py check final` flagged 3
masks of the wrong size. The image files of ids 138 and 496 are smaller than the width/height their JSON declares, so SAM3
masks came out at the file's size and the official scorer cannot IoU them. `fleet.py fit final-seg` and `fit final-cpt`
(`--inline "Competition Data/VistaQA"`) resize those masks to the declared [H, W] (nearest; reads image meta only), then the
merge above is re-run unchanged.

Result: **`final` 48.40** (text 60.76, mIoU 52.05). Ablations, analysis only: final-s3 45.46, final-pc 48.95 (verdicts reused
from final, and final-s3 for the 23 answers before the raise). The count raise lost 0.55 on VistaQA (10 right → wrong, 2 wrong →
right, 11 wrong both ways) after +1.5 on proxy dev; the entry stays `final` as pre-registered. Judges: K04 (final), K05
(final-s3). See RESULTS.md.

Correction 2026-11-03: the first write-up said final-pc 48.89, −0.49, 11/4 and +0.5 on proxy dev. The cache had taken final-s3's
verdicts on identical answers, and 3 of them flip between judge kernels (3/802, so the "0 flips" of the `judge.py cache` row does not hold on VistaQA);
+0.5 was only the third counter's step, not the whole raise. `final` itself did not change.

## After FINAL: scale-up (analysis only, the entry stays `final`)
Started 2026-11-03, after `final` was scored and its results were read. So everything below is informed by VistaQA aggregate
numbers (false alarms, over-masking, weak domains), and the report has to say that. No VistaQA failure inspection from here on.
A new system is frozen on the dev sets below, pre-registered, then scored once as analysis; it cannot replace `final`.

Dev sets: **proxy dev** (COCO, 560) and **proxy2** (`src/build_proxy2.py`, fixed recipe, seeds in the code; 580 samples, 158
hallucination: math 150 synthetic, robotics 150 BOP YCB-V/LM-O, science 130 Data Science Bowl 2018 nuclei, AV 150 BDD100K). Answers
and masks come from source annotations or the drawing code; AV masks are SAM3 on annotated boxes, so AV measures which objects
were found, not mask quality. VistaQA's image sources are avoided. GT stays on the laptop; machines and kernels get images + questions (and, for the AV mask step only, the annotated boxes).

Compute added: 2 shared PCs (PC1 2x RTX 3090, PC2 1x RTX 3090, other people's jobs on them; `fleet.py push --accounts PC1:1+0 --need 13 6`).
Their internet is ~3 MB/s in total and PyPI 30-70 KB/s, so the env is a local copy (torch 2.8 + transformers 5.16.1, Kaggle has torch 2.11).
Private per-account Kaggle datasets work through `kagglehub.dataset_upload` (the CLI's create still fails on Windows).

| Run | What | Accounts | GROVE | Text | mIoU | Notes |
|---|---|---|---|---|---|---|
| dev-q38-27b → cf3b-q38-27b | Qwen3.8-27B nf4 through the cf3 chain (seg-q38-27b, prem-q3vl8b, cpt2-q38-27b, dev-es-q3vl8b) | K12,K13,K14,K15,K16,K11 / K01,K02 / K17 (judge) | **74.70** | 81.6 | 76.1 | vs cf3 73.81: +0.89, 95% CI −0.62 to +2.49; right 21 gained, 11 lost. 17 s/sample vs 32. 534/560 valid JSON (bad brackets); `parse_output` now keeps the labels, which gave 17 more concept prompts and 2 right counts (cf3-q38-27b without that: 74.46) |
| dev-gm4-31b → cf3-gm4-31b | Gemma 4 31B nf4, same chain. Its boxes are [y1,x1,y2,x2] (mean best-box IoU with GT 0.71, vs 0.27 read as xyxy), swapped before SAM3 | K03,K07,K09,K17 / K09 (judge) | 69.77 | 79.5 | 70.7 | text level with 27B but OCR 25.3 (masks) → not adopted. 3-model majority on grounded: 316/410, below Qwen3.8 alone 320 (oracle 338) → no vote |
| cf3-judgefull | cf3 preds re-judged by unquantized Qwen2.5-14B (fp16 T4x2, `max_model_len` 1024, 1 seq, eager) vs AWQ | K05 | 73.33 | 79.1 | 76.5 | AWQ 73.81 / 79.8. 8 of 560 verdicts flip, 6 down and 2 up: the full judge is stricter on OCR casing and near-synonyms |
| p2-prem | 8B premise + confirm on proxy2 (= final-prem) | K12,K08 | — | | | caught 151/158 (96%); false alarms 39/422 (9.2%): robotics 18.2, AV 13.6, math 2.9, science 1.0. VistaQA was 71% / 13% (robotics 34), proxy dev 92% / 4%: proxy2 is a better false-alarm test, but its hallucination Qs are too easy for recall |
| p2-q36-27b → p2-s3/pc/cf3-q36 | the entry's recipe on proxy2 (Qwen3.6-27B, SAM3 on PC2, 8B premise, count raise) | K12,K08,K13,K15 / PC2 / K17,K09 (judge) | s3 55.70, pc 57.53 | 75.2 | 55.3 | baseline for proxy2. First runs on the PC route (`--accounts PC2:0`): SAM3 0.3-0.7 s/sample on a 3090 |
| p2-q38-27b → p2-s3/pc/cf3-q38 | Qwen3.8-27B, same chain | as p2-q36 | s3 57.42, pc **61.04**, cf3 60.80 | 77.1 | 59.5 | Qwen3.8 +3.5 over 3.6 at pc (+0.9 on proxy dev). Premise: halluc 66→93, grounded 54.2→48.8 |
| mask clean-up rules (offline) | `maskfx.py` dedupe (IoU .5/.7/.9), containment, tiny masks on p2-pc-q38 | — | 61.04 | | | IoU rules change no sample, containment 1, tiny-mask drop −0.28 → none adopted |
| count rules (offline) | exact-count rate of max(27B, 8B, SAM3) vs median vs 8B∧SAM3 agreement, plain counting Qs | — | — | | | max: proxy dev +12/+14 of 99/95, proxy2 0/+5 of 85/84; median and agreement ≈ base → keep max |
| **count raise bug** | the entry's 8B count run has `seg: ""`, and `merge_count` takes the winning counter's masks: a raise by the 8B leaves no masks. proxy2: p2-cf3-q38 raised 19, 11 with no masks. VistaQA `final` (own predictions only, no GT): 11 of 23 raised answers lost their masks | — | | | | proxy dev's 8B run had SAM2.1 masks, so it never showed. `countfix --keep-masks`: a raise by a mask-less counter keeps the larger of the SAM3-concept and base mask sets |
| p2-cfk-q38 | p2-pc-q38 + count raise with `--keep-masks` | cache | **61.60** | 77.2 | 60.9 | vs pc 61.04: +0.56 (without the fix −0.24) |
| p2-prem-q38 | Qwen3.8 premise + confirm on the 190 ids the 8B flagged on proxy2 | K04,K16 | — | | | 8B∧Q3.8: hits 139/158 (−12, math −11), false 28/422 (−11) → worse trade than the SAM3 veto, dropped |
| p2-veto → p2-cfv{τ}-q38 | SAM3 concept on the 8B's "missing" phrase (PC2), keep the answer if SAM3 finds it ≥ τ; + count raise `--keep-masks` | PC2 / cache | τ .3 63.03, **.4 63.27**, .5 62.64, .6 62.50, .7 62.16, none 61.60 | 79.3 (τ .4) | 62.4 | hits/false: none 151/39, .5 146/26, .4 144/21, .3 140/18 (of 158/422) |
| cfv{τ}-q38 (proxy dev) | same veto (veto-sam3, prem-q3vl8b) on the Qwen3.8 chain + `--keep-masks` (8B count masks dropped as in the entry: dev-cnt-nomask) | cache / K17 (judge, 6 unjudged answers at τ .3) | τ .3 74.07, .4 74.23, .5 74.30, .6 74.49, .7 74.49, none **74.55** | 81.4 | 76.2 | the veto costs a little on COCO (few false alarms to remove). **Rule: τ maximises GROVE pooled over both dev sets (1,140 samples)**: none 67.96, .7 68.22, .6 68.39, .5 68.37, **.4 68.65**, .3 68.45 → τ = 0.4 |
| p2-v1-q38 → p2-s3-v1 | Qwen3.8, **prompt v1** (evidence = the element that shows the answer: the bar or slice, not its label or legend; the cell, not headers; the shape, not its side lengths), SAM3 on PC1 | K12,K08,K13,K15 / PC1 / K09 (judge) | 57.64 (v0 57.42) | 71.4 (73.8) | 59.5 (57.6) | math 68.0→78.0, av 62.4→61.2, robotics 56.7→52.6, science 40.4→35.8. With premise + veto τ.4: v1 63.75 vs v0 62.71, but text 76.9 vs 78.4: 12 verdicts lost, 3 gained (sign test p≈0.04) |
| p2-hyb-all / p2-hyb-nocount (offline) | v0 answers (and verdicts) with v1's masks, after premise + veto τ.4; nocount keeps v0 masks on COUNT_Q | cache | **64.22** / 64.11 | 78.4 | 64.9 | v1 masks help math only: math 67.9→78.0, robotics 66.1→64.7, science 44.7→41.8, av 69.7→69.5. Perfect routing of figures would give ≈65.3 → test a figure router |
| s3-es-q36-27b (proxy dev) | the entry's chain without premise check and count raise (A+B), for the report's ablation | K05 (judge: 8 answers never judged before) | 65.07 | 71.6 | 68.6 | A+B 65.07 → +C 72.27 → +D 73.81 |
| rt-dev / p2-rt / mp-rt / op-rt | **figure router**: Qwen3-VL-8B, `route: true` ("Is this image a chart, plot, table, diagram or geometric figure, rather than a photograph or a rendering of a real scene?") | K03 / K08 / K17 / K09 | — | | | yes: proxy dev 0/560, proxy2 math 150/150, AV 0/150, robotics 0/150, science 2/130; MathVista probe 199/200; TextVQA photos 11/225 |
| **p2-v2** | Qwen3.8 v0 answers + SAM3 masks; routed figures take prompt v1's boxes (`remask --route`); 8B premise + confirm + SAM3 veto τ .4; count raise `--keep-masks` | cache | **65.87** | 79.3 | 65.5 | vs the entry's recipe on proxy2 (p2-cf3-q36) 57.43: +8.44. av 71.5, math 77.4, robotics 67.1, science 44.7. On proxy dev the router routes nothing, so v2 = cfv0.4-q38 = **74.23** (entry 73.81) |

### Judge cross-check of `final` — pre-registered 2026-11-04, before running
The paper names Qwen 2.5-14B; the entry was judged with the AWQ 4-bit build of it, as pre-registered. This re-judges unchanged
predictions with the unquantized weights. It changes no prediction, prompt, threshold or merge and the entry stays `final`.
- Predictions: `runs/final/preds.jsonl`, copied byte for byte to `runs/final-fulljudge/` (sha256 starts 30d35f539f8aaafc); `final-s3` the same way to `runs/final-s3-fulljudge/`, for the ablation.
- Judge: `Qwen/Qwen2.5-14B-Instruct`, official prompt and parser, temperature 0, `max_tokens` 256, vLLM on Kaggle T4x2, fp16.
  `--jcfg {"llm": {"gpu_memory_utilization": 0.98, "max_model_len": 1024, "max_num_batched_tokens": 768, "max_num_seqs": 1, "enforce_eager": true}}`
  (full weights only just fit T4x2; longest VistaQA prompt ~630 tokens + 256 output). Same config as cf3-judgefull on proxy dev.
- Masks: official scorer, unchanged. Scored with `judge.py score`.
- Reported beside 48.40 whichever way it moves, as the number closest to the paper's judge. 48.40 stays the pre-registered score of the entry. Failures of this run are not inspected for development.
- **Result** (2026-11-04, judges K05, K03): `final` **48.14** (text 60.59, mIoU 52.05), `final-s3` 45.85 (60.24). 56 of 1,157
  verdicts flip on `final` (grounded 24 up / 10 down, hallucination 3 up / 19 down). See RESULTS.md.

### V2 — pre-registered 2026-11-06, before any VistaQA run of it (post-freeze analysis; the entry stays `final`)
Frozen on the two dev sets: **proxy dev 74.23** (cfv0.4-q38; the router routes none of its images) and **proxy2 65.87** (p2-v2).
The entry's recipe scores 73.81 / 57.43 there. Every choice below was made on those sets (table above). Its design was also informed
by VistaQA aggregates of `final` (false-alarm rate, over-masking, weak math/robotics), so it is reported as analysis next to the
entry and never replaces it. Thinking is not part of v2; if the math probe supports it, it gets its own registered row.

Passes on VistaQA v1.1 (`data: hf:mozhina/VistaQA`; each pass reads image + question only), Kaggle T4x2:
| Run | Config |
|---|---|
| v2-ans | `{"vlm": "Qwen/Qwen3.8-27B", "quant": "nf4", "pip": ["bitsandbytes"], "max_memory": ["14GiB", "14GiB"], "seg": "", "expand_short": true}` (HF revision 1d4bf0f2; main 8c0b3d30 differs only in README.md) |
| v2-seg | `--boxes v2-ans --models chonlanawawa/sam3/pyTorch/default/1`, `{"seg": "kaggle:chonlanawawa/sam3/pyTorch/default"}` |
| v2-cpt | `--concepts v2-ans`, same SAM3 cfg |
| v2-route | `{"vlm": "Qwen/Qwen3-VL-8B-Instruct", "route": true}` |
| v2-v1 | v2-ans cfg with `"prompt": "v1", "expand_short": false, "ids": [ids v2-route answered yes]` (only its boxes are used) |
| v2-seg1 | `--boxes v2-v1`, same SAM3 cfg |
| v2-veto | `--concepts final-prem`, same SAM3 cfg (presence veto) |
| reused | `final-prem` (8B premise + confirm) and `final-cnt` (8B count): configs identical to v2's, greedy, so not re-run |

Merge (after `fleet.py fit RUN --inline "Competition Data/VistaQA"` on v2-seg, v2-seg1 and v2-cpt, the declared-size fix of the entry):
```
fleet.py remask   v2-s3  --parts v2-ans v2-seg
fleet.py remask   v2-s3r --parts v2-s3 v2-seg1 --route v2-route
fleet.py override v2-pc  --parts v2-s3r final-prem v2-veto --confirm --veto-tau 0.4
fleet.py countfix v2     --parts v2-pc v2-cpt final-cnt --tau 0.8 --keep-masks
fleet.py check    v2     --inline "Competition Data/VistaQA"
```
Scoring: predictions committed before judging; `judge.py push v2` with the entry's AWQ judge (primary, comparable with 48.40) and
the unquantized judge with the cross-check's `--jcfg` (secondary). Reported whichever way they move.
Registered analysis rows, same judges (verdicts reused where the answer text is identical):
- `v2-s3`, `v2-s3r`, `v2-pc`: the merge stages, for an ablation.
- `final-km`: the entry's merge with `countfix --keep-masks` (fixes the mask loss of the count raise; same text as `final`).
  **Result** (2026-11-06): text identical to `final` on all 1,157, so `final`'s verdicts are reused as they are; 11 samples get
  masks back. **48.47** AWQ / **48.21** full judge, +0.07 over the entry under both: the mask bug explains little of the raise's −0.55.
- `v2-top1`: v2 with only the first mask kept on answers to non-counting questions (tests the over-masking seen in `final`'s
  aggregates; proxy dev's GT disagrees with it, so it is analysis only).
Rules: re-runs only for infrastructure failures, same config; no prompt, threshold, router or merge change after this point;
VistaQA failures of v2 are not inspected for development.
- **Result** (2026-11-06; judges K07, K12): **v2 48.01** (text 59.98, mIoU 53.08) vs entry 48.40, −0.40 (95% CI −2.37
  to +1.50). Stages: v2-s3 42.43 (final-s3 45.46), v2-s3r 42.18, v2-pc 48.25 (final-pc 48.95), v2 48.01; v2-top1 47.18.
  Premise answers: grounded 110 → 64, hallucination 225 → 189. GPU: v2-ans 9.1 h, v2-v1 2.2 h of compute on T4x2. See RESULTS.md.

### Thinking on routed figures — dev test, rule written 2026-11-06 before the full dev results
Question: should router-yes images take Qwen3.8's **thinking** answer (`"thinking": true, "max_new_tokens": 4096`, same prompt,
unclosed traces fall back to the no-think answer) while keeping v2's masks? Dev runs: MathVista probe (`src/build_math_probe.py`,
200, text GT only) `mp-q38` no-think vs `mpt-q38` + `mpt2-q38` thinking (PC1, halves of the ids); proxy2's 152 router-yes images
`p2t-q38` (Kaggle) merged into the p2-v2 chain as `p2-v2t`. Written after a partial look at the probe (53 of 200 thinking outputs,
numeric match 44 vs 35 no-think), before any judge verdict of a thinking answer.
**Rule**: adopt (and pre-register a separate VistaQA row `v2t`) only if (a) judge text accuracy pooled over the 352 dev figure
questions rises and (b) proxy2 GROVE of p2-v2t is not below p2-v2 (65.87). Otherwise thinking is reported as a dev negative.
v2 itself is unchanged either way.
- **p2t-q38 → p2-v2t** (K08, K11, K06, K01; judge K11): all 152 traces closed, median 43 s/it on T4x2 (2.7 GPU-h);
  38 answers changed, 1 right gained, 3 lost (routed text 142 → 140 of 152; identification 80.3 → 69.6, comparison 81.1 → 84.6).
  **GROVE 65.42** (own kernel; 1 judge flip on an unchanged answer), 65.57 with p2-v2's verdicts on unchanged answers: below
  65.87 either way, so **(b) fails and thinking is not adopted**; no `v2t` row.
- **mpt-q38 + mpt2-q38 → mp-think** (PC1 GPU1 / GPU0; judge K01): 182 traces closed, 18 hit 4096 tokens and keep the no-think
  answer; ~60 s/it on a 3090. Text acc **61.0 → 71.0** (36 gained, 16 lost): SciBench-style physics 0 → 7 of 22, puzzles 1 → 7,
  geometry 7 → 11, tables 16 → 19; function plots 16 → 13, bar charts 41 → 40. Pooled over the 352 dev figures 264 → 282, so (a)
  passes, but (b) failed: thinking helps multi-step reasoning, not chart reading. Not adopted, as the rule says.

### Reasoning router + v3 — rule and registration written 2026-11-06, before any v2 verdict and before the router has run
Dev result above: thinking gains on multi-step questions (physics, puzzles, geometry, tables) and loses on reading values off a
chart. Test, one prompt, one shot: Qwen3.8 (no thinking) answers `pipeline.REASON` ("does answering it need several steps of
calculation or reasoning ... rather than reading a value, label or count directly from the image?") per figure question
(`"route": "reason"`); only its yes-questions take the thinking answer.
- Dev runs: `rr-mp` (MathVista probe, 200) and `rr-p2` (proxy2's 152 router-yes ids), cfg `{"vlm": "Qwen/Qwen3.8-27B", "quant": "nf4",
  "seg": "", "route": "reason"}`. Evaluated offline with verdicts already in hand: probe = mp-think's verdict on yes-ids, mp-q38's
  otherwise; proxy2 = `p2-v2r` (the p2-v2 chain with `think` from p2t-q38 on yes-ids only; p2-v2's verdicts on unchanged answers).
- **Rule**: adopt if (a) judge text accuracy pooled over the 352 dev figures beats v2's no-think 264/352 and (b) p2-v2r GROVE is not
  below 65.87. The prompt is not changed after it runs. Fail → reported as a dev negative, nothing runs on VistaQA.
- **If adopted, `v3`** (post-freeze analysis like v2; never replaces `final` or v2), on VistaQA v1.1 (image + question only):
  `v3-rr` = the router cfg above with `"ids": [ids v2-route answered yes]`; `v3-th` = v2-ans cfg + `"thinking": true,
  "max_new_tokens": 4096, "ids": [ids v3-rr answered yes]`. Merge: `think v3-s3r --parts v2-s3r v3-th` → `override v3-pc --parts
  v3-s3r final-prem v2-veto --confirm --veto-tau 0.4` → `countfix v3 --parts v3-pc v2-cpt final-cnt --tau 0.8 --keep-masks` → `check`.
  Hardware: PC1 RTX 3090 (`"max_memory": ["21GiB"]`), or Kaggle T4x2 if the PCs are unreachable; greedy, same config either way.
  Predictions committed before judging; AWQ judge primary, full judge secondary; reported beside v2 whichever way it moves.
- **Dev result** (rr-mp K17, rr-p2 K08; Kaggle, the PCs were unreachable): router yes on 121/200 probe questions and 7/152
  proxy2 figures. Probe text 143/200 (no-think 122, always-think 142); proxy2 routed 142/152, none of the 7 answers change, so
  **p2-v2r = 65.87**. (a) 285/352 > 264 and (b) 65.87 ≥ 65.87: **adopted**, v3 runs as registered. Note (b) holds only because the
  router leaves proxy2's chart-reading questions alone; proxy2 cannot test what thinking does on the questions it routes.
  v2's VistaQA score (48.01) was known when this ran; the rule and v3's config were committed before it (`070365a`).
- **v3 result** (`v3-rr` K08, K17: router yes on 126 of v2's 353 figure questions; `v3-th` K11, K03, K09, K04 on
  Kaggle T4x2, the PCs were busy with E1/E2; median 451 s/question, 16.4 GPU-h): 99 traces closed, 27 hit 4096 tokens and keep the
  no-think answer. Merge as registered: 252 overridden, 20 counts raised; 77 answers differ from v2, masks identical. Predictions
  committed (`b8d7e13`) before judging. AWQ judge (K07): **GROVE 48.21** (text 61.02, mIoU 53.08); 48.35 with v2's verdicts
  on the 1,080 unchanged answers (6 judge flips there, 4 down, 2 up). **v3 − v2 +0.21** (95% CI −0.33 to +0.76), v3 − entry −0.19
  (−2.17 to +1.80). Changed answers: 22 right gained, 8 lost (math 14/4 of 38, science 8/4 of 39); math 21.9 → 24.7, the other
  domains move only by judge flips. Full judge (K05): **47.29** (text 59.81), +0.28 over v2's 47.01 (CI −0.07 to +0.66), −0.85
  vs the entry's 48.14 (−2.76 to +1.08); changed answers 19 gained, 7 lost; no flip on an unchanged answer.

### Veto counterfactuals — registered 2026-11-06 after v2's score, before computing them (analysis only)
v2's aggregates point at the SAM3 veto (hallucination 70.9 → 63.5). Two rows isolate it, built offline from existing passes; no new
inference, verdicts reused from the AWQ/full runs of `final*` and `v2*` (identical answers), judged only if an answer is new:
- `v2-nv`: v2 without the veto: `override v2-pcn --parts v2-s3r final-prem --confirm` → `countfix v2-nv --parts v2-pcn v2-cpt final-cnt --tau 0.8 --keep-masks`.
- `final-v`: the entry with the veto: `override final-pcv --parts final-s3 final-prem v2-veto --confirm --veto-tau 0.4` →
  `countfix final-v --parts final-pcv final-cpt final-cnt --tau 0.8 --keep-masks`.
Reported as post-hoc analysis whichever way they move; nothing is selected by them.
- **Result**: `v2-nv` 47.79 [full 46.95], so the veto adds +0.22 [+0.06] to v2; `final-v` 49.03 [48.74], +0.56 over `final-km`.

### Shared-PC runs while v3 finishes — logged 2026-11-06 before running (no VistaQA scoring)
- **E1 cross-hardware reproducibility** (PC1, RTX 3090, one GPU): `repro-ans` = v2-ans cfg with `"max_memory": ["21GiB"]` on 150
  VistaQA ids (`random.Random(20261106).sample`); `repro-seg` = SAM3 (`@/models/sam3`, the byte-identical HF copy) on v2-ans's own
  Kaggle boxes for the same ids. Compared with the Kaggle T4 runs only as counts: identical answer text, identical boxes, mask IoU
  with v2-seg. No GT, no judge, no answer is read.
- **E2 thinking budget** (dev, PC1): the 18 MathVista probe questions whose trace hit 4096 tokens, re-run with
  `"max_new_tokens": 8192` (`mpt8k-q38`), judged on the probe GT. Changes nothing registered.
- **E1 result**: 3090 vs T4x2 on 150 ids, 11.2 s/sample: answer text identical 147/150, raw output and boxes identical 125/150
  (box coordinates shift by GPU arithmetic), box count equal 148/150. SAM3 on the same boxes: mask count equal 150/150, 259 masks,
  190 byte-identical, IoU mean 0.9998, min 0.975.
- **E2 result** (`mpt8k-q38` GPU1 ids < 121, `mpt8k2-q38` GPU0 ids ≥ 121, merged as `mpt8k-all`; judge K01): 8192 tokens closed
  10 of the 18 traces (mean 330 s/question on a 3090, 1.65 GPU-h). The 10 closed are right 3 times vs 2 for their no-think answers
  (2 gained, 1 lost); the 8 still open are all wrong. Probe text: always-think 142 → 143 of 200, reasoning router 143 → 144 (16 of
  the 18 are router-yes). The traces that hit 4096 are mostly questions the model cannot solve, so a larger budget buys one answer
  for twice the worst-case time. Nothing registered changes.

### Post-freeze rounds R1–R3 — rules written 2026-11-07, before any of their dev results (analysis only; the entry stays `final`)
As for v2/v3, the motivation comes from VistaQA aggregates already reported: the backbone reversal (Qwen3.8 better on dev, worse
on VistaQA), 169 of 460 right grounded answers with too many masks, robotics premise false alarms. Every decision uses proxy dev
(560) and proxy2 (580) only, AWQ judge, verdicts reused where an answer is unchanged. Adopted rounds are bundled into one VistaQA
row `v4`, registered separately before any VistaQA pass of it; failed rounds are reported as dev negatives.
- **R2 mask dedup** (`fleet.py dedup NEW --parts BASE --iou 0.5`, `pipeline.dedup_masks`): masks in the VLM's evidence order; drop
  one whose IoU with an already kept mask is ≥ 0.5. One setting, no grid; text unchanged. On the four final-stage dev runs
  `cf3-s3-es-q36-27b`, `cfv0.4-q38` (proxy dev), `p2-cf3-q36`, `p2-v2` (proxy2). **Adopt** if GROVE pooled over both dev sets rises
  for both recipes and falls on none of the four runs.
- **R3 three-way premise AND** (`override ... --and-prem PREM2`): an override also needs Qwen3.8's premise pass to flag the question
  (its own yes/no check confirming), on top of the 8B confirm and the SAM3 veto τ 0.4. It is the last untested combination: 8B∧Qwen3.8
  without the veto was already worse than the veto on proxy2 (`p2-prem-q38`). Premise stage of the Qwen3.8 recipe: proxy dev
  `s3-q38-27b` + `prem-q3vl8b` + `veto-sam3` + `prem-q38-27b`; proxy2 `p2-s3r` + `p2-prem` + `p2-veto` + `p2-prem-q38`. Without
  `--and-prem` these must rebuild `pcv0.4-q38` and `p2-pcv0.4-q38` exactly. **Adopt** if pooled GROVE rises over those two and falls
  on neither.
- **R1 answer selector** (new pass: `push --pairs A B`, `pipeline.SELECT`; merge `fleet.py pick NEW --parts A B SEL`): Qwen3.8-27B nf4,
  no thinking (the stronger model on both dev sets), sees the image, the question and the two answer-stage answers as A and B, in
  both orders, wherever their normalized text differs. B's prediction (text and masks) replaces A's only if it picks B in both
  orders. A is the Qwen3.6 recipe (the entry's backbone), so the selector must overrule it twice. Answer stage: proxy dev
  `s3-es-q36-27b` (A) vs `s3-q38-27b` (B); proxy2 `p2-s3-q36` vs `p2-s3r` (v3's recipe: its thinking changed none of proxy2's
  answers). Prompt, fixed here: "Question about this image: {question} / Answer A: {a} / Answer B: {b} / Look at the image carefully.
  Which answer is correct? If both or neither are, pick the more accurate one. Reply with the letter A or B only." **Adopt** if the
  picked stage's GROVE beats both single backbones on the pooled dev sets and is not below the better one on either set. The
  best-of-two oracle (by verdicts) is reported as the ceiling and not used for anything.
- Hardware: R2 and R3 offline; R1's pass on PC1 (`"max_memory": ["21GiB"]`), or Kaggle T4x2 if the PCs are busy.
- **R2 result** (`r2-*`): 11, 1, 1 and 0 near-duplicate masks dropped on the four runs. With each run's own verdicts:
  `cf3-s3-es-q36-27b` 73.81 → 73.82, `cfv0.4-q38` 74.23 → 74.23, `p2-cf3-q36` 57.43 → 57.44, `p2-v2` 65.87 → 65.87. v2's recipe
  does not rise, so **not adopted**: the dev sets hardly contain near-duplicate masks, so they cannot test the idea. (Cached
  verdicts first gave `p2-cf3-q36` 57.31; that came from mixing verdicts across judge kernels, not from the masks.)
- **R3 result** (`r3c0-*`, `r3and-*`; 14 never-judged proxy-dev answers judged in `r3-judge-dev`, K01): the rebuilt premise stage
  equals `pcv0.4-q38` byte for byte; `p2-pcv0.4-q38` turned out to be built on the unrouted masks (`p2-s3-q38`, rebuilt exactly),
  so proxy2 is reported on both bases (same text decisions). Both arms use one verdict cache. proxy2: overrides 165 → 145,
  GROVE 65.31 → 65.66 (unrouted 62.71 → 63.04): 7 of 8 removed false alarms turn right (+3.92 summed score), the 12 removed
  overrides on false-premise questions (11 math) go 5 right → wrong, 3 wrong → right (−1.89). Proxy dev: 72.47 → 72.07 (3 catches
  lost, 1 false alarm fixed). Pooled 68.83 → 68.81 and proxy dev falls: **not adopted**.
- **R1 result** (`sel-dev` PC1 3090, `sel-p2` K08 + K01 T4x2 because PC1's second GPU was taken; 20 never-judged proxy-dev
  answers judged in `r1-judge-dev`, K17; each pick keeps its source run's own verdict). Answer pairs differing: 257/560 and
  293/580. Replies (A/B order, swapped order): proxy dev A-A 97, B-B 36, B-A (picks Qwen3.8) 67, A-B (picks Qwen3.6) 57; proxy2
  138, 30, 51, 74: mostly position bias, so the fallback to Qwen3.6 decides most pairs. GROVE, Qwen3.6 / Qwen3.8 / pick / best-of-two
  ceiling: proxy dev 65.07 / 63.88 / **65.46** / 70.76; proxy2 55.70 / 60.04 / **56.85** / 66.34; pooled 60.30 / 61.93 / 61.08.
  The pick is below Qwen3.8 on proxy2 and pooled: **not adopted**. (Correction to the rule's wording: at this answer stage Qwen3.8
  is not the stronger model on proxy dev, only on proxy2 and in the full chains.)
- **Rounds R1–R3: none adopted, so no `v4` from them.** Nothing ran on VistaQA.

### R4 calculator tool — rule written 2026-11-07, before any of its outputs (analysis only)
Question: on the figure questions the reasoning router (`pipeline.REASON`) calls multi-step, does Qwen3.8 writing a program beat
thinking? `pipeline.POT` asks for a short Python program that reads the values off the image into commented variables, computes,
and prints the final answer; `run_program` runs it in a separate interpreter (imports: math only; no `_` names, files or OS;
2 GB, 10 s). Failure (unsafe, error, nothing printed, timeout) → the no-think answer, as for an unclosed trace (`fleet.py think`
treats `raw.ok` like a closed trace). Masks unchanged; `expand_short` restates short outputs as a sentence.
- Dev runs: `mpp-q38` = Qwen3.8 nf4, `{"pot": true, "expand_short": true, "seg": ""}` on all 200 MathVista probe questions
  (so always-tool is reported too); `p2-pot` = the same on proxy2's 7 router-yes ids (`rr-p2`).
- Variants, both fixed now: **(a)** the tool instead of thinking on router-yes questions; **(b)** thinking as in v3, the tool only
  where the thinking trace hit 4096 tokens. Probe = tool answers on `rr-mp`'s 121 yes ids where the program ran (b: only on
  unclosed ones), otherwise `mp-think` (b) or `mp-q38` (a); proxy2 = the `p2-v2r` chain with `think` from `p2-pot`.
- **Adopt** the better variant if (1) probe text accuracy beats v3's recipe, 143/200, and (2) proxy2 GROVE is not below 65.87. If
  adopted, register `v4` = v3 with that change before any VistaQA run of it; otherwise R4 is a dev negative.
- **R4 result** (`mpp-q38` K12, K13, K15, K16, judge K17; `p2-pot` K01): the program ran on 191/200 probe
  questions and 7/7 proxy2 figures, median 32 s/question on T4x2 (2.8 GPU-h for 200; v3's thinking took 451 s on VistaQA). Probe
  text: no-think 122, always-think 142, v3 recipe 143, **(a) tool instead of thinking 135**, **(b) thinking, tool where the trace hit
  4096: 146**, always-tool 140 /200. On router-yes questions (a) vs v3: 9 gained, 17 lost; thinking reasons better, the tool is a
  better fallback than the no-think answer. proxy2: the tool's 7 answers are word for word v3's (simple chart differences and
  perimeters), so both variants give **65.87** (`r4a-p2`; v3's proxy2 chain `p2-v2r` rebuilt exactly with `p2-cpt-q38` + `p2-cnt`;
  `p2-s3rt`/`p2-pcrt` are the always-think chain, not v3's). (b) passes both tests: **adopted**.

### v4 — registered 2026-11-07, before any VistaQA run of it (post-freeze analysis; never replaces `final`, v2 or v3)
v3 plus R4 (b): on v3's routed questions whose thinking trace hit 4096 tokens (27 of v3-th's 126), the calculator tool's answer
where its program runs; otherwise exactly v3. Masks unchanged.
- `v4-pot` = v3-th's cfg with the tool instead of thinking: `{"vlm": "Qwen/Qwen3.8-27B", "quant": "nf4", "pip": ["bitsandbytes"],
  "max_memory": ["14GiB", "14GiB"], "seg": "", "expand_short": true, "pot": true, "ids": [the 27 v3-th ids without "</think>"]}`,
  VistaQA v1.1 (image + question only), Kaggle T4x2.
- Merge: `think v4-s3r --parts v3-s3r v4-pot` → `override v4-pc --parts v4-s3r final-prem v2-veto --confirm --veto-tau 0.4` →
  `countfix v4 --parts v4-pc v2-cpt final-cnt --tau 0.8 --keep-masks` → `check`.
- Predictions committed before judging; AWQ judge primary, full judge secondary; reported beside v3 whichever way it moves. At most
  27 answers can change, so the expected effect is a fraction of a point.
- **v4 result** (`v4-pot` K05, K07 (K10 out of quota); 40 and 36 min wall incl. model load): the program ran on 21/27;
  merge as registered: 252 overridden, 20 counts raised; 17 answers differ from v3 (13 math, 4 science), masks identical.
  Predictions committed (`eeed5af`) before judging. AWQ judge (K11): **GROVE 48.37** (text 61.28, mIoU 53.08), **v4 − v3 +0.16**
  (95% CI −0.23 to +0.56), v4 − v2 +0.36 (−0.15 to +0.90), v4 − entry −0.03 (−2.02 to +1.91). The changed answers: 3 right gained, 3
  lost; the +0.16 is 5 judge flips on unchanged answers (4 up, 1 down): with v3's verdicts there, 48.18. Full judge (K04): **47.37**
  (text 59.98), +0.08 over v3 (−0.19 to +0.34), +0.36 over v2 (−0.10 to +0.83), −0.77 vs the entry (−2.70 to +1.14); changed
  answers 4 gained, 2 lost, no flip on an unchanged answer. **Level with v3**: the probe's +3 (146 vs 143 of 200) is +0 to +2 of 27
  on VistaQA. (CIs: paired bootstrap, 10,000 resamples, seed 0; re-running v3's comparisons this way moves their CI ends by ≤0.04.)

### Post-freeze rounds R6–R7 — rules written 2026-11-07, before any of their dev results (analysis only; the entry stays `final`)
Aimed from dev losses only: on proxy dev (cfv0.4-q38) and proxy2 (r4a-p2) the points go to wrong answers on grounded questions
(14.4 / 17.4), then imperfect masks on right answers (7.8 / 13.8; 6.8 / 8.1 of it on samples with the right number of masks).
Image sizes (meta only): 536 of 1,157 VistaQA images exceed the answerer's cap `max_pixels` 1280² (1.64 MP) and are downscaled;
no proxy dev or proxy2 image does, so neither set can test the cap.
- **R6 input resolution.** Probe `src/build_vstar_probe.py` → `Competition Data/vstar_probe/`: V*Bench (`craigwu/vstar_bench`
  rev d9ae62c9; SA-1B photos; license not stated on the card, disclosed), all 238 questions (attributes 115, relative position
  76, OCR 30, GPT4V-hard 17), median 3.38 MP, 232 above the cap. Free-form question, GT = `options[0]`; target boxes kept locally.
  Runs: v2-ans's cfg (Qwen3.8-27B nf4, prompt v0, expand_short) at `max_pixels` 1280² (`vs-q38`) and 2048² = 4,194,304 (`vshr-q38`),
  both on the same hardware; AWQ judge, official prompt. Box hit = share of V* targets that some evidence box covers with IoU ≥ 0.5.
  **Adopt** if (1) judge text accuracy at 2048² beats 1280² and (2) box hit is not lower. Proxy dev and proxy2 cannot change.
- **R7 mask cleanup** (`fleet.py clean NEW --parts BASE --mode M`, `pipeline.clean_masks`, offline): per predicted mask, (a) `fill`
  interior holes, (b) `lcc` keep the largest 8-connected component, (c) `both`. Text unchanged, so each base's own verdicts are
  reused exactly; GROVE recomputed with the official mask scorer. Bases: proxy dev `cfv0.4-q38` (74.23), proxy2 `r4a-p2` (65.87).
  **Adopt** the variant with the larger mean gain over the two sets if it raises GROVE on both.
- **If either is adopted**, `v5` = v4 plus the adopted rounds, registered in its own row before any VistaQA run. R6 re-runs v2-ans's
  cfg with `"max_pixels": 4194304` on the 536 large ids only (`v5-hr`), SAM3 box masks and concept masks from it (`v5-hrseg`,
  `v5-hrcpt`, then `fit`), `splice`d into v2-ans / v2-seg / v2-cpt; the figure, thinking and tool passes stay as they are (on
  routed figures they override the answer pass anyway). R7 adds `clean` as the last step. Nothing else changes.
- **R7 result** (`r7{none,fill,lcc,both}-{dev,p2}`, base verdicts copied; `none` reproduces 74.23 / 65.87): fill 74.66 / 65.82
  (+0.43 / −0.05), lcc 72.93 / 65.71 (−1.30 / −0.16), both 73.38 / 65.66 (−0.85 / −0.21). Fill helps COCO (GT polygons have no
  holes) and costs proxy2 a little; lcc cuts objects split by occlusion. No variant raises both: **not adopted**.
- **R6 result** (`vs-q38` K01, K09, 59–60 min per shard; `vshr-q38` K12, K13, K16, 63–66 min, ~1.6x time per
  question; no OOM; judges K08, K11): judge text **176 → 192/238** (27 gained, 11 lost; attributes 83 → 85, relative position
  62 → 69, OCR 24 → 27, GPT4V-hard 7 → 11); box hit **194 → 209/292** (66.4 → 71.6%). Both conditions hold: **adopted**.

### v5 — registered 2026-11-07, before any VistaQA run of it (post-freeze analysis; never replaces `final`, v2, v3 or v4)
v4 plus R6: the answer pass sees the 536 VistaQA images whose declared width x height exceeds 1280² at up to 2048² (4.19 MP;
39 are larger and still downscaled), with SAM3 masks and concept masks redone from its boxes. R7 is not part of it.
- `v5-hr` = v2-ans cfg + `"max_pixels": 4194304, "ids": [the 536 ids]`; `v5-hrseg` = `--boxes v5-hr --models
  chonlanawawa/sam3/pyTorch/default/1`, `{"seg": "kaggle:chonlanawawa/sam3/pyTorch/default", "ids": [same]}`; `v5-hrcpt` =
  `--concepts v5-hr`, same SAM3 cfg. VistaQA v1.1 (image + question only), Kaggle T4x2; `fit` on both SAM3 runs.
- Merge (`splice` swaps in the 536 ids; the rest is v4's chain on the new parts):
  ```
  fleet.py splice   v5-ans --parts v2-ans v5-hr
  fleet.py splice   v5-seg --parts v2-seg v5-hrseg --scope v5-hr
  fleet.py splice   v5-cpt --parts v2-cpt v5-hrcpt --scope v5-hr
  fleet.py remask   v5-s3  --parts v5-ans v5-seg
  fleet.py remask   v5-s3r --parts v5-s3 v2-seg1 --route v2-route
  fleet.py think    v5-s3t --parts v5-s3r v3-th
  fleet.py think    v5-s3p --parts v5-s3t v4-pot
  fleet.py override v5-pc  --parts v5-s3p final-prem v2-veto --confirm --veto-tau 0.4
  fleet.py countfix v5     --parts v5-pc v5-cpt final-cnt --tau 0.8 --keep-masks
  fleet.py check    v5     --inline "Competition Data/VistaQA"
  ```
- Predictions committed before judging; AWQ judge primary, full judge secondary; bootstrap against v4, v3 and the entry;
  reported beside v4 whichever way it moves. Re-runs only for infrastructure failures, same config.
- **Infrastructure failure, logged before the re-run** (format check only, no answer read): `v5-hr` (K04, K03, K05, K08,
  K17, K15; 74–91 min per shard) caught CUDA OOM on 6 of 536 (ids 167, 465, 653, 750, 872, 1147: raw "OOM", empty
  answer). With `max_memory` 14+14 GiB the weights fill GPU0, leaving too little for the 4 MP vision activations; the probe's
  `vshr-q38` had 4 such OOMs, counted wrong when R6 was decided. Re-run of those 6 ids as `v5-hrfix` with the weights placed
  `["8GiB", "14GiB"]` (placement only: same model, precision, prompt, inputs), spliced in as `v5-hr2 = splice(v5-hr, v5-hrfix)`,
  which the chain uses in place of `v5-hr`. The as-run chain (6 empty answers) is scored too as `v5-asrun` and reported beside it.
- **The re-run OOMed again (6/6, K09)**, so placement is not the cause: these 6 are the only images processed at the full cap
  (five are 2048x2048, one is larger and downscaled to it); every image that ran was smaller. The log shows Qwen3.8's linear-
  attention layers on their reference PyTorch path (`chunk_gated_delta_rule`; flash-linear-attention was slower on T4). A T4 cannot
  run them at this config; no 24 GB GPU is free (PC1 busy, PC2 lacks the weights). **Fixed now, before any merge or judging:**
  `v5` (headline) = the chain with the fallback a T4 deployment would use when the 4 MP pass runs out of memory: the 1280² pass
  for those 6 ids, i.e. v4's answer and masks there (`v5-hrok` = v5-hr without its 6 OOM lines; the splices use it as patch and
  scope). `v5-asrun` = the chain exactly as registered on `v5-hr` (6 empty answers). Both reported, plus v5 − v4 on the other
  1,151 ids. SAM3 passes run on all 536 ids (the 6 have no boxes, so empty masks).
- **v5 result** (`v5-hrseg` K01, `v5-hrcpt` K12 (296 concept prompts, 53 counting lines); merge as registered: 252
  overridden, 16 counts raised; v5 differs from v4 in 156 answers and 295 mask sets, all on the 530 large images that ran, and is
  identical to v4 elsewhere and on the 6 fallback ids; predictions `f67941c` committed before judging). AWQ judge (K11): **GROVE
  48.03** (text 61.11, mIoU 52.70): **v5 − v4 −0.34** (95% CI −1.26 to +0.59; on the 1,151 ids outside the six −0.35, −1.26 to
  +0.61), v5 − v3 −0.19, v5 − v2 +0.02, v5 − entry −0.38 (−2.39 to +1.56). The 156 changed answers: 22 right gained, 23 lost; 3
  judge flips on unchanged answers; the redone masks cost −0.11 on their own (373 samples with text and verdict unchanged).
  **R6 did not transfer**: V*Bench's small targets in 3.4 MP photos gained 16/238, VistaQA's large images gained nothing.
  One high-res answer ran away to 5,903 characters (1,844 tokens), which the AWQ judge took in stride.
  Full judge (K16): **47.34** (text 60.24): v5 − v4 −0.03 (−0.92 to +0.90), − v3 +0.04, − v2 +0.33, − entry −0.81 (−2.73 to
  +1.12); changed answers 24 gained, 21 lost; no flip on an unchanged answer. **Level with v4 under both judges.**
- **v5-asrun** (6 empty answers; 5 after the premise override): the 5 empty answers judged in `v5e-asrun` (K01), the rest
  reuse v5's verdicts: **48.11**, +0.09 over v5 (CI +0.00 to +0.23), because **the AWQ judge called 3 of the 5 empty answers
  correct** (official prompt and parser). A quirk of the judge, not a gain. Full judge: 47.40 (+0.06 over v5), 2 of 5 called correct.
- **Judge infrastructure** (no verdict changes on shorter prompts; greedy): the full judge's `max_model_len` 1024 rejected v5's
  1,844-token answer (K04; then 1536 on K13, too short), so `v5-fulljudge` runs with 2560 (K16; the KV cache holds 2,816).
  The official parser crashes on a `"correct": null` verdict (`int(None)`); `src/judge_kernel.py` now scores that 0, as the
  official code does for unparseable output. Hit once, on an empty answer with the full judge (`v5e-asrun-fulljudge`,
  K15: 2 of the 5 empty answers called correct).

### Development closed — 2026-11-07, before any VistaQA failure is inspected
No further rounds, systems or VistaQA runs. Rows reported: `final` (entry), v2–v5 and the registered counterfactuals. From here
VistaQA samples are looked at one by one for the first time, only for the report's qualitative appendix figure, and nothing
in the system changes after it. **Selection rule, fixed before looking** (`final`'s predictions, AWQ verdicts, per-sample S):
one panel per outcome: (1) grounded, right answer, S ≥ 0.8; (2) false premise, caught (right, no masks); (3) grounded, right
answer, S < 0.3; (4) grounded, wrong answer; (5) grounded, answered "not present" (false alarm); (6) false premise, missed.
Candidates sorted by image id; `random.Random(0)` draws one per category in that order, redrawing (≤ 50 tries) when the domain
is already used. Each panel: image with GT masks (outline) and predicted masks (fill), question, GT answer, prediction, T and S.
Script: `Scripts/qual_figure.py` → `report/qual.pdf`.
- **Drawn** (rule as written): ids 663, 1039, 1091, 925, 618, 763 (of 174, 218, 116, 269, 113, 88 candidates). Panel 6 (false
  premise, judged wrong) answers "There is no red shopping bag in the image." against GT "there are 0 people carrying a red shopping
  bag": the answer does reject the premise and the judge still calls it wrong, so the panel is titled "False premise, judged
  wrong" rather than "missed". Kept, not redrawn. Report appendix B.

Findings so far (proxy dev):
- **Premise check** (separate pass: "does the question assume something not visible? name it" → POPE yes/no confirm →
  "There is no X in the image." + no masks) is the biggest single gain: +6.8 to +8.7 GROVE. Offline combos: 8B∧27B
  premise 134 hit/13 false (≈ same as 8B alone) → keep 8B confirm only.
- 27B errors: counting mostly under-count by 1 (25/100); identification judged too coarse/fine vs COCO names
  (pigeon vs bird) — proxy artifact; OCR GT noisy (COCO-Text); bare answers ("ImageMD") judged incomplete.
- Backbone: Qwen3.6-27B nf4 (64.91) ≫ Qwen3-VL-8B (58.60) ≫ 4B (39.02). 27B writes full-sentence answers (spatial text 97%).
- two_stage helps hallucination and multi-object masks but its terse answers kill spatial/comparison text; verify on
  self-generated targets does nothing. Hallucination misses are mostly genuine false-premise answers (310, 305 viewed).
- Official judge is strict about completeness: bare "right"/"bed" fail even when correct → `expand_short` rewrites
  answers ≤3 words into a full sentence (no new info).
- Hallucination GT spot-check (ids 302, 307, 316): 2/3 genuine model hallucinations, 1/3 COCO label noise (small bottle).
- Proxy dev is easier than VistaQA (COCO scenes, templated Qs): use it for *relative* comparisons only.

Fixes after v1 OOMs: SAM boxes in chunks of 8, VLM max_memory leaves ~4 GiB on GPU1, empty_cache per sample,
per-sample OOM caught, `expandable_segments`.

### Weight-space programming (WSP) — rules written 2026-11-07, before any of its results (user request after development closed; analysis only)
The user asked to test https://rikiyat.github.io/blog/weight-space-programming/: a large general-purpose LLM reads a smaller
network's weights and activations and writes new weights directly, with no optimizer and no gradient descent; forward passes on
practice data are its only feedback. Here the editor is Claude Opus 5.5 (the assistant that wrote this pipeline) and the edited
model is Qwen3-VL-8B-Instruct, the entry's small model. Analysis only: it never replaces `final` or v2–v5. It is designed after
development closed and after the qualitative figure looked at six VistaQA samples; the target below comes from proxy dev
(`devall-q3vl8b`, Oct 27), not from VistaQA.
- **Target**: single-pass answers that accept a false premise. With prompt v0 the 8B denies 68 of 150 absent objects on proxy
  dev (ABSENT regex; judge text 52%) and 1 of 410 present ones, while its own yes/no check (`prem-q3vl8b`) catches most
  absences. The ability is there but the free-form answer does not use it: the post's "obstacle".
- **Allowed edits**: rank-limited changes to existing weight matrices of the language model and added bias vectors, computed by
  the editor from the model's weights and activations. No gradients, no training loop; vision encoder, prompt, decoding and
  pipeline unchanged.
- **Practice data, the only data in the edit loop**: `src/build_wsp_practice.py`, COCO val2017 images that proxy dev does not use;
  per image one question about an absent, frequently co-occurring category and the same question about a present
  single-instance category, with templates proxy dev does not use. Split A (analysis, 200 images) and B (feedback, 100 images).
  Proxy dev, proxy2 and VistaQA never enter the loop.
- **Budget and objective**: at most 40 edit candidates scored on B. Score = denial rate on absent − 1.35 × denial rate on present
  (VistaQA's published 27% hallucination share; a hit lifts a hallucination sample from 0.01 to 1, a false alarm drops a grounded
  sample from about 0.5 to 0.01: 0.73 × 0.5 / 0.27 ≈ 1.35). Ties go to the smaller edit. The chosen edit is committed before any dev run.
- **Dev A/B**: 8B single pass, prompt v0, `seg: ""`, then SAM3 box masks; unedited (`wsp0-dev`, `wsp0-p2`) vs edited (`wsp1-dev`,
  `wsp1-p2`) on proxy dev (560) and proxy2 (580), both arms on the same hardware, AWQ judge. Reported: GROVE, text, hallucination
  vs grounded, paired bootstrap of the difference over the 1,140 samples. "Helps on dev" = pooled difference > 0 with the 95% CI above 0.
- **VistaQA A/B, registered now**: `v-wsp0` and `v-wsp1`, the same two configs on VistaQA, AWQ judge, each scored once, run
  whatever dev shows (nothing is selected by it). Also reported, offline from cached verdicts where answers match: each arm plus
  the entry's 8B premise check (`final-prem`, `override --confirm`), to compare the weight edit with the prompt-based check.
- **Edit loop log** (practice data only; PC2, one RTX 3090, the cached 8B in fp16 through `pipeline.VLM`):
  - Diagnosis (`wsp.py collect` + `wsp_an.py`, split A, hidden state where the answer starts, after `{"answer": "`): the absent-minus-present
    direction separates each question from its present twin in 97–99% of pairs from layer 16 on (unpaired AUC 0.93–0.97), but up to
    layer 19 it does not predict which absent questions the model denies (denied vs accepted AUC ≈ 0.5); the decision forms in
    layers 19–23 (AUC 0.72 at 20, 0.87 at 24), where the direction's logit lens reads "nonexistent", "不存在", "none", and from
    layer 32 "There". Unedited: 141/200 absent denied, 2/200 present (split A).
  - Detector amended after round 1, before round 2 (measurement only, same practice answers): a denial must name the queried object
    ("There is no chair", "the handbag is not visible", "does not contain a cup") or say the answer cannot be determined (the judge
    accepts that on false premises). The first regex counted "There is no object on top of the clock" (present clock) as a false
    alarm and missed "The image does not contain a handbag". Round 1 re-scored from the saved answers (`wsp.py rescore`).
  - Round 1 (7 candidates; split B, 100 absent / 100 present; hit / false alarm / score): none .75 / .04 / .696; steer along
    abs21 at layer 20 by 0.5 / 1 / 1.5 gaps .83/.05/.762, .86/.08/.752, .91/.13/.734 (rewrites 25 / 40 / 53 of the 100 present
    answers); wire abs18 → abs22 from layers 14–17 .75/.03/.710 and .72/.02/.693; amplify abs18 in 14–17 .78/.04/.726 (13 present
    answers rewritten); amplify abs21 in 18–20 .80/.04/.746 (27).
  - Round 2 (8 candidates): amplify abs21 in 18–20 ×2 / ×3 .81/.04/.756 both; amplify abs22 in 18–21 ×1 / ×2 .80/.04/.746,
    .81/.04/.756; amplify 14–17 and 18–20 together .80/.03/.760 (×2 on 18–20: .746); steer 0.3 gap .81/.05/.742; **steer 0.3 gap +
    amplify abs21 in 18–20: .85/.05/.782**. The amplifiers level off at about .81 with no added false alarms; the small constant
    push adds the rest.
  - Round 3 (6 candidates around it): steer 0.2 / 0.4 / 0.5 + amplify .762 / .782 / .756; steer 0.3 + amplify ×2 .725; + amplify
    14–17 .772; steer 0.3 + amplify abs22 in 18–21 ×2 .769. Stopped at 21 of 40: the scores sit at .76–.78, within noise of
    each other (one question = .01).
  - **Chosen** (registered rule; the .782 tie goes to the smaller push): steer the residual entering layer 21 by 0.3 gap
    (α = 4.04) along the layer-21 absent direction, as a new bias on layer 20's down_proj, and amplify that direction ×2 in what
    layers 18–20 write (o_proj and down_proj, rank-1, β = 1), centered at the midpoint 3.70 by a bias on layer 20. Three of the
    8B's 36 decoder layers change: 6 matrices get a rank-1 update, layer 20's down_proj gets a bias; vision encoder untouched. Practice B: hit .75 → .85, false
    alarm .04 → .05. Frozen as `runs/wsp-edit.json` (spec `runs/wsp-spec.json`; cfgs `runs/wsp-cfg0.json`, `runs/wsp-cfg1.json`),
    committed before any dev run.
  - Split A with the frozen edit (run after the freeze, in-sample: split A gave the directions and the scales): hit .705 → .79,
    false alarm .010 → .015 (200 absent / 200 present), score .692 → .770; the same shift as on B.
- **Dev A/B results** (AWQ judge; unedited arms judged in full, edited arms reuse the unedited verdicts where the answer is
  unchanged and judge the changed answers, `wsp1-*-chg`; SAM3 tracker on each arm's own boxes, Kaggle T4):

  | set | unedited | edited | edited − unedited (95% CI) | text | hallucination | grounded | answers changed | verdicts gained / lost |
  |---|---|---|---|---|---|---|---|---|
  | proxy dev (560) | 58.42 (`wsp0-dev-s3`) | 61.66 (`wsp1-dev-s3`) | +3.24 (+1.91 to +4.73) | 66.1 → 69.5 | 52.1 → 61.8 | 60.7 → 61.6 | 62 (35 hallucination) | 19 / 0 |
  | proxy2 (580) | 48.35 (`wsp0-p2-s3`) | 50.27 (`wsp1-p2-s3`) | +1.92 (+0.86 to +3.14) | 62.1 → 63.8 | 66.8 → 73.4 | 41.5 → 41.6 | 84 (58 hallucination) | 12 / 2 |
  | **pooled (1,140)** | 53.30 | 55.86 | **+2.57 (+1.71 to +3.50)** | | | | | 31 / 2 |

  **Helps on dev** by the registered rule. proxy dev: denials on absent objects (ABSENT regex) 68 → 86 of 150, on present ones
  1 → 2 of 410; proxy2 67 → 98 of 158 and 3 → 4 of 422 (science 14 → 25 of 30, robotics 10 → 17 of 40), but many of proxy2's
  math absences were already judged right in words the regex misses ("There is no vertex labeled V"), so the judge gains less.
  Changed grounded answers are mostly rewording ("Toilet" → "The image shows a toilet."); the two lost verdicts are a count
  ("6" → "Five bars…") and a false "traffic light is not visible".
- **Edit vs the prompt-based premise check** (offline, verdict cache): each arm + the 8B premise check (`prem-q3vl8b` /
  `p2-prem`, `override --confirm`): proxy dev 67.58 → 68.01 (+0.43, CI +0.00 to +0.96), proxy2 52.60 → 52.73 (+0.13, CI −0.29
  to +0.61), pooled +0.28 (CI −0.03 to +0.62). The check alone adds +6.66 pooled; the edit alone recovers +2.57 of that with no
  second pass, and on top of the check it adds nothing measurable. Edited alone vs unedited + check: −4.09 (CI −5.83 to −2.39).
  (Tooling: `fleet.py remask` now copies `judge_out/`, so a remasked cached run scores without re-caching.)
- **VistaQA A/B results** (the registered pair, each arm scored once; same judge protocol as dev: `v-wsp0` judged in full, the
  154 changed answers of `v-wsp1` in `v-wsp1-chg`; answer kernels K01,K11 / K04,K16, SAM3 on the same accounts,
  judge K17 / K06):

  | arm | GROVE | text | hallucination GROVE (text) | grounded GROVE (text) |
  |---|---|---|---|---|
  | unedited 8B + SAM3 (`v-wsp0-s3`) | 38.76 | 49.78 | 39.69 (39.2) | 38.42 (53.7) |
  | edited 8B + SAM3 (`v-wsp1-s3`) | **40.46** | 51.43 | 45.54 (45.2) | 38.57 (53.7) |
  | unedited + premise check (`v-wsp0-pc`, `final-prem --confirm`) | 44.35 | 55.06 | 68.18 (67.2) | 35.47 (50.5) |
  | edited + premise check (`v-wsp1-pc`) | 44.90 | 55.40 | 69.79 (68.8) | 35.63 (50.4) |

  **Edited − unedited = +1.70 (95% CI +0.83 to +2.60)**: 154 answers changed (81 on false-premise questions), 27 verdicts gained,
  8 lost; "not present" answers (ABSENT regex) on false-premise questions 66 → 90 of 314, on grounded ones 5 → 10 of 843.
  With the premise check on both arms: +0.56 (CI −0.02 to +1.18; 10 gained, 6 lost). Edited alone vs unedited + check: −3.88
  (CI −5.70 to −2.08). Same picture as dev (+2.57, +0.28, −4.09): the edit transfers and recovers about 30% (dev 39%) of the
  premise check's gain with no second pass, but adds nothing measurable once the check runs. All rows are 8B-only analysis
  arms; the entry (48.40, Qwen3.6-27B chain) is unchanged.

### Mask verifier (MV) — rules written 2026-11-08, before any of its results (user request; analysis only)
Target: right answers with more masks than the evidence needs. The only VistaQA input is the aggregate already reported
post-hoc (169 of the entry's 460 right grounded answers with masks have too many, mean S 0.33 vs 0.74); no VistaQA sample or
further breakdown is looked at. It never replaces `final`. On dev the entry's recipe over-masks 1 of 310 right grounded answers
on proxy dev and 29 of 290 on proxy2 (math: headers and labels boxed next to the answer cell), so a third dev set is added.
- **Method (Set-of-Mark check, training-free)**: for each prediction with at least 2 masks whose question is not a counting
  question (`COUNT_Q`), draw each mask's bounding box with its number 1..n on the image and ask a VLM the fixed prompt `MARKS`
  (pipeline.py: the question, the answer, "Which numbered boxes mark visual evidence required to support this answer?", JSON
  list). Keep the masks it lists. No list, an empty list or no valid number = keep all. Answers are never changed, so text
  verdicts are reused and nothing is judged again. Passes: `fleet.py push RUN-ver --marks RUN`, merge `fleet.py verify`.
- **Variants (the only two tried)**: V27 = Qwen3.6-27B nf4 with the answerer's settings; V8 = Qwen3-VL-8B-Instruct.
- **Dev sets**: proxy dev (560, base `cf3-s3-es-q36-27b`), proxy2 (580, base `p2-cf3-q36`) and **proxy3** (new, 200, crowded
  scenes): `src/build_proxy3.py`, RefCOCOg val (umd split; COCO train2014 images, none in proxy dev), seed 0, one question per
  image, proxy dev's templates and answer wording with referring expressions in place of category names: side of the image
  (one referent, 70), left/right of another referent (70, no horizontal overlap), which appears larger (60, mask area ratio
  >= 2.5); GT masks = the referents' COCO masks; no false premises, no counting. Base on proxy3 = the entry's answer and mask
  passes (27B prompt v0 `expand_short`, SAM3 on its boxes; `p3-s3`, AWQ judge); the premise and count stages do not apply.
- **Decision**: GROVE pooled over the three dev sets (1,340), paired bootstrap. The variant with the higher pooled GROVE is
  chosen; it "helps on dev" if its pooled difference is > 0 with the 95% CI above 0. Also reported: over-masked right answers
  before/after, and masks dropped from answers whose mask count already matched the GT (false drops).
- **VistaQA, registered now**: `final-mv` = `final` with the chosen variant's check (`final-ver`), run whatever dev shows,
  scored once with `final`'s verdicts (answers identical). Analysis row only; compared with `final` by paired bootstrap.
- proxy3 built (before any run on it): 60 comparison, 70 left/right of another referent, 70 side of the image, from 325 / 389 /
  917 candidates on 1,300 RefCOCOg val images (images in proxy dev skipped). An expression is used only if it has 2-12 words
  after the article, is a noun phrase, and names no position, depth or size (left, right, behind, front, big, small, ...), so
  the question does not give the answer away. Code: `pipeline.MARKS`, `draw_marks`, `parse_keep`; `fleet.py push --marks`,
  `fleet.py verify`. Checked offline on dev bases only: 132 proxy dev and 120 proxy2 predictions qualify for the check.
- **proxy3 base** `p3-s3` (`p3-ans` 27B on K01,K02,K06,K10; `p3-seg` SAM3 on K02; `fleet.py remask`; judge K06):
  **77.95** (text 89.50, mIoU 74.94; spatial 80.6, comparison 71.9). Right grounded answers with masks: 179, of which 19 have
  more masks than the GT (all spatial; S 0.41 vs 0.81 for the 160 exact ones). proxy3 does show over-masking (proxy dev 1 of 310).
- **Dev results** (checks: V27 `mv27-dev` K05,K03, `mv27-p2` K13,K15, `mv27-p3` K01,K10, 8 s/it;
  V8 `mv8-dev` K07, `mv8-p2` K12, `mv8-p3` K11; merged with `fleet.py verify` into `*-mv`, base verdicts reused):

  | set | base | V27 | V27 − base (95% CI) | V8 | V8 − base (95% CI) | over-masked right before → V27 / V8 | changed (false drops) V27 / V8 |
  |---|---|---|---|---|---|---|---|
  | proxy dev (560) | 73.81 | 73.33 | −0.48 (−0.79 to −0.21) | 73.13 | −0.69 (−1.10 to −0.33) | 1 → 0 / 1 | 16 (12) / 22 (16) |
  | proxy2 (580) | 57.43 | 57.05 | −0.38 (−0.74 to −0.05) | 57.03 | −0.40 (−0.77 to −0.06) | 29 → 22 / 21 | 36 (10) / 39 (13) |
  | proxy3 (200) | 77.95 | 77.85 | −0.10 (−1.04 to +0.82) | 76.63 | −1.32 (−2.60 to −0.22) | 19 → 12 / 13 | 20 (10) / 18 (11) |
  | **pooled (1,340)** | 67.34 | **66.96** | **−0.38 (−0.63 to −0.15)** | 66.69 | −0.65 (−0.95 to −0.38) | 49 → 34 / 35 | 72 (32) / 79 (40) |

  **V27 chosen** (higher pooled GROVE; V27 − V8 +0.27, CI −0.07 to +0.62). It **does not help on dev**: the pooled difference
  is negative with the CI below 0. The check removes about 30% of the over-masked right answers, but 32 of the 72 predictions
  it changed already had the GT's mask count; 26 of those are two-object questions ("Which appears larger, the A or the B?",
  "Is the A left or right of the B?") where it lists only one object, the larger one or the subject, though both are the
  evidence (21 comparison, 5 spatial). Answers are untouched (text accuracy identical on every set). As registered, `final-ver` (V27 on `final`) runs on
  VistaQA anyway, as an analysis row.
- **VistaQA result** (`final-ver` on K06,K02,K05,K03: 330 predictions checked, 292 with a valid list, 0 OOM;
  `fleet.py verify final-mv --parts final final-ver`; scored once with `final`'s verdicts): **`final-mv` 48.61** (text 60.76,
  mIoU 52.50) vs `final` 48.40 (mIoU 52.05): **+0.21 (95% CI −0.13 to +0.53)**, not significant. 111 masks dropped from 98
  predictions; aggregate only: over-masked right grounded answers 169 → 140, drops on predictions whose mask count matched the GT 8,
  mean S of the 88 changed grounded predictions 0.288 → 0.347. The dev sets predicted a loss (−0.38) because over-masking is rare
  there (49 of 779 right grounded answers with masks, 6%; VistaQA 169 of 460, 37%), so false drops outweighed the fixes. The
  entry stays `final` (48.40); `final-mv` is an analysis row.

### Model swaps (MS) — rules written 2026-11-08, before any of their results (user request; analysis only)
Two swaps in the entry's chain, each tested alone; the entry `final` (48.40) never changes. Inputs: model names only. No
VistaQA sample or score was looked at to propose them (the user asked for these two models).
- **MS-A, answerer Qwen3-VL-32B-Instruct** in place of Qwen3.6-27B: cfg = `final-ans` with the model swapped,
  `{"vlm": "Qwen/Qwen3-VL-32B-Instruct", "quant": "nf4", "pip": ["bitsandbytes"], "max_memory": ["14GiB", "14GiB"], "seg": "",
  "expand_short": true}` (same 4-bit scheme as the 27B, so only the model differs; Qwen3-VL boxes are 0-1000 like the 27B's).
  Then the entry's other passes on its output: SAM3 on its boxes (`--boxes`), SAM3 concepts from its labels (`--concepts`), the
  same 8B premise run (`override --confirm`) and the same 8B count run (`countfix --tau 0.8`).
- **MS-B, premise checker Gemma 4 31B** (`google/gemma-4-31B-it`) in place of Qwen3-VL-8B: cfg = `final-prem` with the model
  swapped, `{"vlm": "google/gemma-4-31B-it", "quant": "nf4", "pip": ["bitsandbytes"], "max_memory": ["14GiB", "14GiB"],
  "seg": "", "premise": "only"}` (nf4 settings of `dev-gm4-31b`; `seg: ""` only skips loading SAM2.1, which a premise-only pass
  never uses). Same PREMISE and PREMISE_VERIFY v1 prompts, no tuning. Merged with `override --confirm` onto the entry's answer
  + mask stage, then the same `countfix`. Tooling change before any run: `override` reads the yes/no reply with leading markdown
  stripped (`**No**`); on every existing 8B premise run this changes nothing (0 of 2,297 replies), and `final` and the proxy
  dev chain still reproduce byte for byte.
- **Dev chains** (the entry's recipe reproduces `cf3-s3-es-q36-27b` and `p2-cf3-q36` byte for byte):
  - proxy dev: entry `cf3-s3-es-q36-27b` = `override` `s3-es-q36-27b` + `prem-q3vl8b`, `countfix` + `cpt-sam3-q36-27b` + `dev-es-q3vl8b`.
    MS-A `bb32-dev` → `-seg`, `-cpt` → `bb32-dev-s3` → `-pc` → `bb32-dev-cf3`. MS-B `gp-dev` → `gp-dev-pc` → `gp-dev-cf3`.
  - proxy2: entry `p2-cf3-q36` = `p2-s3-q36` + `p2-prem`, + `p2-cpt-q36` + `p2-cnt`. MS-A `bb32-p2…` → `bb32-p2-cf3`; MS-B
    `gp-p2` → `gp-p2-cf3`.
  - proxy3 (no false premises, no counting: the count stage is a no-op; every premise override is a false alarm): entry
    chain `p3-pc` = `override` `p3-s3` + `p3-prem` (8B premise, `final-prem` cfg). MS-A `bb32-p3` → `-seg` → `bb32-p3-s3` →
    `bb32-p3-pc` (same `p3-prem`). MS-B `gp-p3` → `gp-p3-pc` (on `p3-s3`).
- **Judging**: AWQ judge. MS-A chains judged in full; MS-B and `p3-pc` reuse the verdicts of unchanged answers (`judge.py cache`)
  and judge the changed ones (`*-chg`, `--subset`).
- **Decision, per swap**: GROVE pooled over the three dev sets (1,340), swapped chain − entry chain, paired bootstrap; it "helps on
  dev" if the difference is > 0 with the 95% CI above 0. Also reported: text accuracy and mIoU (MS-A); false-premise questions
  answered "not present" and false alarms on grounded questions, per set (MS-B).
- **VistaQA, registered now, run whatever dev shows, analysis rows only** (compared with `final` by paired bootstrap):
  - `bb32` = MS-A through the entry's exact merge: `bb32-ans` (cfg above), `bb32-seg` and `bb32-cpt` (SAM3, `fit` as for the
    entry), `remask bb32-s3` → `override bb32-pc --parts bb32-s3 final-prem --confirm` → `countfix bb32 --parts bb32-pc bb32-cpt
    final-cnt --tau 0.8` → `check`; judged in full.
  - `final-gp` = the entry with MS-B: `gp-v` (cfg above) → `override final-gp-pc --parts final-s3 gp-v --confirm` → `countfix
    final-gp --parts final-gp-pc final-cpt final-cnt --tau 0.8` → `check`; changed answers judged (`final-gp-chg`), the rest keep
    `final`'s verdicts.
  - The VistaQA inference passes may run while dev runs; nothing on VistaQA is judged or scored before the dev results are
    committed. Re-runs only for infrastructure failures (crash, OOM, missing shard), same config.
- **Dev results** (AWQ judge). MS-A answer passes `bb32-dev` K03,K05,K07,K09 (17 s/it, 0 parse failures),
  `bb32-p2` K13,K15,K12,K08 (10 parse failures), `bb32-p3` K17,K01 (22 s/it); SAM3 passes K03,
  K07, K13, K15, K01; judged in full (K01, K03, K13). MS-B premise passes `gp-dev` K02,K04,
  `gp-p2` K06,K10, `gp-p3` K11; 8B `p3-prem` K14; changed answers judged in `p3-pc-chg` (22), `gp-p3-pc-chg` (31),
  `gp-dev-cf3-chg` (43), `gp-p2-cf3-chg` (45) on K11, K05.

  | set | entry chain | MS-A 32B answerer | MS-A − entry (95% CI) | MS-B Gemma premise | MS-B − entry (95% CI) |
  |---|---|---|---|---|---|
  | proxy dev (560) | 73.81 (text 79.82, mIoU 76.50) | 71.80 (78.04, 75.53) | −2.01 (−4.00 to −0.03) | 71.63 (76.96, 74.73) | −2.19 (−4.18 to −0.19) |
  | proxy2 (580) | 57.43 (75.52, 56.88) | 58.85 (74.14, 59.17) | +1.42 (−0.83 to +3.71) | 56.70 (75.17, 56.95) | −0.73 (−2.68 to +1.21) |
  | proxy3 (200) | 70.78 `p3-pc` (81.50, 67.72) | 72.65 (84.00, 67.18) | +1.87 (−1.01 to +4.77) | 66.97 (76.50, 64.32) | −3.82 (−8.03 to +0.26) |
  | **pooled (1,340)** | 66.27 | 66.32 | **+0.05 (−1.27 to +1.42)** | 64.47 | **−1.80 (−3.11 to −0.50)** |

  **Neither swap helps on dev.** MS-A is level: it loses on COCO (proxy dev) and gains on proxy2's figures, scenes and nuclei
  and on proxy3, none significantly. MS-B loses: premise answers changed to "not present" on false-premise questions 105 vs the
  8B's 106 of 150 (proxy dev) and 117 vs 121 of 158 (proxy2), but false alarms on grounded questions 35 vs 17 of 410
  (proxy dev), 33 vs 39 of 422 (proxy2) and 31 vs 22 of 200 (proxy3, where the 8B's own false alarms already cost 7.2 against
  no check, 77.95 → 70.78). VistaQA passes `bb32-ans` (8 accounts) and `gp-v` (K14,K16) ran meanwhile; nothing on
  VistaQA has been judged or scored. Both registered VistaQA rows are judged now.
- **VistaQA rows, as registered** (analysis only; scored once each, after the dev results were committed in `90b4911`; AWQ
  judge; paired bootstrap vs `final` 48.40, text 60.76, mIoU 52.05):
  - `final-gp` (MS-B): **49.76** (text 61.62, mIoU 52.74), **+1.35 (95% CI −0.20 to +2.93)**, not significant. Gemma changed
    239 of 314 false-premise answers (8B 222) and 120 of 843 grounded ones (8B 110); false-premise GROVE 70.9 → 78.6, grounded
    40.0 → 39.0. Premise pass `gp-v` K14,K16; 203 changed answers judged in `final-gp-chg` (K11).
  - `bb32` (MS-A): **43.38** (text 58.43, mIoU 47.33), **−5.02 (95% CI −6.92 to −3.15)**. Answer pass `bb32-ans`
    K17,K01,K02,K04,K06,K10,K14,K16 (5 parse failures vs the 27B's 14), SAM3 `bb32-seg` K03,K05 and
    `bb32-cpt` K07; 335 answers overridden, 14 counts raised;
    judged in full (K02). Grounded GROVE 40.0 → 32.3 (text 56.6 → 53.1), false-premise 70.9 → 73.2; per domain the AV
    images lose most, 67.5 → 45.8 GROVE at text accuracy 76.2 → 73.8, so the loss there is in the masks from its boxes.
  - Dev ranked the two swaps the other way round (MS-A level, MS-B −1.80); neither VistaQA difference is used for anything,
    and the entry stays `final` 48.40.
