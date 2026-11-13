# Results — VistaQA Track 1 (zero-shot)

**Entry: run `final`, GROVE 48.40** (text accuracy 60.76, mIoU 52.05, n = 1,157). Config pre-registered in EXPERIMENTS.md and
committed (`9c0d047`) before any VistaQA scoring; predictions committed (`905582c`) before judging. Files:
`runs/final/preds.jsonl`, `runs/final/evaluation_results.json`.

Paper baselines (quoted, not re-run): GPT-5.4-Thinking+SAM3 45.53, Qwen3-VL-4B+SAM3 43.30, Qwen3-VL-32B+SAM3 42.72.

The report's intervals were all recomputed with `Scripts/boot.py` (10,000 resamples, seed 0). Some intervals in this file come from
earlier computations and differ from the report's by at most 0.06; the report's are the reference.

## Ablation on VistaQA (reported analysis, not used to pick the entry)
| Run | System | GROVE | Text acc | mIoU |
|---|---|---|---|---|
| final-s3 | 27B answers + SAM3 box masks | 45.46 | 59.55 | 47.12 |
| final-pc | + 8B premise check | **48.95** | 61.45 | 52.32 |
| **final** (entry) | + count raise (max of 27B, 8B, SAM3, plain counts only) | **48.40** | 60.76 | 52.05 |

- The premise check adds +3.49 (95% CI +1.65 to +5.35): hits +6.81, false alarms −3.32. Hallucination 45.7 → 70.9, grounded
  45.4 → 40.8. It caught 222/314 hallucination samples but fired on 110/843 grounded ones (13.0%; proxy dev 4%, robotics 34%).
- **The count raise did not transfer: −0.55** (95% CI −1.02 to −0.13). 23 raised: 10 right → wrong, 2 wrong → right, 11 wrong
  both ways; counting 48.0 → 44.6. It was +1.5 on proxy dev (72.29 → 73.81). final-pc scores higher, but the entry stays `final`
  as pre-registered. final-pc is reported, not submitted.
- final-pc has no judge run of its own: it takes `final`'s verdict wherever the answer is identical and `final-s3`'s for the 23
  answers before the raise.

## Breakdown of the entry
| Task | GROVE | | Domain | GROVE | | Group | GROVE |
|---|---|---|---|---|---|---|---|
| ocr | 58.4 | | av | 67.5 | | hallucination (zero masks) | 70.9 |
| attribute | 52.6 | | outdoor | 55.6 | | grounded, all | 40.0 |
| identification | 50.8 | | indoor | 54.1 | | single-instance | 43.1 |
| counting | 44.6 | | science | 48.6 | | multi-instance | 32.7 |
| comparison | 43.2 | | robotics | 32.6 | | | |
| spatial | 41.8 | | math | 26.1 | | | |

Proxy dev scored 73.81 on the same system: it is COCO-only and easier, useful for relative comparisons only.

## Against the published board (vistaqa.github.io; paper Tables 2–3, 14 entries)
95% bootstrap CI of the entry over samples: 45.95–50.90. Challenge results of other teams are not public.

| | Entry | Best published | Gap | Rank of 15 |
|---|---|---|---|---|
| **Overall** | **48.40** | 45.53 GPT-5.4-T + SAM3 | +2.87 | **1** |
| Hallucination | 70.88 | 60.72 Qwen3-VL-4B + SAM3 | +10.16 | 1 |
| Grounded | 40.03 | 44.65 GPT-5.4-T + SAM3 | −4.62 | 3 |
| OCR | 58.37 | 44.16 GPT-5.4-T + SAM3 | +14.21 | 1 |
| Attribute | 52.61 | 47.46 Qwen3-VL-4B + SAM3 | +5.15 | 1 |
| Comparison | 43.24 | 38.93 GPT-5.4-T + SAM3 | +4.31 | 1 |
| Spatial | 41.75 | 38.86 GPT-5.4-T + SAM3 | +2.89 | 1 |
| Identification | 50.77 | 53.62 GPT-5.4-T + SAM3 | −2.85 | 2 |
| Counting | 44.57 | 52.21 GPT-5.4-T + SAM3 | −7.64 | 5 |
| AV | 67.55 | 59.39 Qwen3-VL-4B + SAM3 | +8.16 | 1 |
| Science | 48.63 | 39.87 Gemini 3 + SAM3 | +8.76 | 1 |
| Indoor | 54.09 | 45.39 GPT-5.4-T + SAM3 | +8.70 | 1 |
| Outdoor | 55.65 | 47.63 Qwen3-VL-32B + SAM3 | +8.02 | 1 |
| Math | 26.14 | 41.51 GPT-5.4-T + SAM3 | −15.37 | 5 |
| Robotics | 32.60 | 47.01 GPT-5.4-T + SAM3 | −14.41 | 6 |

Versus GPT-5.4-T + SAM3 the whole lead is hallucination (+6.24 points of the total); grounded questions cost −3.37. Overall text
accuracy is level (60.76 vs 61.02; Gemini 3 62.32) but built differently: hallucination 71.97 vs 39.17, grounded 56.58 vs 69.16
(62.99 in final-s3, before the premise check and count raise). Masks are better: mIoU 52.05 vs 41.98, grounded 41.8 vs 32.59.
Inference cost: 17.9 GPU-hours on free Kaggle T4s, open weights, 4-bit.

## Judge cross-check (pre-registered 2026-11-04, predictions unchanged)
The same predictions re-judged by the unquantized Qwen2.5-14B-Instruct, the paper's judge (official prompt and parser, fp16 on T4x2):
| Run | AWQ judge (registered) | Full judge | Text acc AWQ / full |
|---|---|---|---|
| **final** (entry) | **48.40** | **48.14** | 60.76 / 60.59 |
| final-pc | 48.95 | 48.62 | 61.45 / 61.19 |
| final-s3 | 45.46 | 45.85 | 59.55 / 60.24 |

56 of 1,157 verdicts flip on `final`: grounded 24 up, 10 down; hallucination 3 up, 19 down, so hallucination goes 70.9 → 66.8 and
grounded 40.0 → 41.2. Domains: av 68.4, indoor 53.3, math 27.8, outdoor 55.7, robotics 30.1, science 48.1. The entry's score stays
48.40 as pre-registered; 48.14 is the number closest to the paper's judge, +2.61 over the best published entry. Against the published
board the ranks stay the same except counting (42.1: 5th → 6th, below Gemini 3's 42.81) and math (27.8: 5th → 4th, above Qwen3-VL-4B's 27.37).

## v2, the post-freeze system (pre-registered analysis, not the entry)
Designed after seeing the entry's VistaQA aggregates, chosen on proxy dev and proxy2 only, registered (`b7343be`) before any VistaQA
run; predictions committed (`a1c0073`) before judging. AWQ judge; full judge in brackets where run.
| Stage | proxy2 | VistaQA v2 (v3, v4, v5) | VistaQA, entry's same stage |
|---|---|---|---|
| answers + SAM3 box masks | 57.42 | 42.43 [42.63] | 45.46 [45.85] (`final-s3`, Qwen3.6) |
| + routed figure evidence (353 figures) | 60.04 | 42.18 [42.32] | — |
| + premise check + SAM3 veto τ 0.4 (252 overridden) | 65.31 | 48.25 [47.30] | 48.95 [48.62] (`final-pc`, no veto) |
| + count raise, masks kept (**v2**) | 65.87 | **48.01** [47.01] | **48.40** [48.14] (entry) |
| + thinking on 126 routed reasoning questions (**v3**) | 65.87 | **48.21** [47.29] | — |
| + calculator tool where v3's thinking hit 4096 tokens (**v4**) | 65.87 | **48.37** [47.37] | — |
| + 4.19 MP answerer input on the 536 large images (**v5**) | 65.87 | **48.03** [47.34] | — |

- **v2 − entry = −0.40** (95% CI −2.38 to +1.55) with the AWQ judge, −1.13 with the full judge (47.01 vs 48.14): level or worse, not better. Proxy dev 74.23 vs 73.81, proxy2 65.87 vs 57.43 did not transfer.
- **The backbone swap reversed**: Qwen3.8 was +1.7 on proxy2 and +0.9 on proxy dev, but −3.0 on VistaQA at the first stage
  (hallucination 45.7 → 37.9, math 25.1 → 19.9, science 41.5 → 35.8, outdoor 55.2 → 50.9).
- **Routed figure evidence did nothing** (−0.25; math 19.9 → 19.6). Its +2.6 on proxy2 came from synthetic charts.
- **The veto trades hits for false alarms, for a small net gain**: "not present" answers on grounded questions 110 → 64 of 843,
  on hallucination questions 225 → 189 of 314 (on proxy2 it lost only 7 of 151 hits). Counterfactuals, registered after v2's
  score and before computing them (cached verdicts, no new inference): v2 without the veto **47.79** [46.95] (veto +0.22, full judge +0.06, CI −0.72 to
  +1.17); the entry with the veto and kept masks, `final-v`, **49.03** (+0.56 over `final-km`, CI −0.30 to +1.44). Post-hoc
  analysis only (full judge: `final-v` 48.74): `final-v` is not the entry and was not selected by anything. So v2's shortfall is the backbone (v2 without the
  veto vs the entry with the mask fix: −0.68), not the veto.
- Count raise with masks kept: −0.24 (the entry's raise −0.55; `final-km` 48.47).
- `v2-top1` (first mask only on non-counting answers): 47.18 [46.14], −0.83: cutting to one mask hurts, so over-masking is not the fix.
- **v3** (registered before v2's score, predictions `b8d7e13` committed before judging): v2 plus Qwen3.8 thinking on the 126 of 353
  figure questions a second router called multi-step. **48.21** [47.29]: **+0.21 over v2** (95% CI −0.35 to +0.75; full judge
  +0.28, −0.07 to +0.66), −0.19 vs the entry (−2.17 to +1.80; full judge −0.85). 99 traces closed, 27 hit 4096 tokens and kept the no-think answer; 77 answers changed, 22 right gained,
  8 lost; math 21.9 → 24.7, masks identical to v2. 48.35 with v2's verdicts on the unchanged answers (6 judge flips). Thinking
  helps the questions it targets, but they are too few to move the total.
- **v4** (R4, registered `da87af8` before any VistaQA run, predictions `eeed5af` committed before judging): v3 plus a calculator
  fallback (Qwen3.8 writes a short Python program, run in a sandbox) on the 27 questions whose thinking trace hit 4096 tokens.
  The program ran on 21 and changed 17 answers: 3 right gained, 3 lost (full judge 4 and 2). **48.37** [47.37]: +0.16 over v3
  (95% CI −0.23 to +0.56; full judge +0.08, −0.19 to +0.34), −0.03 vs the entry (full judge −0.77). The AWQ gain is 5 judge flips
  on unchanged answers (48.18 with v3's verdicts there): level with v3. MathVista probe 146 vs 143/200 did not show up at this size.
- **v5** (R6, registered `834dd4a`, predictions `f67941c` committed before judging): v4 with the answerer's input cap raised from
  1280² to 2048² on the 536 images above 1.64 MP (none of proxy dev or proxy2 is; on a V*Bench probe it gained 16/238 answers and
  5 points of box hits). 6 full-cap images ran out of T4 memory and fall back to the 1280² pass (as-run with 6 empty answers:
  48.11 [47.40], higher only because the judge calls some empty answers correct). 156 answers changed: 22 right gained, 23 lost
  (full judge 24 / 21); the redone masks −0.11. **48.03** [47.34]: −0.34 vs v4 (95% CI −1.26 to +0.59; full judge −0.03), −0.38
  vs the entry. Level with v4: the probe's gain did not transfer.
- Dev-only rounds, not run on VistaQA: R1 answer selector, R2 mask dedup, R3 premise AND, R7 mask cleanup (fill holes +0.43 on
  proxy dev but −0.05 on proxy2; largest component hurts both). R5 dropped (no newer open model that fits 2x24 GB).

## Weight-space programming (post-close analysis, user request; not the entry)
Claude Opus 5.5 edited Qwen3-VL-8B-Instruct's weights directly (no gradients) to make its single-pass answer use the
"object is absent" signal it already carries: a rank-1 amplification of the layer-21 absent-vs-present direction in what decoder
layers 18–20 write (6 matrices) plus a bias on layer 20's down_proj (0.3 class gap). Chosen on a COCO practice set only (21 of 40
allowed candidates, practice false-premise denials .75 → .85, false alarms .04 → .05), frozen (`8bcc9fe`) before any dev run;
rules and log in EXPERIMENTS.md. AWQ judge; unchanged answers keep the unedited arm's verdicts.

| set | unedited 8B + SAM3 | edited | diff (95% CI) | both + 8B premise check | diff (95% CI) |
|---|---|---|---|---|---|
| proxy dev (560) | 58.42 | 61.66 | +3.24 (+1.91 to +4.73) | 67.58 → 68.01 | +0.43 (+0.00 to +0.96) |
| proxy2 (580) | 48.35 | 50.27 | +1.92 (+0.86 to +3.14) | 52.60 → 52.73 | +0.13 (−0.29 to +0.61) |
| pooled dev (1,140) | 53.30 | 55.86 | +2.57 (+1.71 to +3.50) | 59.96 → 60.24 | +0.28 (−0.03 to +0.62) |
| **VistaQA (1,157)** | 38.76 | **40.46** | **+1.70 (+0.83 to +2.60)** | 44.35 → 44.90 | +0.56 (−0.02 to +1.18) |

- The edit transfers: on VistaQA, false-premise GROVE 39.7 → 45.5 with grounded unchanged (38.4 → 38.6); 27 verdicts gained, 8 lost.
- It does about 30% (dev 39%) of what the prompt-based premise check does, without the second pass; once the check runs, it adds
  nothing measurable. The entry, whose answers come from the 27B, would not change.

## Mask verifier (post-close analysis, user request; not the entry)
Set-of-Mark check on the entry's masks: for predictions with ≥ 2 masks on non-counting questions, each mask's box is drawn and
numbered, and a VLM lists the boxes that are "visual evidence required to support this answer"; unlisted masks are dropped (no
valid list = keep all). Answers never change, so verdicts are reused. Rules, the new crowded-scene dev set proxy3 (200 RefCOCOg
val questions) and the VistaQA row were registered (`3736b93`) before any result; dev results were committed (`e1d96db`) before
the VistaQA run. AWQ judge.

| set | base | 27B check (chosen) | diff (95% CI) | 8B check | diff (95% CI) | over-masked right answers (27B) |
|---|---|---|---|---|---|---|
| proxy dev (560) | 73.81 | 73.33 | −0.48 (−0.79 to −0.21) | 73.13 | −0.69 (−1.10 to −0.33) | 1 → 0 |
| proxy2 (580) | 57.43 | 57.05 | −0.38 (−0.74 to −0.05) | 57.03 | −0.40 (−0.77 to −0.06) | 29 → 22 |
| proxy3 (200) | 77.95 | 77.85 | −0.10 (−1.04 to +0.82) | 76.63 | −1.32 (−2.60 to −0.22) | 19 → 12 |
| pooled dev (1,340) | 67.34 | 66.96 | −0.38 (−0.63 to −0.15) | 66.69 | −0.65 (−0.95 to −0.38) | 49 → 34 |
| **VistaQA (1,157)** | 48.40 | **48.61** | **+0.21 (−0.13 to +0.53)** | — | — | 169 → 140 |

- Dev: the check fixes about 30% of over-masked answers, but 32 of the 72 predictions it changes already had the GT's mask count;
  in 26 it keeps one of the two objects of "Which appears larger, the A or the B?" or "Is the A left or right of the B?", whose
  GT has both. It does not help on dev by the registered rule.
- VistaQA (registered analysis row `final-mv`): mIoU 52.05 → 52.50, 8 drops on predictions with the right mask count, GROVE +0.21,
  not significant. Over-masking is 6× more common on VistaQA than on the dev sets, so dev under-predicted the check. The entry stays 48.40.

## Model swaps (post-close analysis, user request; not the entry)
Two single-model swaps in the entry's chain, each alone, same prompts, nf4 like the 27B: Qwen3-VL-32B-Instruct as the answerer
(MS-A) and Gemma 4 31B as the premise checker (MS-B). Rules and VistaQA rows registered (`12ac167`) before any result; dev
results committed (`90b4911`) before any VistaQA row was judged. AWQ judge; on proxy3 the entry chain includes the 8B premise
check (70.78; 77.95 without it).

| set | entry chain | MS-A 32B answerer | diff (95% CI) | MS-B Gemma premise | diff (95% CI) |
|---|---|---|---|---|---|
| proxy dev (560) | 73.81 | 71.80 | −2.01 (−4.00 to −0.03) | 71.63 | −2.19 (−4.18 to −0.19) |
| proxy2 (580) | 57.43 | 58.85 | +1.42 (−0.83 to +3.71) | 56.70 | −0.73 (−2.68 to +1.21) |
| proxy3 (200) | 70.78 | 72.65 | +1.87 (−1.01 to +4.77) | 66.97 | −3.82 (−8.03 to +0.26) |
| pooled dev (1,340) | 66.27 | 66.32 | +0.05 (−1.27 to +1.42) | 64.47 | −1.80 (−3.11 to −0.50) |
| **VistaQA (1,157)** | 48.40 | **43.38** `bb32` | **−5.02 (−6.92 to −3.15)** | **49.76** `final-gp` | **+1.35 (−0.20 to +2.93)** |

- Neither swap helps on dev by the registered rule (CI above 0).
- MS-B on VistaQA: false-premise answers changed 239 of 314 (8B 222), grounded 120 of 843 (8B 110); false-premise GROVE
  70.9 → 78.6, grounded 40.0 → 39.0. Not significant; on dev its extra false alarms cost more than its catches gained.
- MS-A on VistaQA: text 60.76 → 58.43, mIoU 52.05 → 47.33. Grounded GROVE 40.0 → 32.3; driving scenes (AV) lose most, 67.5 →
  45.8 GROVE at text accuracy 76.2 → 73.8, so the loss there is in the masks from its boxes. Proxy2's BDD100K questions did
  not show this.
- Dev ranked the swaps the other way round from VistaQA; neither VistaQA row is used for anything. The entry stays 48.40.

## Where the points are lost (post-hoc, aggregate; not used for any change to the entry)
51.60 points lost in total: grounded wrong answer 30.43; grounded right but imperfect masks 11.94; hallucination wrong answer 7.28;
grounded right with no mask 1.32; hallucination right but masks 0.62. Of 460 right grounded answers with masks, 271 have the exact
mask count (mean S 0.74), 169 have too many (0.33), 20 too few (0.27). Proxy dev had 1 such over-masked case, so it could not show this.

## Caveats to disclose
- **Bug in the entry's count raise, found after scoring**: the pre-registered 8B count run has `seg: ""` (no masks), and `merge_count`
  takes the masks of the winning counter, so 11 of the 23 raised answers lost all their masks and can score at most 0.1. Proxy dev's 8B
  run had masks, so proxy dev could not show it. The entry is unchanged. Fixed in the registered analysis row `final-km` (same text,
  `countfix --keep-masks`): 48.47 AWQ / 48.21 full judge, so the bug costs only 0.07 of the −0.55; the rest is wrong raised counts.
- Judge: Qwen2.5-14B-Instruct-**AWQ** (4-bit, vLLM on T4×2) with the official prompt and parser, not the unquantized model of the
  paper, so the registered 48.40 is not exactly comparable with paper numbers (see the cross-check above). Harness checks, run on VistaQA GT before any system was scored: GT-as-pred
  100.00; shuffled answers text acc 1.21%. The judge is not fully deterministic across kernels: 3 flips on 802 identical answers,
  which moves the entry by at most 0.06.
- Amendment before scoring: image files of ids 138 and 496 are smaller than their JSON declares; 3 masks in the entry were resized
  (nearest) to the declared size. No prompt, threshold or merge change.
- SAM3 weights come from a Kaggle Models mirror (`chonlanawawa/sam3`, labelled MIT; SHA-256 identical to facebook/sam3, which is
  under the SAM License).
