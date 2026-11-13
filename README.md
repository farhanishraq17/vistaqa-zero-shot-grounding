# Grounding the Ungroundable: Answers, Masks, and Absences in Zero-Shot VQA

Our system for the **VistaQA Zero-Shot Grounding Challenge, Track 1** (VLM4RWD workshop, NeurIPS 2026).

**Authors:** anonymous for review.

This repository holds the code, frozen configurations, run log, raw model outputs, predictions and evaluation outputs of our entry,
so that its score can be verified and the system re-run. The system report is [`report/main.pdf`](report/main.pdf) (LaTeX source in
[`report/`](report/)); its Table 1 is the full published benchmark with our entry, and its appendices hold the prompts,
implementation and reproduction details, a breakdown of the entry, judge checks, the development record, the post-freeze systems v2–v5,
three further post-freeze tests and dev-to-VistaQA transfer. Every VistaQA run we scored is in the run log below.

**Entry: `final`, GROVE 48.40** (text accuracy 60.76, mIoU 52.05; Qwen2.5-14B-Instruct-AWQ judge, official prompt and parser),
**48.14** with the unquantized Qwen2.5-14B-Instruct judge. Qwen3.6-27B runs in 4-bit NF4, Qwen3-VL-8B and SAM3 in fp16, all on
free Kaggle T4 GPUs; no training or fine-tuning.

| Run | System | GROVE (AWQ judge) | Full judge |
|---|---|---|---|
| **`final`** (entry) | Qwen3.6-27B answers + boxes → SAM3 masks; Qwen3-VL-8B premise check; count raise | **48.40** | 48.14 |
| `v2` (post-freeze analysis) | Qwen3.8-27B, figure router, SAM3 veto on the premise check | 48.01 | 47.01 |
| `v3` (post-freeze analysis) | v2 + thinking on routed multi-step questions | 48.21 | 47.29 |
| `v4` (post-freeze analysis) | v3 + calculator tool where thinking ran out of tokens | 48.37 | 47.37 |
| `v5` (post-freeze analysis) | v4 + 4 MP answerer input on large images | 48.03 | 47.34 |

v2–v5 were each registered before their own VistaQA run, after the entry's score was known. They are analysis and never replace `final`.

**Weight-space programming** (analysis after development closed; report Appendix I): Claude Opus 5.5 edited
Qwen3-VL-8B-Instruct's weights directly, with no gradients, so that its single-pass answers deny absent objects more often. The
edit was chosen on COCO practice questions only (`src/build_wsp_practice.py`, `src/wsp*.py`). It is stored in
`runs/wsp-edit.json` and applied at load by `pipeline.apply_edit` (cfg `edit`). 8B-only arms with SAM3 masks:

- VistaQA `v-wsp0-s3` 38.76 → `v-wsp1-s3` 40.46, +1.70 (95% CI +0.83 to +2.60).
- With the 8B premise check on both arms: +0.56 (CI −0.02 to +1.18).
- Dev sets pooled: +2.57 and +0.28.

**Mask verifier** (analysis after development closed; report Appendix J): a Set-of-Mark check of the entry's masks. Each mask's
box is drawn with a number, and a VLM lists the boxes that are evidence for the answer; masks it does not list are dropped, and
answers never change (`pipeline.MARKS`, `fleet.py push --marks`, `fleet.py verify`). It was tested on a new crowded-scene dev set,
proxy3 (`src/build_proxy3.py`, 200 RefCOCOg questions).

- Dev sets pooled (1,340): the 27B check −0.38 (95% CI −0.63 to −0.15), the 8B check −0.65.
- VistaQA `final-mv` 48.61 vs `final` 48.40: +0.21 (CI −0.13 to +0.53). Over-masked right answers fell from 169 to 140.

**Model swaps** (analysis after development closed; report Appendix K): two single-model swaps in the entry's chain, each 4-bit
with the same prompts. Qwen3-VL-32B-Instruct replaced the 27B answerer, and Gemma 4 31B replaced the 8B premise checker.

- Qwen3-VL-32B answerer: dev sets pooled +0.05 (95% CI −1.27 to +1.42); VistaQA `bb32` 43.38, −5.02 (CI −6.92 to −3.15).
- Gemma 4 31B premise checker: dev sets pooled −1.80 (CI −3.11 to −0.50); VistaQA `final-gp` 49.76, +1.35 (CI −0.20 to +2.93).

## Verification files
- Entry predictions (official format, COCO RLE masks at full resolution): [`runs/final/preds.jsonl`](runs/final/preds.jsonl),
  SHA-256 `ebeaa1937f83d5a16cc7b7d35ce2fecfed42b29902dee8e13180a7fcf8fd790f` (as stored, LF line endings)
- Its evaluation output: [`runs/final/evaluation_results.json`](runs/final/evaluation_results.json),
  per-sample scores [`runs/final/per_sample.jsonl`](runs/final/per_sample.jsonl), unquantized judge in `runs/final-fulljudge/`
- The entry's five passes: `runs/final-ans/`, `runs/final-seg/`, `runs/final-cpt/`, `runs/final-prem/`, `runs/final-cnt/`, each
  with `manifest.json` (exact config and shard layout), `raw.jsonl` (every raw model reply) and `preds.jsonl`
- Every other run: `runs/<run>/` (`preds.jsonl`, `raw.jsonl` model outputs, `manifest.json` config, `evaluation_results.json`)
- Run log with every configuration and result, including the pre-registrations: [`EXPERIMENTS.md`](EXPERIMENTS.md)
  (the entry: section "FINAL"); final numbers: [`RESULTS.md`](RESULTS.md)

## Verify the entry without a GPU
The five passes' outputs are in this repository, so the merge that turns them into the submitted predictions re-runs on a laptop
(Python 3 with `numpy`, `pillow`, `pycocotools` and `huggingface_hub`; no credentials needed):

```bash
python -c "from huggingface_hub import snapshot_download as s; s('mozhina/VistaQA', repo_type='dataset', revision='cf0cb9857e5f0969a7fc2c8efb3ae9d4816ccc1a', local_dir='Competition Data')"
python Scripts/fleet.py fit      final-seg --inline "Competition Data/VistaQA"
python Scripts/fleet.py fit      final-cpt --inline "Competition Data/VistaQA"
python Scripts/fleet.py remask   final-s3  --parts final-ans final-seg
python Scripts/fleet.py override final-pc  --parts final-s3 final-prem --confirm
python Scripts/fleet.py countfix final     --parts final-pc final-cpt final-cnt --tau 0.8
python Scripts/fleet.py check    final     --inline "Competition Data/VistaQA"
git status --short runs/
```

`check` prints `1157 preds | unique 1157 | missing 0 | unknown 0 | bad masks 0 | ...`, and `git status` prints nothing: every file
the merge rewrites is identical to the committed one, including `runs/final/preds.jsonl` (on Windows, where Python writes CRLF,
this relies on Git's default `core.autocrlf=true`). `fit` and `check` read only each
sample's image id, width and height. `runs/final/preds.jsonl` can then be scored with the official evaluation code
([VistaQA-Evaluation](https://github.com/VistaQA/VistaQA-Evaluation), `python main.py --gt ... --pred runs/final/preds.jsonl ...`).
Our scores used the official judge prompt, parser, mask scorer and aggregation, with the judge LLM served by vLLM
(`Scripts/judge.py`, `src/judge_kernel.py`).

## Zero-shot protocol
No VistaQA sample, label or mask was used for any design decision. Every choice was made on external proxy sets built by
`src/build_*.py` (COCO, PACO-LVIS, COCO-Text, MathVista, TextVQA, BDD100K, BOP, DSB 2018, RefCOCOg, V*Bench). At inference the
pipeline reads only the image and the question. Each VistaQA run was pre-registered in `EXPERIMENTS.md` before it ran. Commit
hashes cited there refer to our private development repository. Its history holds account details, so this copy starts a fresh
history; we can show the original commits to the organizers on request.

## Layout
- `src/pipeline.py` — the whole system in one file. A `CFG` dict is prepended per pass, and `python src/pipeline.py --selftest` runs locally.
- `src/judge_kernel.py` — the LLM judge (vLLM, official prompt and parser).
- `src/build_*.py` — builders of the external dev sets and probes (proxy dev, proxy2, proxy3 and the text-only probes).
- `Scripts/fleet.py` — shards passes over Kaggle accounts or GPU machines, and merges passes (`remask`, `override`, `countfix`,
  `think`, `fit`, `check`, `verify`). `Scripts/kgl.py` runs the Kaggle CLI per account. `Scripts/judge.py` handles judging and scoring;
  `Scripts/boot.py` gives the paired bootstrap CIs.
- `report/` — LaTeX source (`main.tex`, `fig_pipeline.tex`, `diagrams/`, `refs.bib`, `neurips_2026.sty`, `qual.pdf`) and PDF of the
  system report. `pdflatex main; bibtex main; pdflatex main; pdflatex main` builds it.

Kaggle account names are replaced by `K01`…`K19` and Kaggle usernames by `user01`…; `PC1` and `PC2` are two shared RTX 3090 machines.

## Re-running the passes (GPU)
**Environment.** Every pass of the entry ran as a Kaggle GPU kernel: 2× Tesla T4 (15 GB each), Kaggle's Python image of November
2026 (we recorded Python 3.13, torch 2.11 and transformers 5.16.1; pycocotools, huggingface_hub and kagglehub come preinstalled).
The kernel pip-installs bitsandbytes (answerer) and vLLM (judge) at run time. Decoding is greedy. Re-running 150 samples on an
RTX 3090 (torch 2.8, transformers 5.16.1) instead of Kaggle T4s reproduced 147 answers byte for byte.

GPU time, summed over shards without model loading: answerer about 12 h (37 s per sample; we used 8 shards), premise check 3.2 h,
8B counts 1.8 h (97 plain counting questions), SAM3 masks 0.9 h, SAM3 counts 0.2 h.

**Checkpoints**, at the revision each pass ran:

| Role | Checkpoint | Revision |
|---|---|---|
| Answerer (A), 4-bit NF4, fp16 compute | `Qwen/Qwen3.6-27B` | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Premise check (C) and 8B counts (D), fp16 | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| Segmenter (B, D), fp16 | `facebook/sam3`, loaded from the Kaggle Models mirror `chonlanawawa/sam3/pyTorch/default/1` (same SHA-256 files) | `3c879f39826c281e95690f02c7821c4de09afae7` |
| Judge (registered) | `Qwen/Qwen2.5-14B-Instruct-AWQ` | `539535859b135b0244c91f3e59816150c8056698` |
| Judge (cross-check) | `Qwen/Qwen2.5-14B-Instruct` | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` |
| Data | `mozhina/VistaQA` (dataset, v1.1) | `cf0cb9857e5f0969a7fc2c8efb3ae9d4816ccc1a` |

The kernels load each model by its Hub id, so they get the current `main`. Every revision above is still `main` except
Qwen3.6-27B's, which gained one commit on 2026-11-07, after our runs. That commit changes only `chat_template.jinja` (generation
tags around assistant turns) and `tokenizer_config.json` (a response template); the weights are unchanged, and the prompt our
passes render (one user turn plus the generation prompt, thinking on or off) is identical under both templates. To pin it anyway,
download that revision and set `"vlm"` to the local folder; `from_pretrained` accepts a path.

**Steps.**
1. Data as in the section above, and the official evaluation code:
   `git clone https://github.com/VistaQA/VistaQA-Evaluation "Competition Data/VistaQA-Evaluation"`.
2. Kaggle credentials in `Secrets.env` at the repository root, one pair per account (any alias name):
   ```
   KAGGLE_USERNAME_ME=<your Kaggle username>
   KAGGLE_API_TOKEN_ME=<your Kaggle API token>
   ```
   `python Scripts/kgl.py check` tests them. Each `--accounts` entry is one shard; an alias may repeat.
3. Passes, with the frozen configurations of `EXPERIMENTS.md` "FINAL" (also in each pass's `manifest.json`):
   ```bash
   python Scripts/fleet.py push final-ans  --accounts ME,... --cfg '{"vlm": "Qwen/Qwen3.6-27B", "quant": "nf4", "pip": ["bitsandbytes"], "max_memory": ["14GiB", "14GiB"], "seg": "", "expand_short": true}'
   python Scripts/fleet.py push final-prem --accounts ME,... --cfg '{"vlm": "Qwen/Qwen3-VL-8B-Instruct", "premise": "only"}'
   python Scripts/fleet.py push final-cnt  --accounts ME,... --cfg '{"vlm": "Qwen/Qwen3-VL-8B-Instruct", "seg": "", "expand_short": true, "plain_only": true}'
   python Scripts/fleet.py status  final-ans    # until every shard has finished, then
   python Scripts/fleet.py collect final-ans    # -> runs/final-ans/preds.jsonl and raw.jsonl; the same for every pass
   # the SAM3 passes read final-ans's boxes and evidence labels, so they run after it is collected
   python Scripts/fleet.py push final-seg --accounts ME,... --boxes final-ans    --models chonlanawawa/sam3/pyTorch/default/1 --cfg '{"seg": "kaggle:chonlanawawa/sam3/pyTorch/default"}'
   python Scripts/fleet.py push final-cpt --accounts ME     --concepts final-ans --models chonlanawawa/sam3/pyTorch/default/1 --cfg '{"seg": "kaggle:chonlanawawa/sam3/pyTorch/default"}'
   ```
   `push` and `collect` overwrite the committed `runs/<pass>/` files, so `git diff` compares a re-run with ours. Each shard is one
   self-contained script (`CFG = {...}` followed by `src/pipeline.py`) that writes `pred_shard<k>.jsonl` and `raw_shard<k>.jsonl`
   to `CFG["out"]`. `fleet.py` can also run shards over SSH on any CUDA machine (`PC1`/`PC2` in `SuperComp.env`: `<ALIAS>_USER`,
   `<ALIAS>_HOST`, `<ALIAS>_WORKSPACE`; environment from `Scripts/dl_setup.sh`; `--accounts PC1:0+1` picks GPUs 0 and 1).
4. Merge: the `fit`, `remask`, `override`, `countfix` and `check` commands of the section above.
5. Score: `python Scripts/judge.py push final --account ME` (vLLM and Qwen2.5-14B-Instruct-AWQ on a Kaggle T4×2); once
   `python Scripts/judge.py status final` shows it finished, `python Scripts/judge.py score final` runs the official mask scorer
   and aggregation and writes
   `runs/final/evaluation_results.json`.

## External resources
The entry: Qwen3.6-27B and Qwen3-VL-8B-Instruct (Apache-2.0); SAM3 (SAM License), from the Kaggle Models mirror
`chonlanawawa/sam3`, whose SHA-256 matches `facebook/sam3`; judges Qwen2.5-14B-Instruct-AWQ and Qwen2.5-14B-Instruct (Apache-2.0);
bitsandbytes, vLLM, transformers, pycocotools. Development and analysis only (report Appendix C): Qwen3.5-9B, Qwen3.6-35B-A3B,
Qwen3-VL-4B-Instruct, Qwen3-VL-32B-Instruct, Qwen3.8-27B, Gemma 4 31B and SAM2.1-L (Apache-2.0); Claude Opus 5.5 (Anthropic) as
the weight editor of Appendix I; the dev-set sources listed under "Zero-shot protocol". VistaQA (CC BY 4.0) is used for final
evaluation only. The code, tooling and report were written with Claude Opus 5.5 as the coding assistant, under the authors'
direction; no pass of the entry calls it.

## Citation
```bibtex
@misc{anonymous2026grounding,
  title        = {Grounding the Ungroundable: Answers, Masks, and Absences in Zero-Shot {VQA}},
  author       = {Anonymous},
  year         = {2026},
  note         = {System report, VistaQA Challenge Track 1, VLM4RWD Workshop at NeurIPS 2026}
}
```

## License
Code, configs, run log, predictions and evaluation outputs: MIT ([LICENSE](LICENSE)). The questions, image ids and the task,
domain, false-premise and instance-count labels in the outputs come from VistaQA (CC BY 4.0, `mozhina/VistaQA`); no images,
ground-truth answers or masks are included. Model weights are downloaded at run time and are not included (Qwen: Apache-2.0;
SAM3: SAM License).
