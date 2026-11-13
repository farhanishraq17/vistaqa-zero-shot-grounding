# Grounding the Ungroundable: Answers, Masks, and Absences in Zero-Shot VQA

Code, predictions and evaluation outputs of our entry to the **VistaQA Zero-Shot Grounding Challenge, Track 1**
(VLM4RWD workshop, NeurIPS 2026).

**Authors:** anonymous for review.

**Result: GROVE 48.40** on VistaQA (text accuracy 60.76, mIoU 52.05) with the Qwen2.5-14B-Instruct-AWQ judge, and **48.14**
with the unquantized Qwen2.5-14B-Instruct judge. The system uses open weights only, with no training or fine-tuning.

## The system
Four passes, each reading only the image and the question, and three fixed merge rules.

| Pass | Model | What it produces |
|---|---|---|
| A. Answer and evidence boxes | Qwen3.6-27B (4-bit NF4) | a one-sentence answer and one box per evidence instance |
| B. Masks | SAM3 | one mask per box |
| C. Premise check | Qwen3-VL-8B-Instruct | what the question assumes and the image lacks, confirmed by a yes/no question; the answer then becomes "There is no X in the image." with no masks |
| D. Count raise | SAM3 and Qwen3-VL-8B-Instruct | on plain counting questions, a larger count from SAM3 (score at least 0.8) or from the 8B replaces the 27B's count |

| Stage | GROVE, AWQ judge | GROVE, unquantized judge | Folder |
|---|---|---|---|
| A + B | 45.46 | 45.85 | `runs/final-s3` |
| + C | 48.95 | 48.62 | `runs/final-pc` |
| **+ D (the entry)** | **48.40** | **48.14** | `runs/final` |

## 1. Setup
Open a terminal in this folder. Python 3.10 or newer; no GPU and no account are needed for steps 1 to 3.

```bash
pip install numpy scipy pillow pycocotools huggingface_hub
python -c "from huggingface_hub import snapshot_download as s; s('mozhina/VistaQA', repo_type='dataset', revision='cf0cb9857e5f0969a7fc2c8efb3ae9d4816ccc1a', local_dir='Competition Data')"
git clone https://github.com/VistaQA/VistaQA-Evaluation "Competition Data/VistaQA-Evaluation"
```

The second line downloads VistaQA (ground truth and images) to `Competition Data/VistaQA`; the third fetches the official
evaluation code.

## 2. Evaluate our submission (no GPU)
```bash
python Scripts/judge.py score final
```

After a few minutes, this prints `GROVE 48.40 | text acc 60.76 | mIoU 52.05 | n=1157` and the per-task and per-domain scores. It scores our
predictions, [`runs/final/preds.jsonl`](runs/final/preds.jsonl), against the VistaQA ground truth with the official mask
scorer and aggregation. The answer verdicts are the judge's stored ones, one 0 or 1 per question
(`runs/final/judge_out/verdicts.jsonl`). For the unquantized judge, `python Scripts/judge.py score final-fulljudge` prints
48.14.

To judge the answers again from scratch, serve the judge and run the official evaluation (one GPU with about 16 GB):

```bash
pip install vllm openai
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-14B-Instruct-AWQ --served-model-name judge --port 8001 --max-model-len 4096 &
export OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://127.0.0.1:8001/v1
python "Competition Data/VistaQA-Evaluation/src/main.py" --gt "Competition Data/VistaQA" --pred runs/final/preds.jsonl --llm_model judge --epsilon 0.01 --output_dir outputs
```

The result is written to `outputs/evaluation_results.json`. Our reported scores used the same official prompt, parser, mask
scorer and aggregation, with the judge run through vLLM's offline interface (`src/judge_kernel.py`) instead of a server.

## 3. Rebuild the submission from the passes' outputs (no GPU)
Every pass's raw output is in `runs/`, so the merge that produces the submitted predictions re-runs on a laptop:

```bash
python Scripts/fleet.py fit      final-seg --inline "Competition Data/VistaQA"
python Scripts/fleet.py fit      final-cpt --inline "Competition Data/VistaQA"
python Scripts/fleet.py remask   final-s3  --parts final-ans final-seg
python Scripts/fleet.py override final-pc  --parts final-s3 final-prem --confirm
python Scripts/fleet.py countfix final     --parts final-pc final-cpt final-cnt --tau 0.8
python Scripts/fleet.py check    final     --inline "Competition Data/VistaQA"
git status --short runs/
```

`check` prints `1157 preds | unique 1157 | missing 0 | unknown 0 | bad masks 0 | ...`, and `git status` prints nothing: the
rebuilt files are identical to the committed ones, including `runs/final/preds.jsonl` (SHA-256 as stored, with LF line
endings: `ebeaa1937f83d5a16cc7b7d35ce2fecfed42b29902dee8e13180a7fcf8fd790f`). On Windows this relies on Git's default
`core.autocrlf=true`. `fit` and `check` read only each sample's image id, width and height.

## 4. Run the system yourself (GPU)
Needs Linux with two GPUs of 15 GB or one of 24 GB (then use `"max_memory": ["21GiB"]` in the first command). The answerer
takes about 12 GPU-hours on two T4s; the other passes take about 6 hours together. Add `"limit": 20` to a `--cfg` to try a
pass on 20 questions first.

```bash
pip install torch transformers==5.16.1 accelerate bitsandbytes kagglehub
python Scripts/fleet.py push final-ans  --accounts LOCAL --cfg '{"vlm": "Qwen/Qwen3.6-27B", "quant": "nf4", "pip": ["bitsandbytes"], "max_memory": ["14GiB", "14GiB"], "seg": "", "expand_short": true}'
python Scripts/fleet.py collect final-ans
python Scripts/fleet.py push final-prem --accounts LOCAL --cfg '{"vlm": "Qwen/Qwen3-VL-8B-Instruct", "premise": "only"}'
python Scripts/fleet.py collect final-prem
python Scripts/fleet.py push final-cnt  --accounts LOCAL --cfg '{"vlm": "Qwen/Qwen3-VL-8B-Instruct", "seg": "", "expand_short": true, "plain_only": true}'
python Scripts/fleet.py collect final-cnt
python Scripts/fleet.py push final-seg  --accounts LOCAL --boxes final-ans    --cfg '{"seg": "kaggle:chonlanawawa/sam3/pyTorch/default"}'
python Scripts/fleet.py collect final-seg
python Scripts/fleet.py push final-cpt  --accounts LOCAL --concepts final-ans --cfg '{"seg": "kaggle:chonlanawawa/sam3/pyTorch/default"}'
python Scripts/fleet.py collect final-cpt
```

These are the entry's frozen configurations. Each `push` runs one pass here and now (it downloads VistaQA and the weights
itself), and `collect` writes its `runs/<pass>/preds.jsonl` and `raw.jsonl`, replacing ours, so `git diff` compares your run
with ours. Then merge with the commands of step 3 and judge the new answers with the official evaluation of step 2 (the
stored verdicts belong to our answers, so `judge.py score` does not apply to a re-run).

The entry itself ran as Kaggle kernels on two T4 GPUs each (Python 3.13, torch 2.11, transformers 5.16.1), with greedy
decoding. `LOCAL` runs the same generated script on your machine; we tested this launcher without the models. Re-running 150
questions on an RTX 3090 reproduced 147 answers byte for byte. With Kaggle credentials in `Secrets.env`
(`KAGGLE_USERNAME_ME=...`, `KAGGLE_API_TOKEN_ME=...`), `--accounts ME` runs a pass as a Kaggle kernel instead, and
`python Scripts/fleet.py status <pass>` shows its progress.

Checkpoints, at the revision each pass ran:

| Role | Checkpoint | Revision |
|---|---|---|
| Answerer (A), 4-bit NF4, fp16 compute | `Qwen/Qwen3.6-27B` | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Premise check (C) and 8B counts (D), fp16 | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| Segmenter (B, D), fp16 | `facebook/sam3`, loaded from the Kaggle Models mirror `chonlanawawa/sam3/pyTorch/default/1` (same SHA-256 files) | `3c879f39826c281e95690f02c7821c4de09afae7` |
| Judge (registered) | `Qwen/Qwen2.5-14B-Instruct-AWQ` | `539535859b135b0244c91f3e59816150c8056698` |
| Judge (cross-check) | `Qwen/Qwen2.5-14B-Instruct` | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` |
| Data | `mozhina/VistaQA` (dataset, v1.1) | `cf0cb9857e5f0969a7fc2c8efb3ae9d4816ccc1a` |

The passes load each model by its Hub id, so they get the current `main`. Every revision above is still `main` except
Qwen3.6-27B's, which gained one commit on 2026-11-07, after our runs; it changes only the chat template files, and the prompt
our passes render is identical under both.

## What is in this repository
- `runs/final/` holds the submission: `preds.jsonl` (official format, COCO RLE masks at full resolution),
  `evaluation_results.json`, `per_sample.jsonl` and the judge verdicts. `runs/final-fulljudge/` is the same submission under
  the unquantized judge.
- `runs/final-ans/`, `final-seg/`, `final-cpt/`, `final-prem/` and `final-cnt/` are the five passes, each with `manifest.json`
  (its exact configuration), `raw.jsonl` (every raw model reply) and `preds.jsonl`. `runs/final-s3/` and `runs/final-pc/` are
  the two earlier stages of the table above, with their own evaluation outputs.
- `src/pipeline.py` is the whole system in one file; `python src/pipeline.py --selftest` runs its checks locally. It also
  carries options that the entry does not use.
- `src/judge_kernel.py` is the judge. `src/build_proxy_dev.py` and the three `src/build_*_probe.py` scripts build the external
  development data.
- `Scripts/fleet.py` runs and merges passes, `Scripts/judge.py` scores, and `Scripts/kgl.py` wraps the Kaggle command line.
- `EXPERIMENTS.md` is our complete run log, with the entry's pre-registration in its section "FINAL". It also records
  development runs and later analysis systems whose outputs are not part of this repository; account names appear as
  `K01`...`K19` and `user01`..., and two shared machines as `PC1` and `PC2`.
- `report/` holds the system report (`main.pdf`) and its LaTeX source.

## Zero-shot protocol
No VistaQA sample, label or mask was used for any design decision. Every choice was made on external data built by the
`src/build_*.py` scripts (COCO, PACO-LVIS, COCO-Text, MathVista, TextVQA). At inference the pipeline reads only the image and
the question. The entry's configuration was registered in `EXPERIMENTS.md` before VistaQA was scored. Commit hashes cited there
refer to our private development repository; we can show the original commits to the organizers on request.

## External resources and license
Qwen3.6-27B, Qwen3-VL-8B-Instruct and the judges Qwen2.5-14B-Instruct-AWQ and Qwen2.5-14B-Instruct (Apache-2.0); SAM3 (SAM
License), from the Kaggle Models mirror `chonlanawawa/sam3`, whose SHA-256 matches `facebook/sam3`; bitsandbytes, vLLM,
transformers and pycocotools. VistaQA (CC BY 4.0) is used for final evaluation only. The code, tooling and report were written
with Claude Opus 5.5 as the coding assistant, under the authors' direction; no pass of the entry calls it.

Code, configurations, run log, predictions and evaluation outputs are released under MIT ([LICENSE](LICENSE)). The questions,
image ids and the task, domain, false-premise and instance-count labels in the outputs come from VistaQA (CC BY 4.0,
`mozhina/VistaQA`); no images, ground-truth answers or masks are included. Model weights are downloaded at run time and are not
included.
