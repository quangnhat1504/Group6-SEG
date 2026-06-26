# Local Small-LLM Router Training (no Colab)

Trains the QLoRA query router **entirely on local hardware** (tested target:
RTX 5070 Ti, 16 GB, Blackwell / sm_120) and produces a **clean train/dev/test
protocol** so the calibration is no longer tuned on the test split against itself.

## Why this exists

The earlier setup generated predictions on Colab and then split the **test**
prediction file into 150 calibration / 150 eval queries — calibration touched the
same 300 test queries it was later evaluated on. The local pipeline fixes this:

| Role      | Split      | Used for                                    |
|-----------|------------|---------------------------------------------|
| train     | `trainfit` | fine-tuning the QLoRA router                |
| dev       | `dev`      | tuning calibration (bias/temp/margin)       |
| test      | `test`     | final evaluation only (never tuned on)      |

`trainfit` + `dev` are a stratified split of the original 809 `train` queries, so
`dev` is held out from BOTH fine-tuning and the test set.

## One-shot

```bash
# from repo root, using the project venv
.venv/Scripts/python.exe scripts/run_local_router_pipeline.py --config configs/scifact.yaml
```

Add `--dry-run` to print the commands without executing, `--skip-splits` to reuse
existing `trainfit`/`dev` splits, `--skip-train` to reuse an adapter, and
`--no-4bit` to force bf16 LoRA.

## Manual steps (equivalent)

```bash
# 1) carve trainfit + dev out of train (stratified by oracle label)
.venv/Scripts/python.exe scripts/make_local_router_splits.py --config configs/scifact.yaml

# 2) fine-tune on trainfit, score dev + test (writes *_llm_router_predictions_local.csv)
.venv/Scripts/python.exe scripts/train_qlora_router_local.py --config configs/scifact.yaml \
    --train-split trainfit --eval-splits dev test

# 3) calibrate on dev, evaluate on test (genuine held-out)
.venv/Scripts/python.exe scripts/calibrate_llm_router_scores.py --config configs/scifact.yaml \
    --calibration-split dev  --calibration-predictions runs/scifact/dev_llm_router_predictions_local.csv \
    --eval-split        test --eval-predictions        runs/scifact/test_llm_router_predictions_local.csv \
    --name "Small LLM QLoRA Router Calibrated (local dev->test)"

# 4) raw (uncalibrated) test metrics
.venv/Scripts/python.exe scripts/evaluate_llm_router_predictions.py --config configs/scifact.yaml \
    --split test --predictions runs/scifact/test_llm_router_predictions_local.csv \
    --name "Small LLM QLoRA Router LogProb (local)"
```

## Dependencies

The retrieval stack is already in the venv. Training additionally needs:

```bash
.venv/Scripts/python.exe -m pip install \
    "transformers>=4.44" "peft>=0.11" "accelerate>=0.30" "datasets>=2.19" "bitsandbytes>=0.43"
```

## 4-bit vs bf16 on Blackwell

`train_qlora_router_local.py` tries 4-bit QLoRA (nf4) first and **auto-falls back
to bf16 LoRA** if `bitsandbytes` is missing or the 4-bit kernels do not load on
sm_120. Qwen2.5-0.5B fits in 16 GB in bf16, so the fallback does not change the
method's intent — the prediction CSV schema is identical either way. The chosen
mode is recorded in `runs/scifact/qlora_router_local/train_summary.json`.

## Outputs

- `runs/scifact/dev_llm_router_predictions_local.csv`
- `runs/scifact/test_llm_router_predictions_local.csv`
- `runs/scifact/qlora_router_local/adapter/` (LoRA adapter)
- `runs/scifact/qlora_router_local/train_summary.json`
- calibrated `test_*_calibrated_{metrics.json,predictions.csv,routed.csv}`
- raw `test_*_metrics.json`

## Reproducibility

Seed `13` controls both the trainfit/dev split and the training seed. Re-running
with the same seed reproduces the split; LLM fine-tuning is not bit-exact on GPU
but is stable in aggregate metrics.
