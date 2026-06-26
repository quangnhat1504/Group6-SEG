# SEG Scientific Retrieval

Codebase for the SEG301m project: uncertainty-aware and cost-efficient multi-stage retrieval for scientific paper search.

The first target is SciFact using only paper title and abstract. The pipeline is intentionally scoped for student hardware:

1. Build BM25, dense, and RRF hybrid retrieval baselines.
2. Create oracle route labels from qrels.
3. Train a query router.
4. Trigger cross-encoder reranking only for uncertain queries.
5. Evaluate retrieval quality and cost.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Checks

```powershell
python -m compileall src scripts
python scripts/run_demo.py
```

## Planned Experiment Flow

```powershell
python scripts/prepare_scifact.py --config configs/scifact.yaml
python scripts/run_base_retrieval.py --config configs/scifact.yaml
python scripts/create_oracle_labels.py --config configs/scifact.yaml
python scripts/evaluate_phase2_routers.py --config configs/scifact.yaml
python scripts/train_router.py --config configs/scifact.yaml
python scripts/run_selective_rerank.py --config configs/scifact.yaml
```

The scripts are designed as staged deliverables. Start with `run_demo.py` to verify the package without downloading data.

## QLoRA Router Notebook

Prepare train/test router data:

```powershell
python scripts/prepare_scifact.py --config configs/scifact.yaml --split train
python scripts/run_base_retrieval.py --config configs/scifact.yaml --split train
python scripts/create_oracle_labels.py --config configs/scifact.yaml --split train
python scripts/export_llm_router_data.py --config configs/scifact.yaml --split train
python scripts/export_llm_router_data.py --config configs/scifact.yaml --split test
Compress-Archive -Path runs\scifact\train_llm_router_data.jsonl, runs\scifact\test_llm_router_data.jsonl -DestinationPath runs\scifact\llm_router_colab_inputs.zip -Force
```

Open `notebooks/qlora_router_training_colab.ipynb` in Colab, upload `runs/scifact/llm_router_colab_inputs.zip`, unzip it in the notebook file picker or upload the two JSONL files directly, then run all cells. The notebook predicts with label log-probability scoring over `bm25`, `dense`, and `hybrid`, and downloads `test_llm_router_predictions.csv` with per-label scores/probabilities.

Evaluate returned predictions locally:

```powershell
python scripts/evaluate_llm_router_predictions.py --config configs/scifact.yaml --split test --predictions test_llm_router_predictions.csv
```

For the log-probability output, tune class bias and confidence-gated fallback from label scores:

```powershell
python scripts/calibrate_llm_router_scores.py --config configs/scifact.yaml --calibration-split test --calibration-predictions "runs/scifact/test_llm_router_predictions (2).csv"
```

When `--calibration-predictions` and `--eval-predictions` point to the same file, treat the result as an in-split diagnostic. For a cleaner experiment, pass held-out calibration predictions and separate eval predictions.

Export lightweight retrieval diagnostics for router analysis:

```powershell
python scripts/analyze_retrieval_diagnostics.py --config configs/scifact.yaml --split test --predictions runs/scifact/test_test_llm_router_predictions_2_calibrated_predictions.csv
```

Create the Phase 2 quality-vs-cost comparison table:

```powershell
python scripts/compare_phase2_cost_quality.py --run-dir runs/scifact --split test
```
