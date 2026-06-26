"""Run ALL validation experiments in one shot.

This script executes the complete validation pipeline in the correct order:
  1. Generate train-split runs (base retrieval + reranking on 809 train queries)
  2. QPP validation on train split
  3. RRF k=5 pipeline (re-fusion + reranking on test split)
  4. Rerank depth ablation (top-50, already done — skips if output exists)
  5. Cross-dataset validation on NFCorpus (BM25 + Dense + RRF + reranking)
  6. Significance test (conformal selective vs always-rerank)
  7. Positioning note generation
  8. Final report generation (LaTeX tables, figures, .tex section)

Usage:
  python scripts/run_all_validation.py

GPU notes:
  - Dense retrieval (SciNCL) and cross-encoder reranking use GPU automatically
    if CUDA is available via PyTorch/Transformers.
  - Set CUDA_VISIBLE_DEVICES=0 to pin to a specific GPU if needed.

Estimated total time (with GPU): ~15-30 minutes depending on hardware.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Fix Keras 3 / TensorFlow compatibility issue with transformers
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable  # Use the same Python that's running this script


def run_script(script_name: str, args: list[str] | None = None, required: bool = True) -> bool:
    """Run a script and return True on success."""
    script_path = ROOT / "scripts" / script_name
    cmd = [PYTHON, str(script_path)] + (args or [])

    # Pass env vars to child processes to avoid Keras/TF issues
    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["TRANSFORMERS_NO_TF"] = "1"

    print(f"\n{'='*70}")
    print(f"  RUNNING: {script_name} {' '.join(args or [])}")
    print(f"{'='*70}\n")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        print(f"\n  ✓ {script_name} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ✗ {script_name} FAILED (exit code {result.returncode}, {elapsed:.1f}s)")
        if required:
            print("  Stopping pipeline due to required step failure.")
            sys.exit(1)
        return False


def check_file_exists(path: Path, description: str) -> bool:
    """Check if a prerequisite file exists."""
    if path.exists():
        return True
    print(f"  [SKIP] {description} — missing: {path}")
    return False


def _rerank_train_split_direct() -> None:
    """Directly rerank the train split hybrid run using CrossEncoderReranker.

    This bypasses run_selective_rerank.py which may have Keras/TF import issues.
    """
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import _bootstrap  # noqa: F401

    from seg_retrieval.config import load_config
    from seg_retrieval.io import load_documents, load_queries, load_run, save_run
    from seg_retrieval.rerank import CrossEncoderReranker

    config = load_config(str(ROOT / "configs" / "scifact.yaml"))
    run_dir = config.outputs.run_dir
    data_dir = config.dataset.data_dir

    print("  Loading train hybrid run, queries, documents...")
    hybrid_run = load_run(run_dir / "train_hybrid.csv")
    queries = load_queries(data_dir / "train_queries.jsonl")
    documents = load_documents(data_dir / "train_documents.jsonl")
    doc_map = {doc.doc_id: doc for doc in documents}

    print(f"  Initializing CrossEncoderReranker ({config.rerank.model})...")
    reranker = CrossEncoderReranker(config.rerank.model)

    print(f"  Reranking top-20 for {len(queries)} train queries...")
    reranked_run = {}
    for i, q in enumerate(queries, 1):
        hits = hybrid_run.get(q.query_id, [])
        reranked_run[q.query_id] = reranker.rerank(q, doc_map, hits, top_k=20)
        if i % 100 == 0 or i == len(queries):
            print(f"    Reranked {i}/{len(queries)} queries...")

    out_path = run_dir / "train_always_rerank.csv"
    save_run(out_path, reranked_run)
    print(f"  ✓ Saved: {out_path}")


def main() -> None:
    print("=" * 70)
    print("  SEG VALIDATION — COMPLETE PIPELINE")
    print("  Running all experiments (GPU-accelerated where available)")
    print("=" * 70)

    t_start = time.perf_counter()

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n  GPU detected: {gpu_name}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("\n  ⚠ No GPU detected — will run on CPU (slower)")
    except ImportError:
        print("\n  ⚠ PyTorch not found — GPU detection skipped")

    # -----------------------------------------------------------------
    # Step 1: Generate train-split base runs (needed for QPP validation)
    # -----------------------------------------------------------------
    train_hybrid = ROOT / "runs" / "scifact" / "train_hybrid.csv"
    train_rerank = ROOT / "runs" / "scifact" / "train_always_rerank.csv"

    if not train_hybrid.exists():
        print("\n[Step 1a] Generating train-split base retrieval runs...")
        run_script("run_base_retrieval.py", ["--split", "train"])
    else:
        print("\n[Step 1a] Train base runs already exist — skipping.")

    if not train_rerank.exists():
        print("\n[Step 1b] Generating train-split always-rerank run...")
        # Try run_selective_rerank first; if it fails due to Keras/TF issues,
        # fall back to a direct reranking approach
        rerank_script = ROOT / "scripts" / "run_selective_rerank.py"
        if rerank_script.exists():
            success = run_script("run_selective_rerank.py", ["--split", "train", "--rerank-all"], required=False)
            if not success:
                print("  Falling back to direct train reranking...")
                _rerank_train_split_direct()
        else:
            _rerank_train_split_direct()
    else:
        print("\n[Step 1b] Train always-rerank run already exists — skipping.")

    # -----------------------------------------------------------------
    # Step 2: QPP validation on train split
    # -----------------------------------------------------------------
    print("\n[Step 2] QPP feature selection validation (train split)...")
    train_qpp = ROOT / "runs" / "scifact" / "train_qpp_features.csv"
    if not train_qpp.exists():
        run_script("run_qpp_validation.py", ["--split", "train"])
    else:
        print("  Already exists — skipping.")

    # -----------------------------------------------------------------
    # Step 3: RRF k=5 full pipeline
    # -----------------------------------------------------------------
    print("\n[Step 3] RRF k=5 full pipeline (re-fusion + reranking)...")
    k5_rerank = ROOT / "runs" / "scifact" / "test_always_rerank_k5.csv"
    if not k5_rerank.exists():
        run_script("run_rrf_k5_pipeline.py")
    else:
        print("  Already exists — skipping.")

    # -----------------------------------------------------------------
    # Step 4: Rerank depth ablation (top-50)
    # -----------------------------------------------------------------
    print("\n[Step 4] Rerank depth ablation (top-50)...")
    depth50 = ROOT / "runs" / "scifact" / "test_always_rerank_depth50.csv"
    if not depth50.exists():
        run_script("run_rerank_depth_ablation.py")
    else:
        print("  Already exists — skipping.")

    # -----------------------------------------------------------------
    # Step 5: Cross-dataset validation (NFCorpus) — most compute-heavy
    # -----------------------------------------------------------------
    print("\n[Step 5] Cross-dataset validation (NFCorpus)...")
    nfcorpus_hybrid = ROOT / "runs" / "nfcorpus" / "test_hybrid.csv"
    if not nfcorpus_hybrid.exists():
        run_script("run_cross_dataset.py")
    else:
        print("  Already exists — skipping.")

    # -----------------------------------------------------------------
    # Step 6: Significance test (conformal selective vs always-rerank)
    # -----------------------------------------------------------------
    print("\n[Step 6] Significance test (conformal selective)...")
    sig_csv = ROOT / "runs" / "scifact" / "test_significance_conformal.csv"
    if not sig_csv.exists():
        run_script("run_significance_conformal.py")
    else:
        print("  Already exists — skipping.")

    # -----------------------------------------------------------------
    # Step 7: Positioning note
    # -----------------------------------------------------------------
    print("\n[Step 7] Generating positioning note...")
    run_script("generate_positioning_note.py", required=False)

    # -----------------------------------------------------------------
    # Step 8: Final report (LaTeX + figures)
    # -----------------------------------------------------------------
    print("\n[Step 8] Generating final report (LaTeX tables, figures, .tex)...")
    run_script("generate_final_report.py", required=False)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    total_time = time.perf_counter() - t_start
    print(f"\n{'='*70}")
    print(f"  ALL VALIDATION EXPERIMENTS COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")

    # Check outputs
    outputs = [
        ("runs/scifact/train_qpp_features.csv", "QPP train features"),
        ("runs/scifact/test_hybrid_k5.csv", "k=5 hybrid run"),
        ("runs/scifact/test_always_rerank_k5.csv", "k=5 reranked run"),
        ("runs/scifact/test_conformal_results_k5.csv", "k=5 CRC results"),
        ("runs/scifact/test_always_rerank_depth50.csv", "Depth-50 reranked"),
        ("runs/scifact/test_depth_ablation_metrics.json", "Depth ablation metrics"),
        ("runs/scifact/test_significance_conformal.csv", "Significance test"),
        ("runs/nfcorpus/test_hybrid.csv", "NFCorpus hybrid"),
        ("runs/nfcorpus/test_always_rerank.csv", "NFCorpus reranked"),
        ("runs/nfcorpus/test_conformal_results.csv", "NFCorpus CRC"),
        ("reports/tables/table_significance_conformal.md", "Significance table"),
        ("reports/tables/table_k5_comparison.md", "k=5 comparison table"),
        ("reports/tables/table_rerank_depth.md", "Depth ablation table"),
        ("reports/tables/table_cross_dataset.md", "Cross-dataset table"),
        ("reports/tables/qpp_validation_comparison.md", "QPP validation table"),
        ("reports/tables/positioning_note_al_joofi.md", "Positioning note"),
        ("paper/sections/validation_experiments.tex", "LaTeX section"),
        ("paper/figures/validation_depth_latency.png", "Depth/latency figure"),
    ]

    print("\nOutput files:")
    for rel_path, desc in outputs:
        full = ROOT / rel_path
        status = "✓" if full.exists() else "✗ MISSING"
        print(f"  {status}  {desc}: {rel_path}")


if __name__ == "__main__":
    main()
