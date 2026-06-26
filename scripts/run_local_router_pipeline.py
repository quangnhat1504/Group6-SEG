"""End-to-end LOCAL Small-LLM router pipeline (no Colab).

Runs the full clean train/dev/test protocol on local hardware (RTX 5070 Ti):

  1. make_local_router_splits.py   carve trainfit + dev out of the train split
  2. train_qlora_router_local.py   fine-tune on trainfit, score dev + test
  3. calibrate_llm_router_scores.py calibrate on dev, evaluate on test
  4. evaluate_llm_router_predictions.py  raw (uncalibrated) test metrics

This gives a genuine held-out separation: the calibration thresholds are tuned on
`dev` (disjoint from both training and test), so the final `test` numbers are not
tuned on themselves. Every stage is a thin wrapper around an existing script, so
you can also run the printed commands by hand.

Usage:
    python scripts/run_local_router_pipeline.py --config configs/scifact.yaml
    python scripts/run_local_router_pipeline.py --skip-splits   # splits already made
    python scripts/run_local_router_pipeline.py --dry-run       # only print commands
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from seg_retrieval.config import load_config

PY = sys.executable


def run(cmd: list[str], dry_run: bool) -> None:
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"\n$ {printable}")
    if dry_run:
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Step failed (exit {result.returncode}): {printable}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--trainfit-split", default="trainfit")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--pred-suffix", default="local")
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--skip-splits", action="store_true",
                        help="Reuse existing trainfit/dev splits.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse an existing adapter + prediction CSVs.")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Force bf16 LoRA instead of 4-bit QLoRA.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    suffix = args.pred_suffix
    dev_pred = run_dir / f"{args.dev_split}_llm_router_predictions_{suffix}.csv"
    test_pred = run_dir / f"{args.test_split}_llm_router_predictions_{suffix}.csv"

    # 1) splits ------------------------------------------------------------- #
    if not args.skip_splits:
        run([PY, "scripts/make_local_router_splits.py",
             "--config", args.config,
             "--trainfit-name", args.trainfit_split,
             "--dev-name", args.dev_split,
             "--dev-fraction", str(args.dev_fraction),
             "--seed", str(args.seed)], args.dry_run)

    # 2) train + predict dev/test ------------------------------------------ #
    train_cmd = [PY, "scripts/train_qlora_router_local.py",
                 "--config", args.config,
                 "--train-split", args.trainfit_split,
                 "--eval-splits", args.dev_split, args.test_split,
                 "--pred-suffix", suffix,
                 "--epochs", str(args.epochs),
                 "--seed", str(args.seed)]
    if args.no_4bit:
        train_cmd.append("--no-4bit")
    if args.skip_train:
        train_cmd.append("--skip-train")
    run(train_cmd, args.dry_run)

    # 3) calibrate on dev, evaluate on test -------------------------------- #
    run([PY, "scripts/calibrate_llm_router_scores.py",
         "--config", args.config,
         "--calibration-split", args.dev_split,
         "--calibration-predictions", str(dev_pred),
         "--eval-split", args.test_split,
         "--eval-predictions", str(test_pred),
         "--name", "Small LLM QLoRA Router Calibrated (local dev->test)"], args.dry_run)

    # 4) raw uncalibrated test metrics ------------------------------------- #
    run([PY, "scripts/evaluate_llm_router_predictions.py",
         "--config", args.config,
         "--split", args.test_split,
         "--predictions", str(test_pred),
         "--name", "Small LLM QLoRA Router LogProb (local)"], args.dry_run)

    print("\n" + "=" * 70)
    print("LOCAL ROUTER PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  dev predictions : {dev_pred}")
    print(f"  test predictions: {test_pred}")
    stem = Path(test_pred).stem.replace(" ", "_").replace("(", "").replace(")", "")
    print(f"  calibrated (dev->test) metrics: "
          f"{run_dir / (args.test_split + '_' + stem + '_calibrated_metrics.json')}")
    print(f"  raw test metrics: {run_dir / (args.test_split + '_' + stem + '_metrics.json')}")


if __name__ == "__main__":
    main()
