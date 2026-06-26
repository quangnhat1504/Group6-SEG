"""Local QLoRA fine-tuning of the Small LLM query router on an RTX 5070 Ti.

This is the local replacement for the Colab notebook
(``notebooks/qlora_router_training_colab.ipynb``). It fine-tunes
``Qwen/Qwen2.5-0.5B-Instruct`` to route a scientific query to exactly one of
``bm25`` / ``dense`` / ``hybrid``, then scores the three label log-probabilities
for every query in one or more eval splits and writes a predictions CSV with the
EXACT schema the rest of the pipeline expects:

    query_id, query, true_label, pred_label,
    bm25_score, dense_score, hybrid_score,
    bm25_prob, dense_prob, hybrid_prob

Recommended clean train/dev/test protocol (no Colab, fully local):

    # 1) carve disjoint trainfit + dev out of the 809 train queries
    python scripts/make_local_router_splits.py --config configs/scifact.yaml

    # 2) fine-tune on trainfit only, predict on dev and test
    python scripts/train_qlora_router_local.py --config configs/scifact.yaml \
        --train-split trainfit --eval-splits dev test

    # 3) calibrate on dev, evaluate on test (genuine held-out)
    python scripts/calibrate_llm_router_scores.py --config configs/scifact.yaml \
        --calibration-split dev  --calibration-predictions runs/scifact/dev_llm_router_predictions_local.csv \
        --eval-split        test --eval-predictions        runs/scifact/test_llm_router_predictions_local.csv

Hardware notes (RTX 5070 Ti, Blackwell / sm_120):
  * Default tries 4-bit QLoRA (bitsandbytes, nf4). If bitsandbytes is missing or
    fails on this GPU, it AUTOMATICALLY falls back to bf16 LoRA — a 0.5B model
    fits comfortably in 16 GB without quantisation, so results are unaffected.
  * Force a mode with --load-in-4bit / --no-4bit.

Dependencies (install into the project venv):
    pip install "transformers>=4.44" "peft>=0.11" "accelerate>=0.30" \
                "datasets>=2.19" "bitsandbytes>=0.43"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_queries

ROUTE_LABELS = ["bm25", "dense", "hybrid"]
SYSTEM_PROMPT = (
    "You are a query router for a scientific paper search engine. "
    "Choose exactly one retrieval route from: bm25, dense, hybrid."
)


# --------------------------------------------------------------------------- #
# Prompt formatting (identical for training and scoring so logits are aligned).
# --------------------------------------------------------------------------- #
def build_messages(query: str, label: str | None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Academic query:\n{query}\n\nRoute:"},
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": label})
    return messages


def load_split_examples(config, split: str) -> list[dict]:
    """Return [{query_id, query, label}] for a split using oracle labels."""
    queries = load_queries(config.dataset.data_dir / f"{split}_queries.jsonl")
    query_by_id = {q.query_id: q.text for q in queries}
    labels = pd.read_csv(config.outputs.run_dir / f"{split}_oracle_labels.csv")
    rows = []
    for row in labels.itertuples():
        qid = str(row.query_id)
        if qid in query_by_id:
            rows.append(
                {
                    "query_id": qid,
                    "query": query_by_id[qid],
                    "label": str(row.oracle_label).strip().lower(),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Model loading with 4-bit -> bf16 fallback.
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(model_id: str, want_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_mode = "none"
    model = None
    if want_4bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            quant_mode = "4bit-nf4"
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 4-bit load failed ({type(exc).__name__}: {exc}); "
                  f"falling back to {compute_dtype} LoRA.")
            model = None

    if model is None:
        device_map = "auto" if torch.cuda.is_available() else None
        # transformers >=5 renamed `torch_dtype` -> `dtype`; support both.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=compute_dtype, device_map=device_map, trust_remote_code=True,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=compute_dtype, device_map=device_map, trust_remote_code=True,
            )
        quant_mode = f"{compute_dtype}".replace("torch.", "")

    return model, tokenizer, quant_mode, use_bf16


def attach_lora(model, want_4bit: bool, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if want_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# --------------------------------------------------------------------------- #
# Training.
# --------------------------------------------------------------------------- #
def train(model, tokenizer, train_rows, args, use_bf16):
    from datasets import Dataset
    from transformers import (DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    def tokenize_example(row):
        text = tokenizer.apply_chat_template(
            build_messages(row["query"], row["label"]),
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = tokenizer(
            text, truncation=True, max_length=args.max_length, padding="max_length"
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    train_ds = Dataset.from_list(train_rows).map(
        tokenize_example, remove_columns=list(train_rows[0].keys())
    )

    training_args = TrainingArguments(
        output_dir=str(Path(args.adapter_dir) / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    return model


# --------------------------------------------------------------------------- #
# Label log-probability scoring (matches the Colab score_label exactly).
# --------------------------------------------------------------------------- #
def make_scorer(model, tokenizer, max_length):
    import torch
    import torch.nn.functional as F

    label_token_ids = {
        label: tokenizer(label, add_special_tokens=False).input_ids
        for label in ROUTE_LABELS
    }

    @torch.no_grad()
    def score_label(prompt: str, label: str) -> float:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        ).to(model.device)
        label_ids = torch.tensor([label_token_ids[label]], device=model.device)
        input_ids = torch.cat([inputs["input_ids"], label_ids], dim=1)
        attention_mask = torch.cat(
            [inputs["attention_mask"], torch.ones_like(label_ids)], dim=1
        )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]
        target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        prompt_len = inputs["input_ids"].shape[1]
        label_log_probs = target_log_probs[
            :, prompt_len - 1: prompt_len - 1 + label_ids.shape[1]
        ]
        return float(label_log_probs.mean().item())

    @torch.no_grad()
    def predict(query: str):
        prompt = tokenizer.apply_chat_template(
            build_messages(query, None), tokenize=False, add_generation_prompt=True
        )
        scores = {label: score_label(prompt, label) for label in ROUTE_LABELS}
        score_tensor = torch.tensor([scores[label] for label in ROUTE_LABELS])
        prob_tensor = torch.softmax(score_tensor, dim=0)
        probs = {label: float(prob_tensor[i]) for i, label in enumerate(ROUTE_LABELS)}
        pred_label = max(scores, key=scores.get)
        return pred_label, scores, probs

    return predict


def predict_split(predict_fn, rows) -> pd.DataFrame:
    from tqdm.auto import tqdm

    out = []
    for row in tqdm(rows, desc="scoring"):
        pred_label, scores, probs = predict_fn(row["query"])
        out.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "true_label": row.get("label"),
                "pred_label": pred_label,
                "bm25_score": scores["bm25"],
                "dense_score": scores["dense"],
                "hybrid_score": scores["hybrid"],
                "bm25_prob": probs["bm25"],
                "dense_prob": probs["dense"],
                "hybrid_prob": probs["hybrid"],
            }
        )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-split", default="trainfit",
                        help="Split used to FINE-TUNE the router (e.g. trainfit).")
    parser.add_argument("--eval-splits", nargs="+", default=["dev", "test"],
                        help="Splits to score after training.")
    parser.add_argument("--adapter-dir", default="runs/scifact/qlora_router_local")
    parser.add_argument("--pred-suffix", default="local",
                        help="Prediction file = <split>_llm_router_predictions_<suffix>.csv")
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument("--load-in-4bit", dest="want_4bit", action="store_true", default=True)
    quant.add_argument("--no-4bit", dest="want_4bit", action="store_false")
    parser.add_argument("--skip-train", action="store_true",
                        help="Load an existing adapter and only run prediction.")
    args = parser.parse_args()

    import torch
    from sklearn.metrics import accuracy_score, f1_score

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[info] GPU: {torch.cuda.get_device_name(0)} "
              f"(capability {torch.cuda.get_device_capability(0)})")

    model, tokenizer, quant_mode, use_bf16 = load_model_and_tokenizer(
        args.model_id, args.want_4bit
    )
    effective_4bit = quant_mode.startswith("4bit")
    print(f"[info] quantisation mode: {quant_mode} | bf16: {use_bf16}")

    if args.skip_train:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir / "adapter"))
    else:
        model = attach_lora(model, effective_4bit, args.lora_r, args.lora_alpha, args.lora_dropout)
        train_rows = load_split_examples(config, args.train_split)
        print(f"[info] training on '{args.train_split}': {len(train_rows)} examples")
        print(pd.Series([r["label"] for r in train_rows]).value_counts().to_string())
        model = train(model, tokenizer, train_rows, args, use_bf16)
        model.save_pretrained(str(adapter_dir / "adapter"))
        print(f"[info] saved adapter to {adapter_dir / 'adapter'}")

    model.eval()
    predict_fn = make_scorer(model, tokenizer, args.max_length)

    summary = {
        "model_id": args.model_id,
        "train_split": args.train_split,
        "quantisation": quant_mode,
        "epochs": args.epochs,
        "splits": {},
    }
    for split in args.eval_splits:
        rows = load_split_examples(config, split)
        df = predict_split(predict_fn, rows)
        out_path = run_dir / f"{split}_llm_router_predictions_{args.pred_suffix}.csv"
        df.to_csv(out_path, index=False)
        acc = accuracy_score(df["true_label"], df["pred_label"])
        mf1 = f1_score(df["true_label"], df["pred_label"], average="macro")
        summary["splits"][split] = {
            "n": int(len(df)),
            "accuracy": float(acc),
            "macro_f1": float(mf1),
            "predictions": str(out_path),
        }
        print(f"[done] {split}: n={len(df)} acc={acc:.4f} macro_f1={mf1:.4f} -> {out_path}")

    (adapter_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
