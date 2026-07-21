"""Train your own BGE-small fine-tuned on SciFact using RRF disagreement signal.

Usage: python scripts/train_bge_small_scifact.py

This creates YOUR OWN fine-tuned model (trained on your data, your pipeline).
Runtime: ~10-15 min on RTX 5070 Ti.

Output: runs/finetuned/bge-small-scifact-rrf/
"""

import _bootstrap
import argparse
import json, random, math, gc, time
from pathlib import Path
from collections import defaultdict
import torch, numpy as np

SEED = 42

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BGE-small on SciFact RRF-disagreement triples.")
    parser.add_argument("--output-dir", default="runs/finetuned/bge-small-scifact-rrf")
    parser.add_argument("--exclude-train-query-ids", default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--eval-split", choices=["dev", "test"], default="test")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--triple-source", choices=["both", "labeled-only", "pseudo-only"], default="both")
    parser.add_argument("--max-pseudo-queries", type=int, default=5000)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--disagreement-top-k", type=int, default=10)
    parser.add_argument("--hard-negative-strategy", choices=["disagreement", "bm25-top", "dense-top", "bm25+dense-top"], default="disagreement")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum-stage1-epochs", type=int, default=3)
    parser.add_argument("--curriculum-stage2-epochs", type=int, default=3)
    return parser.parse_args()

args = parse_args()
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXCLUDED_TRAIN_QUERY_IDS = {x.strip() for x in args.exclude_train_query_ids.split(",") if x.strip()}

print("=" * 65)
print("  TRAIN YOUR OWN MODEL: BGE-small on SciFact via RRF disagreement")
print("  Model will be saved to:", OUTPUT_DIR)
print("=" * 65)

# ─────────────────────────────────────────────
# Phase 1: Mine RRF disagreement triples
# ─────────────────────────────────────────────
from seg_retrieval.io import load_documents, load_queries, load_qrels
from seg_retrieval.retrievers import BM25Retriever, tokenize
from seg_retrieval.fusion import reciprocal_rank_fusion
from sentence_transformers import SentenceTransformer

print("\n[Phase 1] Mining RRF disagreement triples...")

# Load data
train_docs = load_documents(Path("data/scifact/train_documents.jsonl"))
train_qs = load_queries(Path("data/scifact/train_queries.jsonl"))
train_qrels = load_qrels(Path("data/scifact/train_qrels.csv"))
if EXCLUDED_TRAIN_QUERY_IDS:
    before_qs = len(train_qs)
    train_qs = [q for q in train_qs if q.query_id not in EXCLUDED_TRAIN_QUERY_IDS]
    train_qrels = {qid: rels for qid, rels in train_qrels.items() if qid not in EXCLUDED_TRAIN_QUERY_IDS}
    print(f"  Excluded train query ids: {sorted(EXCLUDED_TRAIN_QUERY_IDS)} ({before_qs - len(train_qs)} queries)")
print(f"  Train: {len(train_docs)} docs, {len(train_qs)} queries")

doc_texts = [d.text[:512] for d in train_docs]
doc_ids = [d.doc_id for d in train_docs]
doc_lookup = {d.doc_id: d for d in train_docs}

# BM25 index
bm25 = BM25Retriever(train_docs)

# Sample pseudo-queries from docs
max_pseudo = min(args.max_pseudo_queries, len(train_docs))
sampled = random.sample(train_docs, max_pseudo)
pq_texts = [d.text[:200] for d in sampled]
pq_dids = [d.doc_id for d in sampled]

# BM25 search for pseudo-queries
bm25_run_pq = {}
for qi, txt in enumerate(pq_texts):
    scores = bm25.index.get_scores(tokenize(txt))
    order = np.argsort(scores)[::-1][:100]
    bm25_run_pq[f"pq_{qi}"] = [(doc_ids[i], float(scores[i])) for i in order]

# Dense encode with BGE-small
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
q_emb = model.encode(pq_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
d_emb = model.encode(doc_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
q_t = torch.tensor(q_emb, device=DEVICE)
d_t = torch.tensor(d_emb, device=DEVICE)

# Dense search
dense_run_pq = {}
for qi in range(len(pq_texts)):
    s = torch.matmul(d_t.float(), q_t[qi].float())
    top = torch.topk(s, 100).indices.cpu().numpy()
    dense_run_pq[f"pq_{qi}"] = [(doc_ids[i], float(s[i])) for i in top]

del q_emb, d_emb
gc.collect()

# RRF fusion
rrf_run_pq = reciprocal_rank_fusion([bm25_run_pq, dense_run_pq], k=60, top_k=100)

# Helper: collect hard negatives from a run based on strategy
dk = args.disagreement_top_k
npp = args.negatives_per_positive
hns = args.hard_negative_strategy

def collect_negatives(btop_set, dtop_set, gold_set, bm25_run_for_q, dense_run_for_q):
    if hns == "disagreement":
        sd = (btop_set - dtop_set) | (dtop_set - btop_set)
        return [d for d in sd if d not in gold_set]
    elif hns == "bm25-top":
        return [d for d, _ in bm25_run_for_q if d not in gold_set]
    elif hns == "dense-top":
        return [d for d, _ in dense_run_for_q if d not in gold_set]
    elif hns == "bm25+dense-top":
        all_top = {d for d, _ in bm25_run_for_q[:dk]} | {d for d, _ in dense_run_for_q[:dk]}
        return [d for d in all_top if d not in gold_set]
    return []

all_triples = []

# ── Generate pseudo triples ──
if args.triple_source in ("both", "pseudo-only"):
    npq = 0
    for qi in range(len(pq_texts)):
        qid = f"pq_{qi}"
        bmq = bm25_run_pq.get(qid, [])[:dk]
        dsq = dense_run_pq.get(qid, [])[:dk]
        btop = {d for d, _ in bmq}
        dtop = {d for d, _ in dsq}
        negs = collect_negatives(btop, dtop, set(), bmq, dsq)
        if len(negs) < npp:
            continue
        rr3 = [d for d, _ in rrf_run_pq.get(qid, [])[:3]]
        for pos in rr3:
            for _ in range(npp):
                all_triples.append(("pq", pq_texts[qi], pos, random.choice(negs)))
                npq += 1
    print(f"  Pseudo triples: {npq}")

# ── Generate labeled triples (SciFact train qrels) ──
if args.triple_source in ("both", "labeled-only"):
    bm25_train = bm25.search(train_qs, top_k=100)
    q_emb2 = model.encode([q.text for q in train_qs], batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    q_t2 = torch.tensor(q_emb2, device=DEVICE)

    dense_train = {}
    for qi in range(len(train_qs)):
        s = torch.matmul(d_t.float(), q_t2[qi].float())
        top = torch.topk(s, 100).indices.cpu().numpy()
        dense_train[train_qs[qi].query_id] = [(doc_ids[i], float(s[i])) for i in top]

    nl = 0
    for q in train_qs:
        if q.query_id not in train_qrels:
            continue
        gold = [did for did, rel in train_qrels[q.query_id].items() if rel > 0]
        if not gold:
            continue
        gold_set = set(gold)
        bmq = bm25_train.get(q.query_id, [])[:dk]
        dsq = dense_train.get(q.query_id, [])[:dk]
        btop = {d for d, _ in bmq}
        dtop = {d for d, _ in dsq}
        negs = collect_negatives(btop, dtop, gold_set, bmq, dsq)
        if len(negs) < npp:
            continue
        for gd in gold[:3]:
            for _ in range(npp):
                all_triples.append(("labeled", q.text, gd, random.choice(negs)))
                nl += 1
    print(f"  Labeled triples: {nl}")

# ── Dedup exact triples ──
before_dedup = len(all_triples)
seen = set()
deduped = []
for src, qtxt, pid, nid in all_triples:
    key = (qtxt, pid, nid)
    if key not in seen:
        seen.add(key)
        deduped.append((src, qtxt, pid, nid))
all_triples = deduped
if before_dedup > len(all_triples):
    print(f"  Dedup removed {before_dedup - len(all_triples)} duplicate triples")

print(f"  TOTAL triples: {len(all_triples)}")

for v in ["q_t", "q_t2", "d_t", "dense_run_pq", "bm25_run_pq", "rrf_run_pq"]:
    try:
        del globals()[v]
    except KeyError:
        pass
gc.collect()
torch.cuda.empty_cache()

# ─────────────────────────────────────────────
# Phase 2: Convert to InputExamples
# ─────────────────────────────────────────────
print("\n[Phase 2] Converting to training examples...")
random.shuffle(all_triples)

from sentence_transformers import InputExample

def make_examples(triples):
    exs = []
    for src, qtxt, pid, nid in triples:
        pd = doc_lookup.get(pid)
        nd = doc_lookup.get(nid)
        if pd and nd:
            exs.append(InputExample(texts=[qtxt, pd.text[:512], nd.text[:512]]))
    return exs

if not args.curriculum:
    examples = make_examples(all_triples)
    print(f"  Valid examples: {len(examples)}")
else:
    pq_triples = [(s, q, p, n) for s, q, p, n in all_triples if s == "pq"]
    labeled_triples = [(s, q, p, n) for s, q, p, n in all_triples if s == "labeled"]
    random.shuffle(pq_triples)
    random.shuffle(labeled_triples)
    stage1_examples = make_examples(pq_triples)
    stage2_examples = make_examples(labeled_triples)
    print(f"  Stage 1 (pseudo) examples: {len(stage1_examples)}")
    print(f"  Stage 2 (labeled) examples: {len(stage2_examples)}")

del all_triples, doc_lookup
gc.collect()

# ─────────────────────────────────────────────
# Phase 3: Fine-tune with MNRL
# ─────────────────────────────────────────────
print("\n[Phase 3] Fine-tuning with MNRL...")

from sentence_transformers import losses

loss_fn = losses.MultipleNegativesRankingLoss(model)

if args.curriculum:
    print("  [Stage 1] Pseudo triples (broad)")
    train_dl1 = torch.utils.data.DataLoader(stage1_examples, shuffle=True, batch_size=args.batch_size)
    model.fit(
        train_objectives=[(train_dl1, loss_fn)],
        epochs=args.curriculum_stage1_epochs,
        warmup_steps=int(len(train_dl1) * args.warmup_ratio),
        optimizer_params={"lr": args.learning_rate},
        output_path=None,
        show_progress_bar=True,
    )
    del stage1_examples, train_dl1
    gc.collect()

    print("  [Stage 2] Labeled hard triples (clean)")
    train_dl2 = torch.utils.data.DataLoader(stage2_examples, shuffle=True, batch_size=args.batch_size)
    model.fit(
        train_objectives=[(train_dl2, loss_fn)],
        epochs=args.curriculum_stage2_epochs,
        warmup_steps=int(len(train_dl2) * args.warmup_ratio),
        optimizer_params={"lr": args.learning_rate},
        output_path=str(OUTPUT_DIR),
        save_best_model=True,
        show_progress_bar=True,
    )
    # label examples for eval footer
    examples = stage2_examples
    del stage2_examples, train_dl2
else:
    train_dl = torch.utils.data.DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    model.fit(
        train_objectives=[(train_dl, loss_fn)],
        epochs=args.epochs,
        warmup_steps=int(len(train_dl) * args.warmup_ratio),
        optimizer_params={"lr": args.learning_rate},
        output_path=str(OUTPUT_DIR),
        save_best_model=True,
        show_progress_bar=True,
    )
    del train_dl
torch.cuda.empty_cache()

# ─────────────────────────────────────────────
# Phase 4: Evaluate
# ─────────────────────────────────────────────
print(f"\n[Phase 4] Evaluating on SciFact {args.eval_split}...")

from seg_retrieval.retrievers import DenseRetriever
from seg_retrieval.metrics import evaluate_run
from seg_retrieval.io import save_run

# Load eval data
eval_docs_path = Path(f"data/scifact/{args.eval_split}_documents.jsonl")
if not eval_docs_path.exists():
    eval_docs_path = Path("data/scifact/test_documents.jsonl")
test_docs = load_documents(eval_docs_path)
test_qs = load_queries(Path(f"data/scifact/{args.eval_split}_queries.jsonl"))
test_qrels = load_qrels(Path(f"data/scifact/{args.eval_split}_qrels.csv"))

# Baseline: BGE-small
print("  Evaluating baseline (BGE-small)...")
base_dense = DenseRetriever(test_docs, MODEL_NAME, batch_size=32)
base_run = base_dense.search(test_qs, top_k=100)
base_m = evaluate_run(base_run, test_qrels)

# Fine-tuned: Our model
print("  Evaluating fine-tuned model...")
ft_model = SentenceTransformer(str(OUTPUT_DIR), device=DEVICE)
ft_dense = DenseRetriever(test_docs, str(OUTPUT_DIR), batch_size=32)
ft_run = ft_dense.search(test_qs, top_k=100)
ft_m = evaluate_run(ft_run, test_qrels)

delta = ft_m["ndcg@10"] - base_m["ndcg@10"]

print(f"""
{'=' * 65}
  RESULTS on SciFact {args.eval_split} ({len(test_qs)} queries)
{'=' * 65}

  Baseline (BGE-small, 33M):
    nDCG@10    = {base_m['ndcg@10']:.4f}
    MAP@10     = {base_m.get('map@10', 0):.4f}
    Recall@10  = {base_m['recall@10']:.4f}
    Recall@100 = {base_m['recall@100']:.4f}
    MRR@10     = {base_m['mrr@10']:.4f}

  Ours (BGE-small fine-tuned, 33M):
    nDCG@10    = {ft_m['ndcg@10']:.4f}
    MAP@10     = {ft_m.get('map@10', 0):.4f}
    Recall@10  = {ft_m['recall@10']:.4f}
    Recall@100 = {ft_m['recall@100']:.4f}
    MRR@10     = {ft_m['mrr@10']:.4f}

  Delta nDCG@10: {delta:+.4f} ({delta/base_m['ndcg@10']*100:+.1f}%)
""")

# Also compare with BGE-base (109M)
bge_base_ndcg = 0.7376
print(f"  BGE-base (109M, zero-shot): 0.7376")
print(f"  Delta vs BGE-base: {ft_m['ndcg@10'] - bge_base_ndcg:+.4f}")
print(f"  vstash rrf-v3 (33M, ref):   0.7707")
print(f"  Delta vs vstash:  {ft_m['ndcg@10'] - 0.7707:+.4f}")

# Save run
save_run(Path(f"runs/scifact/{args.eval_split}_dense_bge_small_scifact_rrf.csv"), ft_run)

# Save results
results = {
    "model_name": "bge-small-scifact-rrf",
    "base_model": MODEL_NAME,
    "eval_split": args.eval_split,
    "num_examples": len(examples),
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "warmup_ratio": args.warmup_ratio,
    "learning_rate": args.learning_rate,
    "excluded_train_query_ids": sorted(EXCLUDED_TRAIN_QUERY_IDS),
    "triple_source": args.triple_source,
    "max_pseudo_queries": args.max_pseudo_queries,
    "negatives_per_positive": args.negatives_per_positive,
    "disagreement_top_k": args.disagreement_top_k,
    "hard_negative_strategy": args.hard_negative_strategy,
    "curriculum": args.curriculum,
    "curriculum_stage1_epochs": args.curriculum_stage1_epochs,
    "curriculum_stage2_epochs": args.curriculum_stage2_epochs,
    "baseline_ndcg": base_m["ndcg@10"],
    "ft_ndcg": ft_m["ndcg@10"],
    "delta": delta,
    "ft_map": ft_m.get("map@10", 0),
    "ft_recall_10": ft_m["recall@10"],
    "ft_recall_100": ft_m["recall@100"],
    "ft_mrr": ft_m["mrr@10"],
}
json.dump(results, open(OUTPUT_DIR / "results.json", "w"), indent=2)
manifest = {
    "seed": SEED,
    "base_model": MODEL_NAME,
    "eval_split": args.eval_split,
    "output_dir": str(OUTPUT_DIR),
    "eval_split": args.eval_split,
    "excluded_train_query_ids": sorted(EXCLUDED_TRAIN_QUERY_IDS),
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "warmup_ratio": args.warmup_ratio,
    "learning_rate": args.learning_rate,
    "triple_source": args.triple_source,
    "max_pseudo_queries": args.max_pseudo_queries,
    "negatives_per_positive": args.negatives_per_positive,
    "disagreement_top_k": args.disagreement_top_k,
    "hard_negative_strategy": args.hard_negative_strategy,
    "curriculum": args.curriculum,
    "curriculum_stage1_epochs": args.curriculum_stage1_epochs,
    "curriculum_stage2_epochs": args.curriculum_stage2_epochs,
    "train_files": [
        "data/scifact/train_documents.jsonl",
        "data/scifact/train_queries.jsonl",
        "data/scifact/train_qrels.csv",
    ],
    "test_files": [
        "data/scifact/test_documents.jsonl",
        "data/scifact/test_queries.jsonl",
        "data/scifact/test_qrels.csv",
    ],
}
json.dump(manifest, open(OUTPUT_DIR / "manifest.json", "w"), indent=2)

print(f"\n  Model saved to: {OUTPUT_DIR}")
print(f"  Results saved to: {OUTPUT_DIR / 'results.json'}")
print(f"\n  This is YOUR model — trained on YOUR data with YOUR pipeline.")
print(f"  Upload to HuggingFace: MODEL_ID = '{OUTPUT_DIR}'")
print("=" * 65)
