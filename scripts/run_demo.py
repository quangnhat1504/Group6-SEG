from __future__ import annotations

import _bootstrap  # noqa: F401

from seg_retrieval.fusion import reciprocal_rank_fusion
from seg_retrieval.metrics import evaluate_run
from seg_retrieval.oracle import create_oracle_labels
from seg_retrieval.retrievers import BM25Retriever
from seg_retrieval.types import Document, Query


def main() -> None:
    documents = [
        Document("d1", "BM25 for exact scientific terms", "Lexical retrieval is strong for rare terms."),
        Document("d2", "Dense retrieval for semantic search", "Embedding models retrieve related meaning."),
        Document("d3", "Hybrid retrieval with RRF", "Reciprocal Rank Fusion combines sparse and dense runs."),
    ]
    queries = [
        Query("q1", "exact rare term BM25"),
        Query("q2", "combine lexical and semantic retrieval"),
    ]
    qrels = {
        "q1": {"d1": 1},
        "q2": {"d3": 1, "d2": 1},
    }

    bm25 = BM25Retriever(documents).search(queries, top_k=3)
    dense_stub = {
        "q1": [("d1", 0.8), ("d3", 0.3), ("d2", 0.1)],
        "q2": [("d2", 0.9), ("d3", 0.85), ("d1", 0.2)],
    }
    hybrid = reciprocal_rank_fusion([bm25, dense_stub], k=60, top_k=3)

    print("BM25:", evaluate_run(bm25, qrels))
    print("Hybrid:", evaluate_run(hybrid, qrels))
    print(create_oracle_labels({"bm25": bm25, "dense": dense_stub, "hybrid": hybrid}, qrels))


if __name__ == "__main__":
    main()
