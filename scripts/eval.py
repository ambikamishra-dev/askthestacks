"""Evaluate retrieval quality against a hand-rated query set.

Reads eval/queries.json — a set of queries where each top-5 result was
manually judged as relevant / partial / not_relevant by the operator.

Reports three metrics:
- Precision@5: weighted relevance across the top 5 results
- Hit@1: was the top result rated 'relevant' or 'partial'
- NDCG@5: normalized discounted cumulative gain, weights higher ranks more

Why these metrics:
- Precision@5 is the user-facing number: 'of what I show you, how much is useful?'
- Hit@1 is the 'obvious case' number: 'when there's an obvious right answer, does it surface?'
- NDCG@5 measures ranking quality: rank 1 should be better than rank 5; this metric penalizes
  weak matches at the top and rewards strong matches at the top.

Relevance grades map to numeric scores:
  relevant     -> 1.0
  partial      -> 0.5
  not_relevant -> 0.0

The script runs the live retriever against each query and verifies the top-5 codes match
what we rated. If they DON'T match (e.g., corpus has been rebuilt since rating), we flag a
mismatch and exclude that query from metrics with a clear warning.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import structlog

from askthestacks.config import get_settings
from askthestacks.retrieval import load_retriever

log = structlog.get_logger()

RELEVANCE_GRADES: dict[str, float] = {
    "relevant": 1.0,
    "partial": 0.5,
    "not_relevant": 0.0,
}

EVAL_FILE = Path("eval/queries.json")


def load_eval_set(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def precision_at_k(grades: list[float], k: int = 5) -> float:
    """Sum of relevance scores divided by k. Max = 1.0 (all top-5 fully relevant)."""
    top_k = grades[:k]
    if not top_k:
        return 0.0
    return sum(top_k) / k


def hit_at_1(grades: list[float]) -> float:
    """1.0 if top result is relevant or partial; 0.0 otherwise."""
    if not grades:
        return 0.0
    return 1.0 if grades[0] > 0 else 0.0


def ndcg_at_k(grades: list[float], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k.

    DCG weights higher ranks more (rank 1 worth more than rank 5).
    NDCG normalizes DCG against the ideal ordering (sorted descending) so result is 0-1.
    Score of 1.0 means the system returned results sorted optimally.
    """
    top_k = grades[:k]
    if not top_k or sum(top_k) == 0:
        return 0.0

    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(top_k))
    ideal = sorted(top_k, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(retriever, query_entry: dict) -> dict:
    """Run one query through retriever and compare against rated codes.

    Returns metrics + diagnostics for this query.
    """
    query = query_entry["query"]
    rated = query_entry["rated_results"]
    rated_codes = [r["code"] for r in rated]

    live_results = retriever.search(query, top_k=5)
    live_codes = [r.entry.code for r in live_results]

    # Detect drift: did the corpus return different codes than what was rated?
    drift = live_codes != rated_codes
    if drift:
        log.warning(
            "rated_codes_dont_match_live",
            query=query,
            rated=rated_codes,
            live=live_codes,
        )

    # Build the grade list using the live results, looking up each code in the rated set.
    # If a live code wasn't rated, we treat it as not_relevant (0.0) and flag it.
    rated_lookup = {r["code"]: r["relevance"] for r in rated}
    unrated_live_codes = [c for c in live_codes if c not in rated_lookup]

    grades = [
        RELEVANCE_GRADES.get(rated_lookup.get(c, "not_relevant"), 0.0)
        for c in live_codes
    ]

    return {
        "id": query_entry["id"],
        "query": query,
        "domain": query_entry.get("domain", "unknown"),
        "drift": drift,
        "live_codes": live_codes,
        "unrated_live_codes": unrated_live_codes,
        "grades": grades,
        "precision_at_5": precision_at_k(grades, 5),
        "hit_at_1": hit_at_1(grades),
        "ndcg_at_5": ndcg_at_k(grades, 5),
    }


def print_summary(per_query: list[dict]) -> None:
    """Print a human-readable summary table and aggregate metrics."""
    print("\n" + "=" * 80)
    print(f"{'Query ID':28} {'Domain':14} {'Hit@1':>6} {'P@5':>6} {'NDCG@5':>8}")
    print("-" * 80)

    for r in per_query:
        drift_mark = " (drift)" if r["drift"] else ""
        print(
            f"{r['id']:28} {r['domain']:14} "
            f"{r['hit_at_1']:>6.2f} "
            f"{r['precision_at_5']:>6.2f} "
            f"{r['ndcg_at_5']:>8.2f}"
            f"{drift_mark}"
        )

    print("-" * 80)
    n = len(per_query)
    if n == 0:
        print("No queries evaluated.")
        return

    avg_hit = sum(r["hit_at_1"] for r in per_query) / n
    avg_p5 = sum(r["precision_at_5"] for r in per_query) / n
    avg_ndcg = sum(r["ndcg_at_5"] for r in per_query) / n
    drift_count = sum(1 for r in per_query if r["drift"])

    print(f"{'AVERAGE (n=' + str(n) + ')':28} {'':14} "
          f"{avg_hit:>6.2f} {avg_p5:>6.2f} {avg_ndcg:>8.2f}")
    print()
    print(
        f"Hit@1:          {avg_hit:.2%}   (top result was relevant or partial)")
    print(f"Precision@5:    {avg_p5:.2%}   (relevance-weighted top-5 quality)")
    print(
        f"NDCG@5:         {avg_ndcg:.2%}   (ranking quality, 100% = optimal ordering)")
    if drift_count:
        print()
        print(
            f"WARNING: {drift_count}/{n} queries returned different live codes "
            f"than the rated codes. Re-rate them or rebuild the eval set."
        )


def main() -> int:
    settings = get_settings()
    eval_set = load_eval_set(EVAL_FILE)
    log.info("eval_loaded", queries=len(eval_set["queries"]))

    retriever = load_retriever(settings.corpus_path, settings.index_dir)

    per_query = [evaluate_query(retriever, q) for q in eval_set["queries"]]
    print_summary(per_query)

    # Write a JSON report alongside the metrics for traceability
    output_path = Path("eval/last_run.json")
    output_path.write_text(json.dumps(per_query, indent=2), encoding="utf-8")
    print(f"\nDetailed report written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
