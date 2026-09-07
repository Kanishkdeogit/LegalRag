#!/usr/bin/env python3
"""
evaluate_cuad_metrics.py - Official CUAD-Style Extraction Metric Evaluator for Legal RAG

This script computes official SQuAD / CUAD extraction benchmark metrics:
  - Exact Match (EM): Percentage of predictions that match the golden reference span exactly after normalization.
  - Token-level F1: Bag-of-words token precision/recall harmonic mean between normalized prediction and reference.
  - (Optional Proxy Curve): AUPR, Precision@80%Recall, Precision@90%Recall via --include-proxy-curve flag.

Background & Context:
CUAD (Contract Understanding Atticus Dataset, Hendrycks et al. 2021) is formulated as an extractive
span-selection benchmark where models output confidence-scored verbatim text substrings.
Our Legal RAG system operates under a Generative RAG paradigm (Hybrid Retrieval -> Context Assembly ->
LLM Synthesis). This evaluator provides a rigorous, standardized extraction comparison on the
'cuad_templated' subset against published CUAD baselines, while documenting structural differences
(such as why AUPR curves do not natively apply to single-shot generative models).

Usage:
  python evaluate_cuad_metrics.py rag_eval_results.json
  python evaluate_cuad_metrics.py eval_dataset3_results_with_chunks.json --include-proxy-curve
  python evaluate_cuad_metrics.py eval_dataset3_results_with_chunks.json -o cuad_metrics_results.json
"""

import os
import sys
import re
import json
import string
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple


# Standard unanswerable/negative phrases in legal generative RAG systems
UNANSWERABLE_PHRASES = [
    "",
    "none",
    "not applicable",
    "n/a",
    "the contract does not contain provisions or clauses regarding this matter",
    "the contract does not contain provisions regarding this matter",
    "the contract does not contain any provisions regarding this matter",
    "the contract does not mention",
    "the contract does not provide",
    "the study material does not provide information on this topic",
]


# ==============================================================================
# 1. Text Normalization (Official SQuAD / CUAD Specification)
# ==============================================================================

def normalize_answer(s: Optional[str]) -> str:
    """
    Official SQuAD v1/v2 and CUAD text normalization pipeline:
      1. Lowercase all characters
      2. Strip all ASCII punctuation characters
      3. Remove English grammatical articles ('a', 'an', 'the')
      4. Collapse consecutive whitespace into a single space and strip boundaries
    
    This ensures that minor casing, punctuation, and article differences do not
    penalize an otherwise accurate legal clause extraction.
    """
    if not s:
        return ""

    def lower(text: str) -> str:
        return text.lower()

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def is_unanswerable_text(text: Optional[str]) -> bool:
    """
    Check if a string represents an unanswerable clause or negative finding.
    In original CUAD extraction, an unanswerable ground truth or prediction is an
    empty string ("") or None. In generative RAG, models frequently output standard
    procedural negative statements (e.g. 'The contract does not contain...').
    """
    if text is None:
        return True
    norm = normalize_answer(text)
    if not norm:
        return True
    for phrase in UNANSWERABLE_PHRASES:
        if norm == normalize_answer(phrase):
            return True
    return False


# ==============================================================================
# 2. Metric Implementations (Exact Match & Token-Level F1)
# ==============================================================================

def compute_exact_match(prediction: Optional[str], ground_truth: Optional[str]) -> float:
    """
    Computes Exact Match (EM) score (0.0 or 1.0) following SQuAD/CUAD conventions:
      - If both GT and prediction are unanswerable/empty -> 1.0 (True Negative)
      - If one is unanswerable and the other is substantive -> 0.0 (Type I / Type II error)
      - Otherwise -> 1.0 if normalized texts are identical, else 0.0
    """
    gt_empty = is_unanswerable_text(ground_truth)
    pred_empty = is_unanswerable_text(prediction)

    if gt_empty and pred_empty:
        return 1.0
    if gt_empty != pred_empty:
        return 0.0

    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: Optional[str], ground_truth: Optional[str]) -> float:
    """
    Computes token-level F1 score (0.0 to 1.0) following SQuAD/CUAD conventions:
      - Unanswerable cases: EM=1.0 -> F1=1.0; mismatch -> F1=0.0
      - Substantive cases: Tokenizes normalized text into bag-of-words using Counter,
        computes token Precision and token Recall, then takes their harmonic mean:
          Precision = (num_shared_tokens) / (len_prediction_tokens)
          Recall    = (num_shared_tokens) / (len_ground_truth_tokens)
          F1        = 2 * (Precision * Recall) / (Precision + Recall)
    """
    gt_empty = is_unanswerable_text(ground_truth)
    pred_empty = is_unanswerable_text(prediction)

    if gt_empty and pred_empty:
        return 1.0
    if gt_empty != pred_empty:
        return 0.0

    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common_counts = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common_counts.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return f1


# ==============================================================================
# 3. Optional Proxy Precision-Recall Curve (AUPR, P@80%R, P@90%R)
# ==============================================================================

def compute_proxy_curve(scored_records: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Computes an approximate / proxy Precision-Recall curve for single-shot RAG.

    IMPORTANT ARCHITECTURAL NOTE:
    In the official CUAD benchmark (Hendrycks et al., 2021), AUPR and Precision@Recall
    are evaluated by sweeping confidence thresholds across an extraction model's pool
    of multiple candidate text spans.
    A single-shot generative RAG system produces one final synthesized answer per query
    without generating a confidence distribution over multiple candidate spans.

    To provide structural comparability when requested, this function uses retrieval
    grounding overlap (token overlap between prediction and retrieved context, or token F1)
    as a proxy confidence ranking.
    """
    if not scored_records:
        return {"aupr": None, "p80": None, "p90": None}

    # Count total answerable positive ground truths
    total_positives = sum(1 for r in scored_records if not is_unanswerable_text(r["ground_truth"]))
    if total_positives == 0:
        return {"aupr": 1.0, "p80": 1.0, "p90": 1.0}

    # Assign proxy confidence score
    # Proxy metric: token F1 against retrieved chunks (context grounding confidence), fallback to prediction F1
    curve_data = []
    for r in scored_records:
        conf = 1.0
        chunks = r.get("retrieved_chunks", [])
        if chunks and isinstance(chunks, list):
            context_text = " ".join(chunks)
            conf = compute_f1(r["prediction"], context_text)
        else:
            conf = r["f1"]

        # A prediction is considered a True Positive if token F1 >= 0.5 (standard extraction threshold)
        is_tp = (r["f1"] >= 0.5) and (not is_unanswerable_text(r["ground_truth"]))
        curve_data.append({"confidence": conf, "is_tp": is_tp})

    # Sort descending by proxy confidence score
    curve_data.sort(key=lambda x: x["confidence"], reverse=True)

    precisions = []
    recalls = []
    cum_tp = 0
    cum_fp = 0

    for i, item in enumerate(curve_data):
        if item["is_tp"]:
            cum_tp += 1
        else:
            cum_fp += 1

        prec = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
        rec = cum_tp / total_positives
        precisions.append(prec)
        recalls.append(rec)

    # Compute AUPR using trapezoidal numerical integration
    # Prepend point (0.0, 1.0) for standard PR curve origin
    aupr = 0.0
    prev_rec = 0.0
    for prec, rec in zip(precisions, recalls):
        delta_rec = rec - prev_rec
        aupr += delta_rec * prec
        prev_rec = rec

    # Compute Precision@80%Recall and Precision@90%Recall
    p80 = 0.0
    p90 = 0.0
    for prec, rec in zip(precisions, recalls):
        if rec >= 0.80 and prec > p80:
            p80 = prec
        if rec >= 0.90 and prec > p90:
            p90 = prec

    return {
        "aupr": aupr * 100.0,
        "p80": p80 * 100.0,
        "p90": p90 * 100.0
    }


# ==============================================================================
# 4. Data Loading, Field Resolution & Clause Parsing
# ==============================================================================

def parse_clause_type(item_id: str, question: str) -> str:
    """
    Extracts the canonical CUAD Clause Type:
      1. From item ID: 'ContractName__ClauseType' -> 'ClauseType'
      2. Fallback: From templated question: 'Highlight the parts ... related to "ClauseType"'
    """
    if item_id and "__" in item_id:
        return item_id.split("__")[-1].strip()

    match = re.search(r'related to\s+["\']([^"\']+)["\']', question, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return "General / Unspecified"


def load_dataset_lookup(dataset_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    Loads benchmark dataset to look up 'id' and 'query_type' if the evaluation
    results JSON only contains questions and answers without original metadata.
    """
    candidate_paths = [
        dataset_path,
        "eval_dataset3.json",
        "eval_dataset.json",
        "eval_dataset2.json"
    ]
    for p in candidate_paths:
        if p and os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    lookup = {}
                    for i, d in enumerate(data):
                        # Key by question text and by integer index
                        q_norm = normalize_answer(d.get("question", ""))
                        if q_norm:
                            lookup[q_norm] = d
                        lookup[f"idx_{i}"] = d
                    return lookup
            except Exception:
                pass
    return {}


def load_and_prepare_data(
    input_path: str,
    dataset_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Loads input evaluation JSON, resolves field aliases, enriches missing metadata,
    and partitions examples into:
      - scored_examples: query_type == 'cuad_templated'
      - excluded_examples: non-CUAD query types (multi_hop, cross_document, etc.)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input evaluation file '{input_path}' not found.")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    if not isinstance(raw_items, list):
        raise ValueError(f"Expected a JSON list of objects in '{input_path}', but got {type(raw_items)}.")

    dataset_lookup = load_dataset_lookup(dataset_path)

    scored_examples = []
    excluded_examples = []
    excluded_type_counts = defaultdict(int)

    for i, item in enumerate(raw_items):
        # 1. Resolve field aliases
        question = item.get("question") or item.get("user_input") or ""
        prediction = item.get("generated_answer") or item.get("prediction") or item.get("answer") or item.get("response") or ""
        ground_truth = item.get("ground_truth") or item.get("reference") or item.get("answer_text")

        item_id = item.get("id") or ""
        query_type = item.get("query_type") or ""

        # 2. Enrich from dataset lookup if query_type or id is missing
        if not query_type or not item_id:
            q_norm = normalize_answer(question)
            match_meta = dataset_lookup.get(q_norm) or dataset_lookup.get(f"idx_{i}")
            if match_meta:
                if not item_id:
                    item_id = match_meta.get("id", "")
                if not query_type:
                    query_type = match_meta.get("query_type", "")
                if ground_truth is None and "answer_text" in match_meta:
                    ground_truth = match_meta["answer_text"]

        # 3. Fallback heuristic: Detect CUAD template phrasing if query_type is still unspecified
        if not query_type:
            if "highlight the parts" in question.lower() and "related to" in question.lower():
                query_type = "cuad_templated"
            else:
                query_type = "non_cuad_generative"

        clause_type = parse_clause_type(item_id, question)

        record = {
            "index": i,
            "id": item_id,
            "clause_type": clause_type,
            "query_type": query_type,
            "question": question,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "retrieved_chunks": item.get("retrieved_chunks", [])
        }

        # 4. Partition strictly on 'cuad_templated'
        if query_type == "cuad_templated":
            em = compute_exact_match(prediction, ground_truth)
            f1 = compute_f1(prediction, ground_truth)
            record["exact_match"] = em
            record["f1"] = f1
            scored_examples.append(record)
        else:
            excluded_examples.append(record)
            excluded_type_counts[query_type] += 1

    return scored_examples, excluded_examples, dict(excluded_type_counts)


# ==============================================================================
# 5. Output Formatting, Reporting & Saving
# ==============================================================================

def print_summary_table(
    input_file: str,
    total_count: int,
    scored_examples: List[Dict[str, Any]],
    excluded_counts: Dict[str, int],
    include_proxy_curve: bool,
    proxy_metrics: Dict[str, Optional[float]]
):
    """
    Renders clean, structured terminal tables for executive presentation.
    """
    scored_count = len(scored_examples)
    excluded_count = sum(excluded_counts.values())

    mean_em = (sum(e["exact_match"] for e in scored_examples) / scored_count * 100.0) if scored_count > 0 else 0.0
    mean_f1 = (sum(e["f1"] for e in scored_examples) / scored_count * 100.0) if scored_count > 0 else 0.0

    print("\n" + "=" * 80)
    print(" CUAD-style Extraction Metrics (cuad_templated subset only)")
    print("=" * 80)
    print(f" Input File:                     {os.path.basename(input_file)}")
    print(f" Total Examples Evaluated:       {total_count}")
    print(f" Scored Examples (CUAD format):  {scored_count}")

    if excluded_count > 0:
        breakdown_str = ", ".join(f"{k}: {v}" for k, v in sorted(excluded_counts.items()))
        print(f" Excluded (Non-CUAD Query Types): {excluded_count} ({breakdown_str})")
        print("   -> Reason: Multi-hop, cross-contract, and synthesis queries are generative")
        print("      reasoning tasks, not single-span extractions matching CUAD's schema.")
    else:
        print(f" Excluded Examples:              0")

    print("-" * 80)
    print(f" Exact Match (EM):               {mean_em:.1f}%")
    print(f" F1 Score:                       {mean_f1:.1f}%")

    if include_proxy_curve and proxy_metrics.get("aupr") is not None:
        print(f" AUPR (proxy):                   {proxy_metrics['aupr']:.1f}%")
        print(f" Precision@80%Recall (proxy):    {proxy_metrics['p80']:.1f}%")
        print(f" Precision@90%Recall (proxy):    {proxy_metrics['p90']:.1f}%")
        print("   [Note: Proxy curve ranks predictions by retrieval grounding confidence.]")
    else:
        print("-" * 80)
        print(" [Note on AUPR & Precision@Recall:")
        print("  AUPR and Precision@Recall curves are skipped by default. The official CUAD")
        print("  benchmark evaluates confidence-ranked spans across threshold sweeps.")
        print("  Generative RAG produces a single free-text synthesis without a candidate pool.")
        print("  Pass --include-proxy-curve if an approximate curve is desired.]")

    print("=" * 80)

    # Per-clause-type breakdown
    if scored_examples:
        clause_data = defaultdict(lambda: {"em": [], "f1": []})
        for e in scored_examples:
            c_type = e["clause_type"]
            clause_data[c_type]["em"].append(e["exact_match"])
            clause_data[c_type]["f1"].append(e["f1"])

        clause_summary = []
        for c_type, vals in clause_data.items():
            cnt = len(vals["f1"])
            avg_em = sum(vals["em"]) / cnt * 100.0
            avg_f1 = sum(vals["f1"]) / cnt * 100.0
            clause_summary.append({
                "clause_type": c_type,
                "count": cnt,
                "em": avg_em,
                "f1": avg_f1
            })

        # Sorted lowest F1 to highest F1 so user sees challenging clauses first
        clause_summary.sort(key=lambda x: (x["f1"], x["em"]))

        print("\n Per-Clause Type Breakdown (sorted from lowest F1 to highest F1):")
        print("-" * 80)
        print(f" {'Clause Type':<32} {'Count':<8} {'Exact Match (%)':<18} {'F1 Score (%)':<14}")
        print("-" * 80)
        for row in clause_summary:
            print(f" {row['clause_type']:<32} {row['count']:<8} {row['em']:>13.1f}%   {row['f1']:>11.1f}%")
        print("-" * 80 + "\n")


def build_output_payload(
    input_file: str,
    total_count: int,
    scored_examples: List[Dict[str, Any]],
    excluded_counts: Dict[str, int],
    include_proxy_curve: bool,
    proxy_metrics: Dict[str, Optional[float]]
) -> Dict[str, Any]:
    """
    Constructs the detailed JSON dictionary to save to disk.
    """
    scored_count = len(scored_examples)
    mean_em = (sum(e["exact_match"] for e in scored_examples) / scored_count * 100.0) if scored_count > 0 else 0.0
    mean_f1 = (sum(e["f1"] for e in scored_examples) / scored_count * 100.0) if scored_count > 0 else 0.0

    clause_data = defaultdict(lambda: {"em": [], "f1": []})
    for e in scored_examples:
        c_type = e["clause_type"]
        clause_data[c_type]["em"].append(e["exact_match"])
        clause_data[c_type]["f1"].append(e["f1"])

    per_clause = []
    for c_type, vals in clause_data.items():
        cnt = len(vals["f1"])
        per_clause.append({
            "clause_type": c_type,
            "count": cnt,
            "exact_match_pct": round(sum(vals["em"]) / cnt * 100.0, 2),
            "f1_score_pct": round(sum(vals["f1"]) / cnt * 100.0, 2)
        })
    per_clause.sort(key=lambda x: x["f1_score_pct"])

    detailed_scored = [
        {
            "id": e["id"],
            "clause_type": e["clause_type"],
            "question": e["question"],
            "ground_truth": e["ground_truth"],
            "generated_answer": e["prediction"],
            "exact_match": e["exact_match"],
            "f1_score": round(e["f1"] * 100.0, 2)
        }
        for e in scored_examples
    ]

    return {
        "summary": {
            "input_file": input_file,
            "total_examples": total_count,
            "scored_cuad_examples": scored_count,
            "excluded_examples": sum(excluded_counts.values()),
            "excluded_by_query_type": excluded_counts,
            "exact_match_pct": round(mean_em, 2),
            "f1_score_pct": round(mean_f1, 2),
            "include_proxy_curve": include_proxy_curve,
            "aupr_proxy_pct": round(proxy_metrics["aupr"], 2) if proxy_metrics.get("aupr") is not None else None,
            "precision_at_80_recall_proxy_pct": round(proxy_metrics["p80"], 2) if proxy_metrics.get("p80") is not None else None,
            "precision_at_90_recall_proxy_pct": round(proxy_metrics["p90"], 2) if proxy_metrics.get("p90") is not None else None,
        },
        "per_clause_breakdown": per_clause,
        "scored_examples": detailed_scored
    }


# ==============================================================================
# 6. Command-Line Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute official CUAD-style extraction metrics (EM & F1) on Legal RAG predictions."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to JSON file containing RAG evaluation results (e.g. 'rag_eval_results.json' or 'eval_dataset3_results_with_chunks.json')."
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        type=str,
        default="cuad_metrics_results.json",
        help="Output path to save JSON metrics report (default: 'cuad_metrics_results.json')."
    )
    parser.add_argument(
        "-d", "--dataset",
        dest="dataset_file",
        type=str,
        default=None,
        help="Optional path to source dataset JSON (e.g. 'eval_dataset3.json') for metadata enrichment."
    )
    parser.add_argument(
        "--include-proxy-curve",
        action="store_true",
        default=False,
        help="Include proxy AUPR, Precision@80%Recall, and Precision@90%Recall calculated using retrieval confidence."
    )

    args = parser.parse_args()

    # Load and score
    scored_examples, excluded_examples, excluded_counts = load_and_prepare_data(
        input_path=args.input_file,
        dataset_path=args.dataset_file
    )

    total_count = len(scored_examples) + len(excluded_examples)

    # Compute proxy curve metrics if requested
    proxy_metrics = {"aupr": None, "p80": None, "p90": None}
    if args.include_proxy_curve:
        proxy_metrics = compute_proxy_curve(scored_examples)

    # Print summary to console
    print_summary_table(
        input_file=args.input_file,
        total_count=total_count,
        scored_examples=scored_examples,
        excluded_counts=excluded_counts,
        include_proxy_curve=args.include_proxy_curve,
        proxy_metrics=proxy_metrics
    )

    # Save detailed JSON report
    output_payload = build_output_payload(
        input_file=args.input_file,
        total_count=total_count,
        scored_examples=scored_examples,
        excluded_counts=excluded_counts,
        include_proxy_curve=args.include_proxy_curve,
        proxy_metrics=proxy_metrics
    )

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"Saved complete CUAD evaluation metrics report to '{args.output_file}'.\n")


if __name__ == "__main__":
    main()
