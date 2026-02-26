"""
quality_checker.py
==================
Annotation Quality Assurance Module

Performs multi-layer quality checks on annotated datasets:

  1. Completeness Check    — missing labels, empty text, null fields
  2. Consistency Check     — inter-annotator agreement (Cohen's Kappa, Krippendorff's α)
  3. Confidence Analysis   — flag low-confidence items needing human review
  4. Bias Detection        — class imbalance, label drift across time/batches
  5. Override Analysis     — human correction rate by confidence band
  6. Duplicate Detection   — exact and near-duplicate texts
  7. Text Quality          — length extremes, encoding issues, boilerplate
  8. Golden Set Validation — accuracy vs a reference set of known labels
  9. HTML Report Generator — standalone HTML report with all metrics

Usage examples
--------------
  # Check a single annotated CSV:
  python quality_checker.py --input data/annotated/sample_sentiment_20240101.csv

  # Compare two annotator files (inter-annotator agreement):
  python quality_checker.py --annotator1 ann_a.csv --annotator2 ann_b.csv

  # Validate against a golden set:
  python quality_checker.py --input ann.csv --golden data/golden_set.csv

  # Full pipeline check with HTML report:
  python quality_checker.py --input ann.csv --report html --output reports/

Author : Qowiyu Yusrizal <hihakai123@gmail.com>
GitHub : https://github.com/Exynos93
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent
VERSION  = "1.0.0"

# Severity colour codes (for console)
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


# ═════════════════════════════════════════════════════════════════════════════
# LOGGER
# ═════════════════════════════════════════════════════════════════════════════
def _get_logger() -> logging.Logger:
    log = logging.getLogger("quality_checker")
    if not log.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s",
                                          datefmt="%H:%M:%S"))
        log.addHandler(ch)
    log.setLevel(logging.INFO)
    return log

logger = _get_logger()


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ═════════════════════════════════════════════════════════════════════════════
REQUIRED_COLS = {"text", "final_label", "auto_label", "auto_confidence"}
OPTIONAL_COLS = {"source_id", "annotator", "is_human_reviewed",
                 "human_override", "created_at", "annotation_time_sec"}


def load_annotated(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip().str.lower()

    # Flexible column mapping
    col_aliases = {
        "label":       "final_label",
        "prediction":  "auto_label",
        "confidence":  "auto_confidence",
        "content":     "text",
        "review":      "text",
        "comment":     "text",
    }
    for alias, canonical in col_aliases.items():
        if alias in df.columns and canonical not in df.columns:
            df.rename(columns={alias: canonical}, inplace=True)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"File columns: {list(df.columns)}"
        )

    # Type coerce
    df["auto_confidence"] = pd.to_numeric(df["auto_confidence"], errors="coerce").fillna(0.0)
    for bool_col in ["is_human_reviewed", "human_override"]:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].map(
                {"True": True, "False": False, "1": True, "0": False, "": False}
            ).fillna(False)

    logger.info(f"Loaded {len(df):,} records from '{path.name}'")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 1 · COMPLETENESS
# ═════════════════════════════════════════════════════════════════════════════
def check_completeness(df: pd.DataFrame) -> dict:
    """Find missing values, empty text, and null labels."""
    results = {
        "check": "Completeness",
        "total_records": len(df),
        "issues": [],
        "flags": {},
        "passed": True,
    }

    # Missing final_label
    missing_label = df["final_label"].isin(["", "nan", "None", "null"]).sum()
    if missing_label:
        results["issues"].append(f"{missing_label:,} records have no final_label")
        results["flags"]["missing_labels"] = int(missing_label)
        results["passed"] = False

    # Empty text
    empty_text = (df["text"].str.strip() == "").sum()
    if empty_text:
        results["issues"].append(f"{empty_text:,} records have empty text")
        results["flags"]["empty_text"] = int(empty_text)
        results["passed"] = False

    # Missing confidence
    missing_conf = df["auto_confidence"].isna().sum()
    if missing_conf:
        results["issues"].append(f"{missing_conf:,} records missing confidence score")
        results["flags"]["missing_confidence"] = int(missing_conf)

    # Null auto_label
    missing_auto = df["auto_label"].isin(["", "nan", "None", "null"]).sum()
    if missing_auto:
        results["issues"].append(f"{missing_auto:,} records have no auto_label")
        results["flags"]["missing_auto_labels"] = int(missing_auto)

    results["completeness_pct"] = round(
        (1 - (missing_label + empty_text) / len(df)) * 100, 2
    ) if len(df) else 0

    return results


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 2 · CONFIDENCE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def check_confidence(df: pd.DataFrame, threshold: float = 0.20) -> dict:
    """Analyse confidence score distribution and flag low-confidence items."""
    confs = df["auto_confidence"].dropna()
    n = len(confs)

    buckets = {
        "zero (no signal)":    int((confs == 0.0).sum()),
        f"very low (<{threshold:.0%})": int(((confs > 0) & (confs < threshold)).sum()),
        "low (20–40%)":        int(((confs >= 0.20) & (confs < 0.40)).sum()),
        "medium (40–70%)":     int(((confs >= 0.40) & (confs < 0.70)).sum()),
        "high (≥70%)":         int((confs >= 0.70).sum()),
    }

    needs_review = int((confs < threshold).sum())
    review_pct   = round(needs_review / n * 100, 2) if n else 0

    return {
        "check":             "Confidence Analysis",
        "threshold":         threshold,
        "stats": {
            "mean":   round(float(confs.mean()),   4) if n else 0,
            "median": round(float(confs.median()), 4) if n else 0,
            "std":    round(float(confs.std()),    4) if n else 0,
            "min":    round(float(confs.min()),    4) if n else 0,
            "max":    round(float(confs.max()),    4) if n else 0,
        },
        "buckets":           buckets,
        "needs_review_count": needs_review,
        "needs_review_pct":  review_pct,
        "passed":            review_pct < 30,  # pass if <30% need review
        "issues": [
            f"{needs_review:,} items ({review_pct}%) below confidence threshold {threshold:.0%}"
        ] if needs_review > 0 else [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 3 · LABEL DISTRIBUTION & BIAS
# ═════════════════════════════════════════════════════════════════════════════
def check_label_distribution(df: pd.DataFrame) -> dict:
    """Detect class imbalance and label distribution issues."""
    label_counts   = df["final_label"].value_counts()
    n              = len(df)
    n_classes      = len(label_counts)
    expected_even  = 1 / n_classes if n_classes else 1

    dist = {}
    imbalance_flags = []
    for lbl, cnt in label_counts.items():
        pct = cnt / n * 100
        dist[str(lbl)] = {"count": int(cnt), "pct": round(pct, 2)}
        # Flag if any class is >5× the even share
        if pct > expected_even * 500:
            imbalance_flags.append(
                f"'{lbl}' is heavily dominant ({pct:.1f}% vs expected {expected_even*100:.1f}%)"
            )

    # Gini coefficient as imbalance measure (0=equal, 1=extreme imbalance)
    freq = np.array([v for v in label_counts.values], dtype=float)
    freq /= freq.sum()
    freq.sort()
    n_c   = len(freq)
    gini  = (2 * np.sum((np.arange(1, n_c+1)) * freq) - (n_c + 1)) / n_c if n_c > 1 else 0.0

    # Compare auto vs final labels (override rate)
    if "auto_label" in df.columns:
        match     = (df["auto_label"] == df["final_label"]).sum()
        agreement = round(match / n * 100, 2)
    else:
        agreement = None

    return {
        "check":             "Label Distribution",
        "n_classes":         n_classes,
        "distribution":      dist,
        "gini_coefficient":  round(float(gini), 4),
        "auto_final_agreement_pct": agreement,
        "imbalance_flags":   imbalance_flags,
        "passed":            len(imbalance_flags) == 0 and gini < 0.6,
        "issues":            imbalance_flags,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 4 · INTER-ANNOTATOR AGREEMENT (Cohen's Kappa)
# ═════════════════════════════════════════════════════════════════════════════
def cohens_kappa(labels_a: list, labels_b: list) -> float:
    """
    Compute Cohen's Kappa for two lists of labels.
    κ = (P_o - P_e) / (1 - P_e)
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Both label lists must have the same length.")
    n = len(labels_a)
    if n == 0:
        return 0.0

    classes = sorted(set(labels_a) | set(labels_b))
    k       = len(classes)
    idx     = {c: i for i, c in enumerate(classes)}

    # Confusion matrix
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(labels_a, labels_b):
        cm[idx[a]][idx[b]] += 1

    p_o = np.trace(cm) / n                       # observed agreement
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    p_e = float(np.dot(row_sums, col_sums)) / (n * n)   # expected agreement

    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 1.0
    return round(float(kappa), 4)


def interpret_kappa(k: float) -> str:
    if k < 0:      return "Poor (worse than chance)"
    elif k < 0.20: return "Slight"
    elif k < 0.40: return "Fair"
    elif k < 0.60: return "Moderate"
    elif k < 0.80: return "Substantial"
    else:          return "Almost Perfect"


def check_inter_annotator(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    id_col: str = "source_id",
) -> dict:
    """
    Compute agreement metrics between two annotator DataFrames.
    Both must have columns: source_id, final_label
    """
    # Align on shared IDs
    merged = df_a[[id_col, "final_label"]].merge(
        df_b[[id_col, "final_label"]],
        on=id_col, suffixes=("_a", "_b"), how="inner"
    )

    n_shared = len(merged)
    if n_shared == 0:
        return {"check": "Inter-Annotator Agreement", "error": "No shared IDs found."}

    labels_a = merged["final_label_a"].tolist()
    labels_b = merged["final_label_b"].tolist()

    exact_match  = sum(a == b for a, b in zip(labels_a, labels_b))
    accuracy     = round(exact_match / n_shared * 100, 2)
    kappa        = cohens_kappa(labels_a, labels_b)
    interpretation = interpret_kappa(kappa)

    # Disagreement analysis
    disagreements = merged[merged["final_label_a"] != merged["final_label_b"]]
    top_conflicts = (
        disagreements.groupby(["final_label_a", "final_label_b"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(5)
        .to_dict("records")
    )

    return {
        "check":            "Inter-Annotator Agreement",
        "shared_items":     n_shared,
        "exact_agreement_pct": accuracy,
        "cohens_kappa":     kappa,
        "kappa_interpretation": interpretation,
        "disagreement_count": len(disagreements),
        "top_conflicts":    top_conflicts,
        "passed":           kappa >= 0.60,  # Substantial or above
        "issues": [
            f"Kappa={kappa:.3f} ({interpretation}) — target ≥ 0.60"
        ] if kappa < 0.60 else [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 5 · DUPLICATE DETECTION
# ═════════════════════════════════════════════════════════════════════════════
def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace for near-duplicate detection."""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def check_duplicates(df: pd.DataFrame) -> dict:
    """Find exact and near-exact duplicate texts."""
    normalised  = df["text"].apply(_normalise)

    # Exact (after normalisation)
    dup_mask    = normalised.duplicated(keep=False)
    exact_count = int(dup_mask.sum())

    # Group duplicates with conflicting labels
    label_conflicts = []
    if exact_count > 0:
        groups = df[dup_mask].groupby(normalised[dup_mask])["final_label"].nunique()
        conflict_texts = groups[groups > 1].index.tolist()
        label_conflicts = conflict_texts[:5]  # show up to 5

    return {
        "check":             "Duplicate Detection",
        "exact_duplicates":  exact_count,
        "duplicate_pct":     round(exact_count / len(df) * 100, 2),
        "label_conflicts":   len(label_conflicts),
        "example_conflicts": label_conflicts,
        "passed":            exact_count == 0,
        "issues": [
            f"{exact_count:,} duplicate texts found "
            f"({len(label_conflicts)} with conflicting labels)"
        ] if exact_count > 0 else [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 6 · TEXT QUALITY
# ═════════════════════════════════════════════════════════════════════════════
def check_text_quality(df: pd.DataFrame) -> dict:
    """Flag length extremes, encoding issues, and boilerplate."""
    lengths   = df["text"].str.len()
    issues    = []
    flags     = {}

    # Very short (<5 chars)
    too_short = int((lengths < 5).sum())
    if too_short:
        issues.append(f"{too_short:,} texts under 5 characters")
        flags["too_short"] = too_short

    # Very long (>5000 chars)
    too_long = int((lengths > 5000).sum())
    if too_long:
        issues.append(f"{too_long:,} texts over 5,000 characters")
        flags["too_long"] = too_long

    # Encoding issues (replacement character)
    encoding_issues = int(df["text"].str.contains("\\ufffd", na=False).sum())
    if encoding_issues:
        issues.append(f"{encoding_issues:,} texts with encoding issues (U+FFFD)")
        flags["encoding_issues"] = encoding_issues

    # All-caps (shouting)
    all_caps = int(df["text"].apply(
        lambda t: t.isupper() and len(t.strip()) > 10
    ).sum())
    if all_caps:
        flags["all_caps"] = all_caps

    # Excessive punctuation (possible noise)
    excess_punct = int(df["text"].str.count(r"[!?]{3,}").gt(0).sum())
    if excess_punct:
        flags["excess_punctuation"] = excess_punct

    return {
        "check":  "Text Quality",
        "stats": {
            "mean_length":   round(float(lengths.mean()),   1),
            "median_length": round(float(lengths.median()), 1),
            "min_length":    int(lengths.min()),
            "max_length":    int(lengths.max()),
        },
        "flags":  flags,
        "passed": len(issues) == 0,
        "issues": issues,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 7 · GOLDEN SET VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
def check_golden_set(
    df:         pd.DataFrame,
    golden_df:  pd.DataFrame,
    id_col:     str = "source_id",
) -> dict:
    """
    Validate model labels against a known-correct reference set.
    golden_df must have: id_col, final_label (the ground truth)
    """
    merged = df[[id_col, "auto_label", "auto_confidence"]].merge(
        golden_df[[id_col, "final_label"]].rename(columns={"final_label": "golden_label"}),
        on=id_col,
        how="inner",
    )
    n = len(merged)
    if n == 0:
        return {"check": "Golden Set Validation", "error": "No matching IDs found."}

    correct = (merged["auto_label"] == merged["golden_label"]).sum()
    accuracy = round(correct / n * 100, 2)

    # Per-label precision / recall
    labels   = sorted(set(merged["golden_label"].unique()) | set(merged["auto_label"].unique()))
    per_label = {}
    for lbl in labels:
        tp = int(((merged["auto_label"] == lbl) & (merged["golden_label"] == lbl)).sum())
        fp = int(((merged["auto_label"] == lbl) & (merged["golden_label"] != lbl)).sum())
        fn = int(((merged["auto_label"] != lbl) & (merged["golden_label"] == lbl)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label[str(lbl)] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   tp + fn,
        }

    macro_f1 = round(float(np.mean([v["f1"] for v in per_label.values()])), 4)

    return {
        "check":          "Golden Set Validation",
        "n_validated":    n,
        "accuracy_pct":   accuracy,
        "macro_f1":       macro_f1,
        "per_label":      per_label,
        "passed":         accuracy >= 70.0,
        "issues": [
            f"Accuracy {accuracy}% below 70% target"
        ] if accuracy < 70 else [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 8 · HUMAN OVERRIDE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def check_override_analysis(df: pd.DataFrame) -> dict:
    """Analyse human override patterns to identify weak rules."""
    if "is_human_reviewed" not in df.columns:
        return {
            "check":   "Override Analysis",
            "skipped": True,
            "reason":  "No 'is_human_reviewed' column found.",
        }

    reviewed = df[df["is_human_reviewed"].astype(str).isin(["True", "1", "true"])]
    if len(reviewed) == 0:
        return {
            "check":   "Override Analysis",
            "skipped": True,
            "reason":  "No human-reviewed items in dataset.",
        }

    overrides     = reviewed[reviewed["human_override"].astype(str).isin(["True", "1", "true"])]
    override_rate = round(len(overrides) / len(reviewed) * 100, 2)

    # Override rate by confidence band
    bands = [(0, 0.0), (0.2, 0.0), (0.4, 0.2), (0.7, 0.4), (1.01, 0.7)]
    band_stats = {}
    for hi, lo in bands:
        band_df = reviewed[(reviewed["auto_confidence"] >= lo) &
                           (reviewed["auto_confidence"] < hi)]
        if len(band_df) == 0:
            continue
        band_overrides = int(band_df["human_override"].astype(str).isin(["True","1","true"]).sum())
        lbl = f"{lo:.0%}–{hi:.0%}" if hi < 1 else f"≥{lo:.0%}"
        band_stats[lbl] = {
            "reviewed":      len(band_df),
            "overrides":     int(band_overrides),
            "override_rate": round(band_overrides / len(band_df) * 100, 2),
        }

    # Most-overridden auto labels
    if len(overrides) > 0:
        override_from = dict(overrides["auto_label"].value_counts().head(5))
        override_to   = dict(overrides["final_label"].value_counts().head(5))
    else:
        override_from = {}
        override_to   = {}

    issues = []
    if override_rate > 25:
        issues.append(
            f"High override rate {override_rate}% — consider improving rules "
            f"for '{max(override_from, key=override_from.get, default='N/A')}'"
        )

    return {
        "check":              "Override Analysis",
        "human_reviewed":     len(reviewed),
        "overrides":          len(overrides),
        "overall_override_rate_pct": override_rate,
        "by_confidence_band": band_stats,
        "most_overridden_from": {str(k): int(v) for k, v in override_from.items()},
        "most_corrected_to":    {str(k): int(v) for k, v in override_to.items()},
        "passed":             override_rate < 25,
        "issues":             issues,
    }


# ═════════════════════════════════════════════════════════════════════════════
# HTML REPORT GENERATOR
# ═════════════════════════════════════════════════════════════════════════════
def _badge(passed: bool | None) -> str:
    if passed is None:  return "<span class='badge skip'>SKIP</span>"
    if passed:          return "<span class='badge pass'>PASS ✓</span>"
    return              "<span class='badge fail'>FAIL ✗</span>"


def generate_html_report(
    checks:    list[dict],
    df:        pd.DataFrame,
    input_name: str,
    output_dir: Path,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_pass    = sum(1 for c in checks if c.get("passed") is True)
    n_fail    = sum(1 for c in checks if c.get("passed") is False)
    n_skip    = sum(1 for c in checks if c.get("passed") is None)
    overall   = "PASS" if n_fail == 0 else "FAIL"
    oc        = "green" if overall == "PASS" else "red"

    # Label distribution data
    label_dist = df["final_label"].value_counts()
    ld_rows    = "".join(
        f"<tr><td>{lbl}</td><td>{cnt}</td>"
        f"<td>{cnt/len(df)*100:.1f}%</td></tr>"
        for lbl, cnt in label_dist.items()
    )

    checks_html = ""
    for c in checks:
        psd    = c.get("passed")
        name   = c.get("check", "Unknown")
        issues = c.get("issues", [])
        skipped= c.get("skipped", False)

        issue_html = ""
        if issues:
            issue_html = "<ul>" + "".join(f"<li class='issue'>{i}</li>" for i in issues) + "</ul>"
        elif skipped:
            issue_html = f"<p class='skip-note'>{c.get('reason','')}</p>"
        else:
            issue_html = "<p class='ok-note'>✓ No issues found.</p>"

        # Render check-specific detail tables
        detail_html = ""
        if "stats" in c and isinstance(c["stats"], dict):
            detail_html += "<table class='detail'><tr><th>Metric</th><th>Value</th></tr>"
            for k, v in c["stats"].items():
                detail_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
            detail_html += "</table>"

        if "distribution" in c and isinstance(c["distribution"], dict):
            detail_html += "<table class='detail'><tr><th>Label</th><th>Count</th><th>%</th></tr>"
            for lbl, info in c["distribution"].items():
                detail_html += f"<tr><td>{lbl}</td><td>{info['count']}</td><td>{info['pct']}%</td></tr>"
            detail_html += "</table>"

        if "per_label" in c:
            detail_html += "<table class='detail'><tr><th>Label</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>"
            for lbl, m in c["per_label"].items():
                detail_html += (f"<tr><td>{lbl}</td><td>{m['precision']:.2%}</td>"
                                f"<td>{m['recall']:.2%}</td><td>{m['f1']:.2%}</td>"
                                f"<td>{m['support']}</td></tr>")
            detail_html += "</table>"

        checks_html += f"""
        <div class='check-card {"pass-card" if psd else "fail-card" if psd is False else "skip-card"}'>
          <div class='check-header'>
            <span class='check-name'>{name}</span>
            {_badge(psd)}
          </div>
          <div class='check-body'>
            {issue_html}
            {detail_html}
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Annotation QA Report — {input_name}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fb; color: #1a1a2e; }}
    header {{ background: #1A56DB; color: white; padding: 24px 32px; }}
    header h1 {{ font-size: 22px; font-weight: 700; }}
    header p  {{ font-size: 13px; opacity: .8; margin-top: 4px; }}
    .container {{ max-width: 1100px; margin: 32px auto; padding: 0 24px; }}
    .summary-bar {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
    .kpi {{ background: white; border-radius: 10px; padding: 16px 20px;
             flex: 1; min-width: 140px; box-shadow: 0 1px 4px rgba(0,0,0,.07); }}
    .kpi .val {{ font-size: 28px; font-weight: 700; color: {oc}; }}
    .kpi .lbl {{ font-size: 11px; color: #6b7280; text-transform: uppercase; margin-top: 2px; }}
    .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
              font-size: 12px; font-weight: 700; }}
    .pass  {{ background: #d1fae5; color: #065f46; }}
    .fail  {{ background: #fee2e2; color: #991b1b; }}
    .skip  {{ background: #f3f4f6; color: #6b7280; }}
    .check-card {{ background: white; border-radius: 10px; margin-bottom: 16px;
                   box-shadow: 0 1px 4px rgba(0,0,0,.07); overflow: hidden; }}
    .pass-card {{ border-left: 5px solid #10b981; }}
    .fail-card {{ border-left: 5px solid #ef4444; }}
    .skip-card {{ border-left: 5px solid #d1d5db; }}
    .check-header {{ display: flex; align-items: center; justify-content: space-between;
                     padding: 14px 20px; background: #fafafa; border-bottom: 1px solid #f0f0f0; }}
    .check-name {{ font-weight: 700; font-size: 15px; }}
    .check-body  {{ padding: 14px 20px; }}
    .issue {{ color: #dc2626; margin: 4px 0; }}
    .ok-note   {{ color: #059669; }}
    .skip-note {{ color: #6b7280; font-style: italic; }}
    table.detail {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    table.detail th {{ background: #1A56DB; color: white; padding: 7px 10px; text-align: left; }}
    table.detail td {{ padding: 6px 10px; border-bottom: 1px solid #f0f0f0; }}
    table.detail tr:hover td {{ background: #f0f4ff; }}
    .dist-table {{ background: white; border-radius: 10px; padding: 20px;
                   box-shadow: 0 1px 4px rgba(0,0,0,.07); margin-bottom: 28px; }}
    footer {{ text-align: center; padding: 24px; color: #9ca3af; font-size: 12px; margin-top: 20px; }}
    h2 {{ font-size: 18px; margin-bottom: 14px; color: #1a1a2e; }}
  </style>
</head>
<body>
  <header>
    <h1>📋 Annotation Quality Report</h1>
    <p>File: <strong>{input_name}</strong> &nbsp;|&nbsp; Generated: {timestamp}
       &nbsp;|&nbsp; Author: Qowiyu Yusrizal &nbsp;|&nbsp;
       <a href="https://github.com/Exynos93" style="color:#93c5fd">github.com/Exynos93</a>
    </p>
  </header>

  <div class="container">

    <div class="summary-bar">
      <div class="kpi"><div class="val" style="color:{oc}">{overall}</div>
        <div class="lbl">Overall Status</div></div>
      <div class="kpi"><div class="val">{len(df):,}</div>
        <div class="lbl">Total Records</div></div>
      <div class="kpi"><div class="val" style="color:#10b981">{n_pass}</div>
        <div class="lbl">Checks Passed</div></div>
      <div class="kpi"><div class="val" style="color:#ef4444">{n_fail}</div>
        <div class="lbl">Checks Failed</div></div>
      <div class="kpi"><div class="val" style="color:#6b7280">{n_skip}</div>
        <div class="lbl">Checks Skipped</div></div>
    </div>

    <div class="dist-table">
      <h2>📊 Label Distribution</h2>
      <table class="detail">
        <tr><th>Label</th><th>Count</th><th>Share</th></tr>
        {ld_rows}
      </table>
    </div>

    <h2>🔍 Quality Checks ({len(checks)})</h2>
    {checks_html}

  </div>
  <footer>
    Annotation QA Report &nbsp;·&nbsp; Data Annotation Pipeline v{VERSION}
    &nbsp;·&nbsp; Qowiyu Yusrizal &nbsp;·&nbsp;
    <a href="https://github.com/Exynos93">github.com/Exynos93</a>
  </footer>
</body>
</html>"""

    out_path = output_dir / f"qa_report_{Path(input_name).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"HTML report → {out_path}")
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════
class QualityChecker:
    """
    Run all quality checks on an annotated dataset and produce reports.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.20,
        output_dir: Path | None = None,
    ):
        self.threshold  = confidence_threshold
        self.output_dir = output_dir or (BASE_DIR / "data" / "exports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []

    # ── Run all checks ─────────────────────────────────────────────────────────
    def run(
        self,
        df:           pd.DataFrame,
        input_name:   str,
        golden_df:    pd.DataFrame | None = None,
        df_b:         pd.DataFrame | None = None,
    ) -> dict:
        """Execute all checks and return consolidated results."""
        print(f"\n{BOLD}{'═'*60}")
        print(f"  ANNOTATION QUALITY CHECKER  v{VERSION}")
        print(f"{'═'*60}{RESET}")
        print(f"  File         : {input_name}")
        print(f"  Records      : {len(df):,}")
        print(f"  Threshold    : {self.threshold:.0%}")
        print(f"{'─'*60}\n")

        self.results = []

        checks = [
            ("Completeness",         lambda: check_completeness(df)),
            ("Confidence Analysis",  lambda: check_confidence(df, self.threshold)),
            ("Label Distribution",   lambda: check_label_distribution(df)),
            ("Duplicate Detection",  lambda: check_duplicates(df)),
            ("Text Quality",         lambda: check_text_quality(df)),
            ("Override Analysis",    lambda: check_override_analysis(df)),
        ]

        if golden_df is not None:
            checks.append(("Golden Set Validation",
                           lambda: check_golden_set(df, golden_df)))

        if df_b is not None:
            checks.append(("Inter-Annotator Agreement",
                           lambda: check_inter_annotator(df, df_b)))

        for name, fn in checks:
            result = fn()
            self.results.append(result)
            passed = result.get("passed")
            status = (f"{GREEN}PASS{RESET}" if passed
                      else f"{RED}FAIL{RESET}" if passed is False
                      else f"{YELLOW}SKIP{RESET}")
            issues = result.get("issues", [])
            print(f"  [{status}]  {name}")
            for issue in issues:
                print(f"          {YELLOW}⚠  {issue}{RESET}")

        # ── Persist JSON ───────────────────────────────────────────────────────
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_out = self.output_dir / f"qa_results_{Path(input_name).stem}_{ts}.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "input":        input_name,
                "total_records": len(df),
                "checks":       self.results,
            }, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"QA JSON → {json_out}")

        return {"checks": self.results, "json_path": str(json_out)}

    # ── Console banner ─────────────────────────────────────────────────────────
    def print_final_banner(self):
        n_pass = sum(1 for c in self.results if c.get("passed") is True)
        n_fail = sum(1 for c in self.results if c.get("passed") is False)
        n_skip = sum(1 for c in self.results if c.get("passed") is None)
        overall = "PASS" if n_fail == 0 else "FAIL"
        colour  = GREEN if overall == "PASS" else RED

        print(f"\n{BOLD}{'═'*60}")
        print(f"  FINAL STATUS : {colour}{overall}{RESET}{BOLD}")
        print(f"  Passed : {GREEN}{n_pass}{RESET}{BOLD}   "
              f"Failed : {RED}{n_fail}{RESET}{BOLD}   "
              f"Skipped : {YELLOW}{n_skip}{RESET}")
        print(f"{'═'*60}{RESET}\n")

    # ── Generate HTML report ───────────────────────────────────────────────────
    def generate_report(self, df: pd.DataFrame, input_name: str) -> Path:
        return generate_html_report(
            self.results, df, input_name, self.output_dir
        )


# ═════════════════════════════════════════════════════════════════════════════
# GOLDEN SET GENERATOR  (creates a reference CSV for testing)
# ═════════════════════════════════════════════════════════════════════════════
GOLDEN_REFERENCES = {
    "sentiment": [
        ("g001", "This product exceeded my expectations completely. Love it!",    "Positive"),
        ("g002", "Worst purchase of my life. Totally broken on arrival.",          "Negative"),
        ("g003", "It is fine. Neither great nor terrible.",                        "Neutral"),
        ("g004", "Amazing quality, amazing service, amazing experience!",          "Positive"),
        ("g005", "I want a refund. This is not what was described.",               "Negative"),
        ("g006", "Average product. Does what it says.",                            "Neutral"),
        ("g007", "Absolutely fantastic! I recommend this to everyone!",            "Positive"),
        ("g008", "Delayed shipping, poor packaging, bad product.",                 "Negative"),
        ("g009", "Not bad, not great. About what I expected.",                     "Neutral"),
        ("g010", "Five stars! Perfect in every way. Will buy again!",              "Positive"),
    ],
    "intent": [
        ("g001", "What is machine learning?",                                      "Informational"),
        ("g002", "Buy discounted laptop online free delivery",                     "Transactional"),
        ("g003", "Login to my account",                                            "Navigational"),
        ("g004", "Hey, any restaurant recommendations near me?",                   "Conversational"),
        ("g005", "Explain how neural networks work step by step",                  "Informational"),
        ("g006", "Subscribe to premium plan with 20% discount",                    "Transactional"),
        ("g007", "Go to the official Apple support page",                          "Navigational"),
        ("g008", "Hi there! I was wondering what you think about AI?",             "Conversational"),
        ("g009", "History of the Roman Empire",                                    "Informational"),
        ("g010", "Download free ebook now",                                        "Transactional"),
    ],
    "content_category": [
        ("g001", "Earn money fast with this simple trick! Click here!",            "Spam"),
        ("g002", "Today we visited the botanical garden. Beautiful flowers! 🌸",   "Safe"),
        ("g003", "Scientists discovered they don't want you to know this cure!",   "Misinformation"),
        ("g004", "Our office is closed on Monday for a public holiday.",           "Safe"),
        ("g005", "Win a free iPhone! You've been selected — act now!",             "Spam"),
        ("g006", "Join us for the community cleanup drive this Saturday.",         "Safe"),
        ("g007", "100% guaranteed cure with no side effects doctors hate him",     "Misinformation"),
        ("g008", "Recipe of the week: homemade pasta with basil pesto",            "Safe"),
        ("g009", "F4F follow back instantly guaranteed",                           "Spam"),
        ("g010", "Team outing at the lake was so much fun! Great weather too.",    "Safe"),
    ],
}


def generate_golden_set(task: str) -> Path:
    refs = GOLDEN_REFERENCES.get(task)
    if refs is None:
        logger.warning(f"No golden set for task '{task}'. Using sentiment defaults.")
        refs = GOLDEN_REFERENCES["sentiment"]

    rows = [{"source_id": r[0], "text": r[1], "final_label": r[2]} for r in refs]
    path = BASE_DIR / "data" / "raw" / f"golden_{task}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"Golden set → {path}  ({len(rows)} items)")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quality_checker",
        description=(
            f"Annotation Quality Checker  v{VERSION}\n"
            "Author: Qowiyu Yusrizal  |  github.com/Exynos93"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  # Basic quality check on a single file:
  python quality_checker.py --input data/annotated/sample.csv

  # Full check with HTML report:
  python quality_checker.py --input data/annotated/sample.csv --report html

  # Validate against a golden set:
  python quality_checker.py --input data/annotated/ann.csv --golden data/raw/golden_sentiment.csv

  # Inter-annotator agreement between two annotators:
  python quality_checker.py --annotator1 ann_a.csv --annotator2 ann_b.csv

  # Generate a fresh golden set:
  python quality_checker.py --gen-golden sentiment
        """,
    )
    p.add_argument("--input",       "-i", help="Annotated CSV to check")
    p.add_argument("--golden",      "-g", help="Golden reference CSV (source_id, final_label)")
    p.add_argument("--annotator1",        help="First annotator CSV (for IAA)")
    p.add_argument("--annotator2",        help="Second annotator CSV (for IAA)")
    p.add_argument("--report",      choices=["none","json","html","all"], default="html",
                   help="Report format (default: html)")
    p.add_argument("--threshold",   type=float, default=0.20,
                   help="Low-confidence threshold (default: 0.20)")
    p.add_argument("--output-dir",  help="Directory for output reports")
    p.add_argument("--gen-golden",  metavar="TASK",
                   help="Generate a golden reference set for the given task")
    p.add_argument("--version",     action="version", version=f"%(prog)s {VERSION}")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════╗
║   ANNOTATION QUALITY CHECKER  v{VERSION}                  ║
║   Author: Qowiyu Yusrizal  |  github.com/Exynos93   ║
╚══════════════════════════════════════════════════════╝
""")

    # ── Generate golden set mode ───────────────────────────────────────────────
    if args.gen_golden:
        path = generate_golden_set(args.gen_golden)
        print(f"  Golden set saved to: {path}")
        return

    # ── Must have at least input or annotator1+2 ──────────────────────────────
    if not args.input and not (args.annotator1 and args.annotator2):
        parser.error("Provide --input OR (--annotator1 + --annotator2)")

    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "data" / "exports"
    checker    = QualityChecker(
        confidence_threshold=args.threshold,
        output_dir=output_dir,
    )

    # ── Load main file ─────────────────────────────────────────────────────────
    if args.input:
        df         = load_annotated(args.input)
        input_name = Path(args.input).name

        golden_df  = load_annotated(args.golden) if args.golden else None
        df_b       = load_annotated(args.annotator2) if args.annotator2 else None
        if args.annotator1:
            df = load_annotated(args.annotator1)

        checker.run(df, input_name, golden_df=golden_df, df_b=df_b)
        checker.print_final_banner()

        if args.report in ("html", "all"):
            report_path = checker.generate_report(df, input_name)
            print(f"  HTML report → {report_path}\n")

    elif args.annotator1 and args.annotator2:
        df_a = load_annotated(args.annotator1)
        df_b = load_annotated(args.annotator2)
        iaa  = check_inter_annotator(df_a, df_b)

        print(f"\n  Shared items    : {iaa['shared_items']:,}")
        print(f"  Agreement       : {iaa['exact_agreement_pct']}%")
        print(f"  Cohen's Kappa   : {iaa['cohens_kappa']}  → {iaa['kappa_interpretation']}")
        if iaa["top_conflicts"]:
            print("\n  Top disagreements:")
            for c in iaa["top_conflicts"][:5]:
                print(f"    '{c['final_label_a']}' vs '{c['final_label_b']}' : {c['count']}×")


if __name__ == "__main__":
    main()
