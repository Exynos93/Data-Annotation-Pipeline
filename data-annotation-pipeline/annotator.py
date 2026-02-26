"""
annotator.py
============
Semi-Automated Data Annotation Pipeline

Supports three annotation tasks:
  • sentiment     — Positive / Negative / Neutral
  • intent        — Informational / Transactional / Navigational / Conversational
  • content_category — Spam / Hate Speech / Violence / Safe / NSFW / Misinformation

Modes:
  --mode auto          Rule-based pre-labeling (no human needed)
  --mode interactive   Auto-label → human review per item
  --mode batch         Auto-label entire file, save JSON for Label Studio import

Usage examples
--------------
  # Auto-label a CSV of product reviews for sentiment:
  python annotator.py --input data/raw/reviews.csv --task sentiment --mode auto

  # Interactive session — review each auto-label and override:
  python annotator.py --input data/raw/reviews.csv --task sentiment --mode interactive

  # Batch-label for content moderation, export Label Studio JSON:
  python annotator.py --input data/raw/posts.csv --task content_category --mode batch --export ls

  # Use a custom config file:
  python annotator.py --input data/raw/chats.csv --task intent --mode auto --config config/intent_rules.json

Author : Qowiyu Yusrizal <hihakai123@gmail.com>
GitHub : https://github.com/Exynos93
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS & PATHS
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR       = Path(__file__).parent
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_ANNOTATED = BASE_DIR / "data" / "annotated"
DATA_EXPORTS   = BASE_DIR / "data" / "exports"
CONFIG_DIR     = BASE_DIR / "config"
LOGS_DIR       = BASE_DIR / "logs"

for d in [DATA_RAW, DATA_ANNOTATED, DATA_EXPORTS, CONFIG_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

VERSION = "1.0.0"

# ═════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═════════════════════════════════════════════════════════════════════════════
def setup_logger(name: str = "annotator") -> logging.Logger:
    log_file = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger   = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # File handler (full detail)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler (info+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logger()


# ═════════════════════════════════════════════════════════════════════════════
# RULE ENGINE — BUILT-IN KEYWORD DICTIONARIES
# ═════════════════════════════════════════════════════════════════════════════
# Each rule set maps label → list-of-regex patterns.
# Patterns are matched case-insensitively against the full text.
# First matching label wins; tie-break goes to the label with more matches.
# Confidence = matched_signals / total_signals_for_winner.

BUILT_IN_RULES: dict[str, dict[str, list[str]]] = {

    # ── Sentiment ─────────────────────────────────────────────────────────────
    "sentiment": {
        "Positive": [
            r"\b(love|loved|loving|amazing|awesome|excellent|fantastic|great|wonderful)\b",
            r"\b(perfect|brilliant|outstanding|superb|magnificent|delightful|impressed)\b",
            r"\b(happy|glad|pleased|satisfied|enjoyed|enjoy|thankful|grateful|appreciate)\b",
            r"\b(best|better|improve|improved|recommend|recommended|worth|worthy|value)\b",
            r"\b(fast|quick|smooth|easy|simple|clean|clear|friendly|helpful|useful)\b",
            r"[😊😍🥰😁🤩👍❤️💯✨🎉]",
            r"\b(good|nice|well|cool|fine|decent|solid|reliable|consistent|quality)\b",
        ],
        "Negative": [
            r"\b(terrible|horrible|awful|dreadful|disgusting|hate|hated|worst|bad)\b",
            r"\b(broken|defective|damaged|faulty|useless|worthless|waste|wasted)\b",
            r"\b(slow|delayed|late|never|didn't|won't|doesn't|can't|fail|failed)\b",
            r"\b(disappointed|disappointing|frustrat|annoying|annoyed|angry|upset)\b",
            r"\b(overpriced|expensive|cheap quality|rip.?off|scam|fraud|lie|lying)\b",
            r"\b(poor|weak|mediocre|lacking|missing|broken|crash|error|bug|issue)\b",
            r"[😠😡🤬👎💔😤😞😟]",
            r"\b(refund|return|complaint|problem|issue|wrong|incorrect|mislead)\b",
        ],
        "Neutral": [
            r"\b(okay|ok|alright|average|normal|standard|regular|typical|usual)\b",
            r"\b(neither|both|some|maybe|perhaps|possibly|not sure|unsure|unclear)\b",
            r"\b(review|information|let me know|wondering|question|asking|curious)\b",
        ],
    },

    # ── Intent ────────────────────────────────────────────────────────────────
    "intent": {
        "Informational": [
            r"\b(what|why|how|when|where|who|which|explain|tell me|definition|meaning)\b",
            r"\b(learn|understand|know|curious|research|study|read|article|guide|tutorial)\b",
            r"\b(history|cause|reason|fact|statistic|data|information|detail|overview)\b",
        ],
        "Transactional": [
            r"\b(buy|purchase|order|checkout|add to cart|payment|pay|price|cost|cheap)\b",
            r"\b(discount|coupon|promo|deal|offer|sale|subscribe|sign up|register)\b",
            r"\b(download|install|get|free|trial|demo|book|reserve|hire|quote)\b",
        ],
        "Navigational": [
            r"\b(login|log in|sign in|account|profile|dashboard|settings|homepage|official)\b",
            r"\b(website|site|app|platform|portal|link|url|go to|visit|open|launch)\b",
            r"\b(find|locate|search for|where is|contact|support|help center|faq)\b",
        ],
        "Conversational": [
            r"\b(hi|hello|hey|thanks|thank you|please|sorry|excuse me|good morning)\b",
            r"\b(how are you|what's up|nice to meet|talk|chat|discuss|opinion|think)\b",
            r"\b(feel|feeling|recommend|suggest|advice|tip|idea|thought|experience)\b",
        ],
    },

    # ── Content Category (Trust & Safety) ─────────────────────────────────────
    "content_category": {
        "Spam": [
            r"\b(click here|limited offer|act now|earn money|make money fast|winner)\b",
            r"\b(free gift|100% free|no risk|guaranteed|prize|lottery|selected)\b",
            r"(https?://\S+){3,}",                      # 3+ URLs in one text
            r"(\$\d+){2,}",                              # multiple price mentions
            r"\b(follow back|follow for follow|f4f|sub4sub|like4like)\b",
        ],
        "Hate Speech": [
            r"\b(hate|kill|destroy|eliminate)\s+(all|every|those)\b",
            r"\b(inferior|superior race|subhuman|vermin|parasite)\b",
            r"\b(go back to|not welcome|don't belong|outsider)\b",
        ],
        "Violence": [
            r"\b(bomb|explode|shoot|stab|kill|murder|attack|threat|assault)\b",
            r"\b(weapon|gun|knife|poison|harm|hurt|injure|wound|torture)\b",
            r"\b(riot|mob|gang|terror|extremis|radicali)\b",
        ],
        "Misinformation": [
            r"\b(fake news|hoax|conspiracy|they don't want you to know|secret cure)\b",
            r"\b(proven by scientists but hidden|government hiding|cover.?up|truth they hide)\b",
            r"\b(miracle cure|100% effective|no side effects|doctors hate)\b",
        ],
        "NSFW": [
            r"\b(nsfw|adult content|18\+|explicit|mature|x.?rated)\b",
        ],
        "Safe": [
            r"\b(hello|thank|welcome|nice|enjoy|share|help|learn|create|build)\b",
            r"\b(news|update|event|product|service|review|feedback|story|article)\b",
        ],
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# RULE ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class RuleEngine:
    """
    Applies keyword/regex rules to text and returns:
        label       — predicted label string
        confidence  — float 0-1 (ratio of matched patterns)
        signals     — dict of {label: match_count}
        method      — always 'rule_based'
    """

    def __init__(self, task: str, rules: dict[str, list[str]] | None = None):
        self.task = task
        if rules:
            self.rules = rules
        elif task in BUILT_IN_RULES:
            self.rules = BUILT_IN_RULES[task]
        else:
            raise ValueError(
                f"Unknown task '{task}'. Built-in tasks: "
                f"{list(BUILT_IN_RULES.keys())}. "
                "Pass custom rules via --config or the 'rules' parameter."
            )
        # Pre-compile all patterns
        self._compiled: dict[str, list[re.Pattern]] = {}
        for label, patterns in self.rules.items():
            self._compiled[label] = [
                re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns
            ]

    def predict(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            return {
                "label":      "Unknown",
                "confidence": 0.0,
                "signals":    {},
                "method":     "rule_based",
            }

        signals: dict[str, int] = {}
        for label, patterns in self._compiled.items():
            signals[label] = sum(1 for p in patterns if p.search(text))

        if not any(v > 0 for v in signals.values()):
            # No signals at all
            default_label = list(self.rules.keys())[-1]   # last = safest default
            return {
                "label":      default_label,
                "confidence": 0.0,
                "signals":    signals,
                "method":     "rule_based",
            }

        winner     = max(signals, key=lambda k: signals[k])
        max_count  = signals[winner]
        total_pats = len(self._compiled[winner])
        confidence = min(max_count / total_pats, 1.0)

        return {
            "label":      winner,
            "confidence": round(confidence, 4),
            "signals":    signals,
            "method":     "rule_based",
        }


# ═════════════════════════════════════════════════════════════════════════════
# ANNOTATION RECORD
# ═════════════════════════════════════════════════════════════════════════════
class AnnotationRecord:
    """Immutable snapshot of one annotation event."""

    __slots__ = (
        "record_id", "source_id", "text", "task",
        "auto_label", "auto_confidence", "auto_signals",
        "final_label", "annotator", "is_human_reviewed",
        "human_override", "annotation_time_sec",
        "created_at", "metadata",
    )

    def __init__(
        self,
        source_id: str,
        text: str,
        task: str,
        auto_result: dict,
        final_label: str | None = None,
        annotator: str = "auto",
        is_human_reviewed: bool = False,
        annotation_time_sec: float = 0.0,
        metadata: dict | None = None,
    ):
        self.record_id           = str(uuid.uuid4())
        self.source_id           = str(source_id)
        self.text                = text
        self.task                = task
        self.auto_label          = auto_result.get("label")
        self.auto_confidence     = auto_result.get("confidence", 0.0)
        self.auto_signals        = auto_result.get("signals", {})
        self.final_label         = final_label or self.auto_label
        self.annotator           = annotator
        self.is_human_reviewed   = is_human_reviewed
        self.human_override      = (
            is_human_reviewed and (final_label != self.auto_label)
        )
        self.annotation_time_sec = round(annotation_time_sec, 3)
        self.created_at          = datetime.now(timezone.utc).isoformat()
        self.metadata            = metadata or {}

    def to_dict(self) -> dict:
        return {
            "record_id":            self.record_id,
            "source_id":            self.source_id,
            "text":                 self.text,
            "task":                 self.task,
            "auto_label":           self.auto_label,
            "auto_confidence":      self.auto_confidence,
            "auto_signals":         self.auto_signals,
            "final_label":          self.final_label,
            "annotator":            self.annotator,
            "is_human_reviewed":    self.is_human_reviewed,
            "human_override":       self.human_override,
            "annotation_time_sec":  self.annotation_time_sec,
            "created_at":           self.created_at,
            "metadata":             self.metadata,
        }


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ═════════════════════════════════════════════════════════════════════════════
class DataLoader:
    """
    Load input data from CSV or JSON.
    Auto-detects the text column from common names.
    """

    COMMON_TEXT_COLS = [
        "text", "content", "review", "comment", "message",
        "description", "title", "post", "tweet", "body",
        "review_text", "review_body", "sentence", "document",
    ]
    COMMON_ID_COLS = ["id", "review_id", "post_id", "comment_id", "item_id"]

    @classmethod
    def load(
        cls,
        path: str | Path,
        text_col: str | None = None,
        id_col:   str | None = None,
        limit:    int | None = None,
    ) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        elif suffix == ".json":
            df = pd.read_json(path, dtype=str)
        elif suffix in (".jsonl", ".ndjson"):
            df = pd.read_json(path, lines=True, dtype=str)
        elif suffix == ".tsv":
            df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        else:
            raise ValueError(f"Unsupported format: {suffix}. Use CSV, JSON, JSONL, or TSV.")

        # ── Find text column ───────────────────────────────────────────────────
        if text_col:
            if text_col not in df.columns:
                raise KeyError(f"Column '{text_col}' not found. Available: {list(df.columns)}")
        else:
            lower_cols = {c.lower(): c for c in df.columns}
            found = next(
                (lower_cols[k] for k in cls.COMMON_TEXT_COLS if k in lower_cols),
                None
            )
            if found is None:
                raise KeyError(
                    f"Cannot auto-detect text column. "
                    f"Columns found: {list(df.columns)}. "
                    f"Use --text-col to specify."
                )
            text_col = found

        # ── Find / create ID column ────────────────────────────────────────────
        if id_col:
            if id_col not in df.columns:
                raise KeyError(f"ID column '{id_col}' not found.")
        else:
            lower_cols = {c.lower(): c for c in df.columns}
            id_col_found = next(
                (lower_cols[k] for k in cls.COMMON_ID_COLS if k in lower_cols),
                None
            )
            if id_col_found:
                id_col = id_col_found
            else:
                df["_id"] = range(len(df))
                id_col = "_id"

        # ── Clean up ───────────────────────────────────────────────────────────
        df["_text"] = df[text_col].astype(str).str.strip()
        df["_id"]   = df[id_col].astype(str)
        df = df[df["_text"].str.len() > 0].reset_index(drop=True)

        if limit:
            df = df.head(limit)

        logger.info(
            f"Loaded {len(df):,} records from '{path.name}' "
            f"(text='{text_col}', id='{id_col}')"
        )
        return df


# ═════════════════════════════════════════════════════════════════════════════
# EXPORTERS
# ═════════════════════════════════════════════════════════════════════════════
class Exporter:
    """Write annotation results to various formats."""

    @staticmethod
    def to_csv(records: list[AnnotationRecord], path: Path) -> Path:
        rows = [r.to_dict() for r in records]
        df   = pd.DataFrame(rows)
        # Flatten signals dict → individual columns
        if "auto_signals" in df.columns:
            signals_df = pd.json_normalize(df["auto_signals"])
            signals_df.columns = [f"signal_{c}" for c in signals_df.columns]
            df = pd.concat([df.drop("auto_signals", axis=1), signals_df], axis=1)
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"Saved CSV  → {path}")
        return path

    @staticmethod
    def to_jsonl(records: list[AnnotationRecord], path: Path) -> Path:
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"Saved JSONL → {path}")
        return path

    @staticmethod
    def to_label_studio(
        records: list[AnnotationRecord],
        path: Path,
        task: str,
    ) -> Path:
        """
        Export Label Studio-compatible JSON for human review.
        Import this file via Label Studio: Projects → Import.
        """
        tasks = []
        for r in records:
            ls_task = {
                "id":   r.source_id,
                "data": {
                    "text":       r.text,
                    "source_id":  r.source_id,
                    "task_type":  task,
                },
                "annotations": [
                    {
                        "id":          r.record_id,
                        "created_at":  r.created_at,
                        "result": [
                            {
                                "from_name": "label",
                                "to_name":   "text",
                                "type":      "choices",
                                "value": {
                                    "choices":    [r.final_label],
                                    "confidence": r.auto_confidence,
                                }
                            }
                        ],
                        "was_cancelled":    False,
                        "ground_truth":     False,
                        "lead_time":        r.annotation_time_sec,
                    }
                ],
                "predictions": [
                    {
                        "model_version": f"rule_engine_v{VERSION}",
                        "score":         r.auto_confidence,
                        "result": [
                            {
                                "from_name": "label",
                                "to_name":   "text",
                                "type":      "choices",
                                "value": {"choices": [r.auto_label]},
                            }
                        ],
                    }
                ],
            }
            tasks.append(ls_task)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved Label Studio JSON → {path}")
        return path

    @staticmethod
    def to_summary_json(records: list[AnnotationRecord], path: Path) -> dict:
        """Write a human-readable summary report."""
        labels       = [r.final_label for r in records]
        auto_labels  = [r.auto_label  for r in records]
        confidences  = [r.auto_confidence for r in records]
        overrides    = [r for r in records if r.human_override]
        reviewed     = [r for r in records if r.is_human_reviewed]

        summary = {
            "generated_at":     datetime.now(timezone.utc).isoformat(),
            "pipeline_version": VERSION,
            "total_records":    len(records),
            "task":             records[0].task if records else None,
            "label_distribution": dict(Counter(labels)),
            "auto_label_distribution": dict(Counter(auto_labels)),
            "confidence_stats": {
                "mean":   round(float(np.mean(confidences)), 4)  if confidences else 0,
                "median": round(float(np.median(confidences)), 4) if confidences else 0,
                "min":    round(float(np.min(confidences)), 4)    if confidences else 0,
                "max":    round(float(np.max(confidences)), 4)    if confidences else 0,
                "pct_zero":  round(sum(1 for c in confidences if c == 0) / len(confidences) * 100, 2) if confidences else 0,
            },
            "human_review": {
                "reviewed_count": len(reviewed),
                "override_count": len(overrides),
                "override_rate_pct": round(len(overrides) / len(reviewed) * 100, 2) if reviewed else 0,
            },
            "quality_flags": {
                "low_confidence_count":  sum(1 for c in confidences if 0 < c < 0.2),
                "zero_signal_count":     sum(1 for c in confidences if c == 0.0),
                "needs_human_review":    sum(1 for c in confidences if c < 0.2),
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved summary → {path}")
        return summary


# ═════════════════════════════════════════════════════════════════════════════
# ANNOTATION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
class AnnotationPipeline:
    """
    Orchestrates the full annotation workflow:
    1. Load data
    2. Auto-label with RuleEngine
    3. (Optional) human review
    4. Export results
    """

    def __init__(
        self,
        task:        str,
        mode:        str = "auto",
        export_fmt:  str = "csv",
        annotator:   str = "auto",
        config_path: str | Path | None = None,
        confidence_threshold: float = 0.15,
        review_low_confidence: bool = True,
    ):
        self.task                  = task
        self.mode                  = mode
        self.export_fmt            = export_fmt
        self.annotator             = annotator
        self.confidence_threshold  = confidence_threshold
        self.review_low_confidence = review_low_confidence
        self.records: list[AnnotationRecord] = []
        self.stats   = defaultdict(int)

        # Load rules
        custom_rules = None
        if config_path:
            custom_rules = self._load_config(Path(config_path))
        self.engine = RuleEngine(task, rules=custom_rules)

        logger.info(
            f"Pipeline initialised  task={task}  mode={mode}  "
            f"export={export_fmt}  threshold={confidence_threshold}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _load_config(path: Path) -> dict[str, list[str]]:
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        # Expected format: {"LabelA": ["pattern1", "pattern2"], ...}
        # Skip keys starting with _ (comments/metadata)
        if not isinstance(cfg, dict):
            raise ValueError("Config must be a JSON object: {label: [patterns]}")
        return {k: v for k, v in cfg.items() if not k.startswith("_")}

    # ── Label display ─────────────────────────────────────────────────────────
    @staticmethod
    def _label_colours(label: str) -> str:
        colours = {
            "Positive":     "\033[92m",   # green
            "Negative":     "\033[91m",   # red
            "Neutral":      "\033[93m",   # yellow
            "Spam":         "\033[91m",
            "Hate Speech":  "\033[91m",
            "Violence":     "\033[91m",
            "Misinformation":"\033[91m",
            "NSFW":         "\033[95m",   # magenta
            "Safe":         "\033[92m",
            "Informational":"\033[94m",   # blue
            "Transactional":"\033[96m",   # cyan
            "Navigational": "\033[93m",
            "Conversational":"\033[37m",  # white
            "Unknown":      "\033[90m",   # grey
        }
        reset = "\033[0m"
        return colours.get(label, "") + label + reset

    # ── Interactive session ───────────────────────────────────────────────────
    def _interactive_review(
        self,
        idx:         int,
        total:       int,
        text:        str,
        auto_result: dict,
    ) -> tuple[str, float]:
        """Present one item for human review. Returns (final_label, time_sec)."""
        valid_labels = list(self.engine.rules.keys())
        label_map    = {str(i+1): lbl for i, lbl in enumerate(valid_labels)}

        print("\n" + "─" * 70)
        print(f"[{idx+1}/{total}]  ID: {idx}")
        print(f"\nTEXT:\n  {text[:300]}{'...' if len(text) > 300 else ''}")
        print(f"\nAUTO LABEL  : {self._label_colours(auto_result['label'])}")
        print(f"CONFIDENCE  : {auto_result['confidence']:.1%}")
        if auto_result["signals"]:
            sig_str = "  |  ".join(
                f"{k}: {v}" for k, v in auto_result["signals"].items() if v > 0
            )
            print(f"SIGNALS     : {sig_str}")
        print()
        for num, lbl in label_map.items():
            marker = " ← auto" if lbl == auto_result["label"] else ""
            print(f"  [{num}] {lbl}{marker}")
        print("  [Enter] Accept auto-label")
        print("  [s]     Skip this item")
        print("  [q]     Quit and save progress")
        print()

        t_start = time.monotonic()
        while True:
            try:
                choice = input("  Your choice: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n\nInterrupted. Saving progress …")
                choice = "q"

            if choice == "":
                return auto_result["label"], time.monotonic() - t_start
            elif choice == "s":
                return auto_result["label"], time.monotonic() - t_start  # skip = keep auto
            elif choice == "q":
                print("Saving and exiting …")
                raise StopIteration
            elif choice in label_map:
                selected = label_map[choice]
                print(f"  ✔ Labelled as: {self._label_colours(selected)}")
                return selected, time.monotonic() - t_start
            else:
                print(f"  ✘ Invalid. Enter 1–{len(valid_labels)}, Enter, 's', or 'q'.")

    # ── Main run ──────────────────────────────────────────────────────────────
    def run(
        self,
        df:           pd.DataFrame,
        output_stem:  str,
    ) -> list[AnnotationRecord]:
        """
        Process a DataFrame and produce AnnotationRecord objects.
        df must have columns: _text, _id
        """
        total = len(df)
        logger.info(f"Starting annotation: {total:,} records")

        for idx, (_, row) in enumerate(df.iterrows()):
            text      = str(row["_text"])
            source_id = str(row["_id"])

            auto_result = self.engine.predict(text)
            self.stats["total"] += 1

            if self.mode == "auto":
                # ── Fully automatic ──────────────────────────────────────────
                rec = AnnotationRecord(
                    source_id=source_id,
                    text=text,
                    task=self.task,
                    auto_result=auto_result,
                    annotator="auto",
                    is_human_reviewed=False,
                )
                self.stats["auto"] += 1

            elif self.mode == "interactive":
                # ── Human review per item ────────────────────────────────────
                should_review = (
                    auto_result["confidence"] < self.confidence_threshold
                    or self.mode == "interactive"
                )
                if not should_review:
                    rec = AnnotationRecord(
                        source_id=source_id,
                        text=text,
                        task=self.task,
                        auto_result=auto_result,
                        annotator="auto",
                        is_human_reviewed=False,
                    )
                    self.stats["auto"] += 1
                else:
                    try:
                        final_label, t = self._interactive_review(
                            idx, total, text, auto_result
                        )
                    except StopIteration:
                        # User quit — save remaining as auto
                        for _, remaining_row in df.iloc[idx:].iterrows():
                            r_auto = self.engine.predict(str(remaining_row["_text"]))
                            self.records.append(AnnotationRecord(
                                source_id=str(remaining_row["_id"]),
                                text=str(remaining_row["_text"]),
                                task=self.task,
                                auto_result=r_auto,
                                annotator="auto",
                                is_human_reviewed=False,
                            ))
                        break

                    is_human = True
                    self.stats["human"] += 1
                    if final_label != auto_result["label"]:
                        self.stats["override"] += 1
                    rec = AnnotationRecord(
                        source_id=source_id,
                        text=text,
                        task=self.task,
                        auto_result=auto_result,
                        final_label=final_label,
                        annotator=self.annotator,
                        is_human_reviewed=is_human,
                        annotation_time_sec=t,
                    )

            else:  # batch — same as auto
                rec = AnnotationRecord(
                    source_id=source_id,
                    text=text,
                    task=self.task,
                    auto_result=auto_result,
                    annotator="auto",
                    is_human_reviewed=False,
                )
                self.stats["auto"] += 1

            self.records.append(rec)

            # Progress bar (every 100 or last item)
            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                pct = (idx + 1) / total * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(
                    f"\r  [{bar}] {pct:5.1f}%  {idx+1:,}/{total:,}",
                    end="",
                    flush=True,
                )
        print()  # newline after progress bar

        # ── Export ─────────────────────────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem      = f"{output_stem}_{self.task}_{timestamp}"

        self._export(stem)
        return self.records

    # ── Export dispatcher ─────────────────────────────────────────────────────
    def _export(self, stem: str):
        out_csv    = DATA_ANNOTATED / f"{stem}.csv"
        out_jsonl  = DATA_ANNOTATED / f"{stem}.jsonl"
        out_ls     = DATA_EXPORTS   / f"{stem}_label_studio.json"
        out_summary= DATA_EXPORTS   / f"{stem}_summary.json"

        Exporter.to_csv(self.records, out_csv)
        Exporter.to_jsonl(self.records, out_jsonl)

        if self.export_fmt in ("ls", "label_studio", "all"):
            Exporter.to_label_studio(self.records, out_ls, self.task)

        summary = Exporter.to_summary_json(self.records, out_summary)
        self._print_summary(summary)

    # ── Console summary ───────────────────────────────────────────────────────
    @staticmethod
    def _print_summary(s: dict):
        print("\n" + "═" * 55)
        print("  ANNOTATION SUMMARY")
        print("═" * 55)
        print(f"  Total records   : {s['total_records']:,}")
        print(f"  Task            : {s['task']}")
        print()
        print("  Label distribution:")
        for lbl, cnt in sorted(s["label_distribution"].items(), key=lambda x: -x[1]):
            pct = cnt / s["total_records"] * 100
            bar = "█" * int(pct / 3)
            print(f"    {lbl:<22} {cnt:>5,}  {pct:5.1f}%  {bar}")
        print()
        cs = s["confidence_stats"]
        print(f"  Confidence: mean={cs['mean']:.1%}  median={cs['median']:.1%}  "
              f"min={cs['min']:.1%}  max={cs['max']:.1%}")
        qf = s["quality_flags"]
        print(f"  Low confidence  : {qf['low_confidence_count']:,} items (<20%)")
        print(f"  Zero signals    : {qf['zero_signal_count']:,} items")
        hr = s["human_review"]
        if hr["reviewed_count"]:
            print(f"  Human reviewed  : {hr['reviewed_count']:,}")
            print(f"  Override rate   : {hr['override_rate_pct']:.1f}%")
        print("═" * 55)


# ═════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA GENERATOR  (for demo / testing)
# ═════════════════════════════════════════════════════════════════════════════
SAMPLE_TEXTS = {
    "sentiment": [
        ("1", "This product is absolutely amazing! Best purchase I've made this year. 😍"),
        ("2", "Terrible quality. Broke after 2 days. Complete waste of money. 😠"),
        ("3", "It's okay. Nothing special, but does the job."),
        ("4", "Fantastic customer service! They resolved my issue in minutes."),
        ("5", "Never buying from this store again. Shipment was 3 weeks late!"),
        ("6", "Mediocre product. Average at best."),
        ("7", "Exceeded all my expectations. Outstanding build quality!"),
        ("8", "Disappointed with the packaging but the product itself is decent."),
        ("9", "Great value for money. Highly recommend to anyone looking for a budget option."),
        ("10","The item arrived damaged. Very frustrated with this experience."),
        ("11","Perfect for my needs. Simple, clean design and works perfectly."),
        ("12","Not sure about this one. It has both good and bad aspects."),
        ("13","Absolutely love it! Would give 6 stars if I could! 💯"),
        ("14","Poor customer support. Took 3 weeks to get a response."),
        ("15","Solid product. Nothing extraordinary but reliable and consistent."),
    ],
    "intent": [
        ("1", "What is the capital city of Brazil?"),
        ("2", "Buy cheap laptops online free shipping"),
        ("3", "Log in to my account dashboard"),
        ("4", "Hey! Can you recommend a good restaurant near me?"),
        ("5", "How does machine learning work?"),
        ("6", "Order iPhone 15 Pro Max with discount code"),
        ("7", "Sign up for the free trial today"),
        ("8", "Where is the official help center website?"),
        ("9", "Why is the sky blue? Explain the science."),
        ("10","Download the mobile app for free"),
        ("11","Hello, how are you today?"),
        ("12","Best Python tutorial for beginners 2024"),
        ("13","Reset my password account settings"),
        ("14","Limited offer: get 50% off all products now"),
        ("15","What are the causes of climate change?"),
    ],
    "content_category": [
        ("1", "EARN $5000 A WEEK FROM HOME! Click here NOW! Limited offer! 🤑"),
        ("2", "Check out this amazing recipe for chocolate chip cookies!"),
        ("3", "Scientists HATE this one trick that cures all diseases!"),
        ("4", "Had a great time at the local farmers market today 🌺"),
        ("5", "New update: our app now supports dark mode. Download v2.1 today."),
        ("6", "FREE GIFT for the first 100 subscribers! Act now! No risk!"),
        ("7", "Reminder: community meeting tonight at 7pm in the main hall."),
        ("8", "The government is hiding the real data about this virus!"),
        ("9", "Beautiful sunset at the beach today 🌅 Nature is amazing."),
        ("10","WIN WIN WIN! You have been selected for our lottery prize!"),
        ("11","Workshop registration open: Python for Data Science - free seats"),
        ("12","Doctors hate him for discovering this miracle cure with no side effects"),
        ("13","Team lunch today was great! Thanks everyone for coming 🎉"),
        ("14","ADULT CONTENT 18+ — mature themes ahead"),
        ("15","Our quarterly results are now published on the investor relations page."),
    ],
}


def generate_sample_data(task: str, n: int = 100) -> Path:
    """
    Generate a CSV of sample texts for the given task.
    Uses the 15 seed texts above, cycling to reach n rows.
    """
    seeds = SAMPLE_TEXTS.get(task, SAMPLE_TEXTS["sentiment"])
    rows  = []
    for i in range(n):
        seed_id, seed_text = seeds[i % len(seeds)]
        # Slight perturbation to make each row unique
        suffix = f" (sample #{i+1})" if i >= len(seeds) else ""
        rows.append({"id": f"{task}_{i+1:04d}", "text": seed_text + suffix})

    path = DATA_RAW / f"sample_{task}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"Generated sample data → {path}  ({n} rows)")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="annotator",
        description=(
            "Semi-Automated Data Annotation Pipeline  "
            f"v{VERSION}\n"
            "Author: Qowiyu Yusrizal  |  github.com/Exynos93"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  # Auto-label 500 rows from a CSV (sentiment):
  python annotator.py --input data/raw/reviews.csv --task sentiment --mode auto

  # Interactive review (human verifies each prediction):
  python annotator.py --input data/raw/posts.csv --task content_category --mode interactive

  # Batch with Label Studio export:
  python annotator.py --input data/raw/queries.csv --task intent --mode batch --export ls

  # Use custom rules from a JSON config:
  python annotator.py --input data/raw/chats.csv --task custom --config config/my_rules.json

  # Generate sample data and run demo:
  python annotator.py --demo --task sentiment
        """,
    )
    p.add_argument("--input",    "-i", help="Path to input CSV/JSON/JSONL/TSV")
    p.add_argument("--task",     "-t",
                   choices=["sentiment", "intent", "content_category", "custom"],
                   default="sentiment",
                   help="Annotation task (default: sentiment)")
    p.add_argument("--mode",     "-m",
                   choices=["auto", "interactive", "batch"],
                   default="auto",
                   help="Annotation mode (default: auto)")
    p.add_argument("--export",   "-e",
                   choices=["csv", "jsonl", "ls", "all"],
                   default="csv",
                   help="Output format (default: csv). ls = Label Studio JSON")
    p.add_argument("--text-col", help="Name of the text column (auto-detected if omitted)")
    p.add_argument("--id-col",   help="Name of the ID column   (auto-detected if omitted)")
    p.add_argument("--limit",    "-n", type=int, help="Max rows to process")
    p.add_argument("--config",   "-c", help="Path to custom JSON rules config")
    p.add_argument("--annotator",     default="Qowiyu Yusrizal",
                   help="Annotator name stored in records (default: your name)")
    p.add_argument("--threshold", type=float, default=0.15,
                   help="Confidence threshold for human review flag (default: 0.15)")
    p.add_argument("--demo",     action="store_true",
                   help="Generate and run a demo with sample data")
    p.add_argument("--version",  action="version", version=f"%(prog)s {VERSION}")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════╗
║   DATA ANNOTATION PIPELINE  v{VERSION}                   ║
║   Author: Qowiyu Yusrizal  |  github.com/Exynos93   ║
╚══════════════════════════════════════════════════════╝
""")

    # ── Demo mode ────────────────────────────────────────────────────────────
    if args.demo:
        print(f"[DEMO] Generating 50 sample rows for task: {args.task}")
        input_path = generate_sample_data(args.task, n=50)
    elif args.input:
        input_path = Path(args.input)
    else:
        parser.error("Provide --input <file> or use --demo")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = DataLoader.load(
        input_path,
        text_col=args.text_col,
        id_col=args.id_col,
        limit=args.limit,
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    pipeline = AnnotationPipeline(
        task=args.task,
        mode=args.mode,
        export_fmt=args.export,
        annotator=args.annotator,
        config_path=args.config,
        confidence_threshold=args.threshold,
    )

    output_stem = input_path.stem
    pipeline.run(df, output_stem)

    print(f"\nDone! Results saved to:\n"
          f"  {DATA_ANNOTATED}/\n"
          f"  {DATA_EXPORTS}/\n")


if __name__ == "__main__":
    main()
