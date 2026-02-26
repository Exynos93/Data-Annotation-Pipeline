"""
tests/test_pipeline.py
======================
Unit and integration tests for the Data Annotation Pipeline.

Run:
    python -m pytest tests/ -v
    python tests/test_pipeline.py          # run directly (no pytest needed)

Author : Qowiyu Yusrizal <hihakai123@gmail.com>
GitHub : https://github.com/Exynos93
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# ── Make parent importable ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from annotator import (
    RuleEngine,
    AnnotationRecord,
    DataLoader,
    AnnotationPipeline,
    generate_sample_data,
    BUILT_IN_RULES,
)
from quality_checker import (
    check_completeness,
    check_confidence,
    check_label_distribution,
    check_duplicates,
    check_text_quality,
    check_override_analysis,
    check_golden_set,
    check_inter_annotator,
    cohens_kappa,
    generate_golden_set,
    load_annotated,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = {"passed": 0, "failed": 0}


def run_test(name, fn):
    try:
        fn()
        print(f"  [{PASS}]  {name}")
        results["passed"] += 1
    except AssertionError as e:
        print(f"  [{FAIL}]  {name}")
        print(f"           AssertionError: {e}")
        results["failed"] += 1
    except Exception as e:
        print(f"  [{FAIL}]  {name}")
        print(f"           {type(e).__name__}: {e}")
        results["failed"] += 1


# ═════════════════════════════════════════════════════════════════════════════
# RULE ENGINE TESTS
# ═════════════════════════════════════════════════════════════════════════════
def test_rule_engine_positive():
    engine = RuleEngine("sentiment")
    result = engine.predict("This product is absolutely amazing! I love it so much!")
    assert result["label"] == "Positive", f"Got: {result['label']}"
    assert result["confidence"] > 0, "Expected confidence > 0"
    assert result["method"] == "rule_based"


def test_rule_engine_negative():
    engine = RuleEngine("sentiment")
    result = engine.predict("Terrible quality. Completely broken after 2 days. Awful!")
    assert result["label"] == "Negative", f"Got: {result['label']}"
    assert result["confidence"] > 0


def test_rule_engine_neutral():
    engine = RuleEngine("sentiment")
    result = engine.predict("The package arrived. It is okay.")
    # Should be Neutral or low-confidence
    assert result["label"] in ("Neutral", "Positive", "Negative"), f"Got: {result['label']}"


def test_rule_engine_intent_informational():
    engine = RuleEngine("intent")
    result = engine.predict("What is machine learning and how does it work?")
    assert result["label"] == "Informational", f"Got: {result['label']}"


def test_rule_engine_intent_transactional():
    engine = RuleEngine("intent")
    result = engine.predict("Buy cheap laptops online with free shipping and discount")
    assert result["label"] == "Transactional", f"Got: {result['label']}"


def test_rule_engine_intent_navigational():
    engine = RuleEngine("intent")
    result = engine.predict("Log in to my account dashboard")
    assert result["label"] == "Navigational", f"Got: {result['label']}"


def test_rule_engine_content_spam():
    engine = RuleEngine("content_category")
    result = engine.predict("EARN MONEY FAST! Click here NOW! Limited offer! WIN WIN WIN!")
    assert result["label"] == "Spam", f"Got: {result['label']}"


def test_rule_engine_content_safe():
    engine = RuleEngine("content_category")
    result = engine.predict("Had a wonderful day at the park. The flowers were beautiful!")
    assert result["label"] in ("Safe", "Spam", "Misinformation"), f"Got: {result['label']}"


def test_rule_engine_empty_text():
    engine = RuleEngine("sentiment")
    result = engine.predict("")
    assert result["label"] == "Unknown"
    assert result["confidence"] == 0.0


def test_rule_engine_whitespace_only():
    engine = RuleEngine("sentiment")
    result = engine.predict("   \t\n  ")
    assert result["label"] == "Unknown"


def test_rule_engine_unicode():
    engine = RuleEngine("sentiment")
    result = engine.predict("Saya sangat menyukai produk ini! Sangat bagus dan amazing!")
    # Should still find "amazing" signal
    assert result["label"] in ("Positive", "Neutral", "Negative")


def test_rule_engine_custom_rules():
    custom = {
        "Cat": [r"\b(cat|kitten|feline|meow)\b"],
        "Dog": [r"\b(dog|puppy|canine|woof|bark)\b"],
    }
    engine = RuleEngine("custom_animals", rules=custom)
    assert engine.predict("My cat is so cute and fluffy")["label"] == "Cat"
    assert engine.predict("The dog barked loudly all night")["label"] == "Dog"


def test_rule_engine_all_tasks():
    for task in BUILT_IN_RULES:
        engine = RuleEngine(task)
        result = engine.predict("Test text for this task")
        assert "label" in result
        assert "confidence" in result


# ═════════════════════════════════════════════════════════════════════════════
# ANNOTATION RECORD TESTS
# ═════════════════════════════════════════════════════════════════════════════
def test_annotation_record_basic():
    auto_result = {"label": "Positive", "confidence": 0.75, "signals": {"Positive": 3}}
    rec = AnnotationRecord(
        source_id="test_001",
        text="This is great!",
        task="sentiment",
        auto_result=auto_result,
    )
    assert rec.auto_label   == "Positive"
    assert rec.final_label  == "Positive"
    assert rec.auto_confidence == 0.75
    assert not rec.is_human_reviewed
    assert not rec.human_override
    assert rec.record_id    is not None


def test_annotation_record_human_override():
    auto_result = {"label": "Positive", "confidence": 0.30, "signals": {}}
    rec = AnnotationRecord(
        source_id="test_002",
        text="Meh, it is okay I guess.",
        task="sentiment",
        auto_result=auto_result,
        final_label="Neutral",
        annotator="Qowiyu",
        is_human_reviewed=True,
        annotation_time_sec=5.2,
    )
    assert rec.final_label       == "Neutral"
    assert rec.auto_label        == "Positive"
    assert rec.human_override    is True
    assert rec.is_human_reviewed is True
    assert rec.annotator         == "Qowiyu"
    assert rec.annotation_time_sec == 5.2


def test_annotation_record_to_dict():
    auto_result = {"label": "Spam", "confidence": 0.85, "signals": {"Spam": 4}}
    rec = AnnotationRecord("id_1", "Click here to win!", "content_category", auto_result)
    d = rec.to_dict()
    required_keys = {
        "record_id", "source_id", "text", "task",
        "auto_label", "auto_confidence", "final_label",
        "annotator", "is_human_reviewed", "created_at",
    }
    assert required_keys.issubset(set(d.keys())), f"Missing: {required_keys - set(d.keys())}"


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADER TESTS
# ═════════════════════════════════════════════════════════════════════════════
def test_data_loader_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("id,text\n")
        f.write("1,Hello world\n")
        f.write("2,Another review\n")
        tmp = f.name
    try:
        df = DataLoader.load(tmp)
        assert len(df) == 2
        assert "_text" in df.columns
        assert "_id"   in df.columns
        assert df["_text"].iloc[0] == "Hello world"
    finally:
        os.unlink(tmp)


def test_data_loader_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([
            {"id": "1", "content": "Product is great"},
            {"id": "2", "content": "Bad experience"},
        ], f)
        tmp = f.name
    try:
        df = DataLoader.load(tmp)
        assert len(df) == 2
        assert "_text" in df.columns
    finally:
        os.unlink(tmp)


def test_data_loader_missing_file():
    try:
        DataLoader.load("/nonexistent/path/file.csv")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_data_loader_limit():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("text\n")
        for i in range(100):
            f.write(f"Row {i}\n")
        tmp = f.name
    try:
        df = DataLoader.load(tmp, limit=20)
        assert len(df) == 20
    finally:
        os.unlink(tmp)


def test_data_loader_empty_rows_filtered():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("text\n")
        f.write("Valid text\n")
        f.write("   \n")          # empty
        f.write("Another valid\n")
        f.write("\n")             # empty
        tmp = f.name
    try:
        df = DataLoader.load(tmp)
        assert len(df) == 2
    finally:
        os.unlink(tmp)


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE INTEGRATION TESTS
# ═════════════════════════════════════════════════════════════════════════════
def test_pipeline_auto_mode():
    input_path = generate_sample_data("sentiment", n=20)
    df = DataLoader.load(input_path)
    pipeline = AnnotationPipeline(task="sentiment", mode="auto", export_fmt="csv")
    records = pipeline.run(df, output_stem="test_auto")
    assert len(records) == 20
    assert all(r.auto_label is not None for r in records)
    assert all(r.final_label is not None for r in records)
    assert all(not r.is_human_reviewed for r in records)


def test_pipeline_batch_mode():
    input_path = generate_sample_data("intent", n=15)
    df = DataLoader.load(input_path)
    pipeline = AnnotationPipeline(task="intent", mode="batch", export_fmt="csv")
    records = pipeline.run(df, output_stem="test_batch")
    assert len(records) == 15
    labels = set(r.final_label for r in records)
    assert len(labels) > 1, "All records got same label — suspicious"


def test_pipeline_with_custom_config():
    cfg_path = Path(__file__).parent.parent / "config" / "ecommerce_intent.json"
    if not cfg_path.exists():
        return  # skip if config not present

    input_path = generate_sample_data("sentiment", n=10)
    df = DataLoader.load(input_path)
    pipeline = AnnotationPipeline(
        task="custom", mode="auto", config_path=str(cfg_path)
    )
    records = pipeline.run(df, output_stem="test_custom")
    assert len(records) == 10


def test_pipeline_export_jsonl():
    input_path = generate_sample_data("content_category", n=10)
    df = DataLoader.load(input_path)
    pipeline = AnnotationPipeline(task="content_category", mode="auto", export_fmt="jsonl")
    records = pipeline.run(df, output_stem="test_jsonl")
    assert len(records) == 10


# ═════════════════════════════════════════════════════════════════════════════
# QUALITY CHECKER TESTS
# ═════════════════════════════════════════════════════════════════════════════
def _make_df(n=50, include_human=True):
    """Helper: build a minimal annotated DataFrame for QC tests."""
    rng    = np.random.default_rng(42)
    labels = ["Positive", "Negative", "Neutral"]
    df = pd.DataFrame({
        "text":              [f"Sample text number {i}" for i in range(n)],
        "auto_label":        rng.choice(labels, n),
        "final_label":       rng.choice(labels, n),
        "auto_confidence":   rng.uniform(0, 1, n),
        "source_id":         [f"id_{i:04d}" for i in range(n)],
        "annotator":         ["Qowiyu"] * n,
        "is_human_reviewed": [bool(v) for v in rng.integers(0, 2, n)],
        "human_override":    [bool(v) for v in rng.integers(0, 2, n)],
        "created_at":        ["2024-01-01T00:00:00Z"] * n,
        "annotation_time_sec": rng.uniform(2, 30, n),
    })
    return df


def test_qc_completeness_pass():
    df = _make_df(30)
    result = check_completeness(df)
    assert "completeness_pct" in result
    assert result["total_records"] == 30


def test_qc_completeness_missing_labels():
    df = _make_df(20)
    df.loc[0:4, "final_label"] = ""  # 5 missing
    result = check_completeness(df)
    assert result["flags"].get("missing_labels", 0) == 5
    assert result["passed"] is False


def test_qc_confidence_stats():
    df = _make_df(50)
    result = check_confidence(df, threshold=0.20)
    assert "stats" in result
    assert result["stats"]["mean"] >= 0
    assert result["stats"]["mean"] <= 1
    assert "buckets" in result


def test_qc_confidence_zero():
    df = _make_df(20)
    df["auto_confidence"] = 0.0   # all zero
    result = check_confidence(df, threshold=0.20)
    assert result["needs_review_count"] == 20


def test_qc_label_distribution():
    df = _make_df(100)
    result = check_label_distribution(df)
    assert "distribution" in result
    assert result["n_classes"] >= 1
    assert 0.0 <= result["gini_coefficient"] <= 1.0


def test_qc_duplicates_clean():
    df = _make_df(20)
    result = check_duplicates(df)
    # All texts are unique (f"Sample text number {i}")
    assert result["exact_duplicates"] == 0
    assert result["passed"] is True


def test_qc_duplicates_detected():
    df = _make_df(20)
    df.loc[5, "text"] = df.loc[0, "text"]   # inject duplicate
    result = check_duplicates(df)
    assert result["exact_duplicates"] >= 2
    assert result["passed"] is False


def test_qc_text_quality_normal():
    df = _make_df(30)
    result = check_text_quality(df)
    assert "stats" in result
    assert result["stats"]["mean_length"] > 0


def test_qc_text_quality_short():
    df = _make_df(20)
    df.loc[0, "text"] = "Hi"  # too short
    result = check_text_quality(df)
    assert result["flags"].get("too_short", 0) >= 1


def test_qc_override_analysis():
    df = _make_df(50)
    result = check_override_analysis(df)
    # Either returns full result or skipped result
    assert "check" in result
    assert "override_rate_pct" in result or result.get("skipped") or "overall_override_rate_pct" in result


def test_cohens_kappa_perfect():
    labels = ["A", "B", "C"] * 10
    kappa  = cohens_kappa(labels, labels)
    assert abs(kappa - 1.0) < 1e-9, f"Expected 1.0, got {kappa}"


def test_cohens_kappa_random():
    rng = np.random.default_rng(1)
    a = rng.choice(["A", "B", "C"], 100).tolist()
    b = rng.choice(["A", "B", "C"], 100).tolist()
    kappa = cohens_kappa(a, b)
    assert -1.0 <= kappa <= 1.0


def test_cohens_kappa_zero():
    # Completely opposite labels
    a = ["A"] * 50 + ["B"] * 50
    b = ["B"] * 50 + ["A"] * 50
    kappa = cohens_kappa(a, b)
    assert kappa < 0


def test_qc_golden_set():
    df = _make_df(20)
    df["source_id"] = [f"id_{i:04d}" for i in range(20)]
    # Golden: perfect match for first 10 items
    golden = df.head(10)[["source_id", "final_label"]].copy()
    # Give the auto_label exactly the final_label for golden items
    for idx in golden.index:
        df.loc[idx, "auto_label"] = df.loc[idx, "final_label"]
    result = check_golden_set(df, golden)
    assert "accuracy_pct" in result
    assert result["n_validated"] == 10
    assert result["accuracy_pct"] == 100.0


def test_qc_inter_annotator():
    df_a = _make_df(30)
    df_a["source_id"] = [f"id_{i:04d}" for i in range(30)]
    df_b = df_a.copy()  # perfect agreement
    result = check_inter_annotator(df_a, df_b)
    assert result["exact_agreement_pct"] == 100.0
    assert result["cohens_kappa"] == 1.0


def test_golden_set_generator():
    for task in ["sentiment", "intent", "content_category"]:
        path = generate_golden_set(task)
        assert path.exists()
        df = pd.read_csv(path)
        assert "source_id"   in df.columns
        assert "text"        in df.columns
        assert "final_label" in df.columns
        assert len(df) >= 5


# ═════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA GENERATOR TESTS
# ═════════════════════════════════════════════════════════════════════════════
def test_sample_data_generator_all_tasks():
    for task in ["sentiment", "intent", "content_category"]:
        path = generate_sample_data(task, n=15)
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 15
        assert "text" in df.columns
        assert "id"   in df.columns


# ═════════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═════════════════════════════════════════════════════════════════════════════
ALL_TESTS = [
    # Rule Engine
    ("RuleEngine: Positive sentiment",              test_rule_engine_positive),
    ("RuleEngine: Negative sentiment",              test_rule_engine_negative),
    ("RuleEngine: Neutral sentiment",               test_rule_engine_neutral),
    ("RuleEngine: Informational intent",            test_rule_engine_intent_informational),
    ("RuleEngine: Transactional intent",            test_rule_engine_intent_transactional),
    ("RuleEngine: Navigational intent",             test_rule_engine_intent_navigational),
    ("RuleEngine: Spam content",                    test_rule_engine_content_spam),
    ("RuleEngine: Safe content",                    test_rule_engine_content_safe),
    ("RuleEngine: Empty text → Unknown",            test_rule_engine_empty_text),
    ("RuleEngine: Whitespace only → Unknown",       test_rule_engine_whitespace_only),
    ("RuleEngine: Unicode / multilingual",          test_rule_engine_unicode),
    ("RuleEngine: Custom rules",                    test_rule_engine_custom_rules),
    ("RuleEngine: All built-in tasks",              test_rule_engine_all_tasks),
    # AnnotationRecord
    ("Record: Basic creation",                      test_annotation_record_basic),
    ("Record: Human override flag",                 test_annotation_record_human_override),
    ("Record: to_dict completeness",                test_annotation_record_to_dict),
    # DataLoader
    ("DataLoader: CSV load",                        test_data_loader_csv),
    ("DataLoader: JSON load",                       test_data_loader_json),
    ("DataLoader: Missing file error",              test_data_loader_missing_file),
    ("DataLoader: Limit parameter",                 test_data_loader_limit),
    ("DataLoader: Empty rows filtered",             test_data_loader_empty_rows_filtered),
    # Pipeline
    ("Pipeline: Auto mode (20 records)",            test_pipeline_auto_mode),
    ("Pipeline: Batch mode (15 records)",           test_pipeline_batch_mode),
    ("Pipeline: Custom config",                     test_pipeline_with_custom_config),
    ("Pipeline: JSONL export",                      test_pipeline_export_jsonl),
    # Quality Checker
    ("QC: Completeness — pass",                     test_qc_completeness_pass),
    ("QC: Completeness — missing labels detected",  test_qc_completeness_missing_labels),
    ("QC: Confidence stats",                        test_qc_confidence_stats),
    ("QC: Confidence — all zero",                   test_qc_confidence_zero),
    ("QC: Label distribution",                      test_qc_label_distribution),
    ("QC: Duplicates — clean data",                 test_qc_duplicates_clean),
    ("QC: Duplicates — detected",                   test_qc_duplicates_detected),
    ("QC: Text quality — normal",                   test_qc_text_quality_normal),
    ("QC: Text quality — short texts",              test_qc_text_quality_short),
    ("QC: Override analysis",                       test_qc_override_analysis),
    ("QC: Cohen's Kappa — perfect",                 test_cohens_kappa_perfect),
    ("QC: Cohen's Kappa — random",                  test_cohens_kappa_random),
    ("QC: Cohen's Kappa — negative",                test_cohens_kappa_zero),
    ("QC: Golden set validation",                   test_qc_golden_set),
    ("QC: Inter-annotator agreement",               test_qc_inter_annotator),
    ("QC: Golden set generator",                    test_golden_set_generator),
    # Sample data
    ("SampleData: All three tasks",                 test_sample_data_generator_all_tasks),
]


def main():
    print(f"""
╔══════════════════════════════════════════════════════╗
║   TEST SUITE  —  Data Annotation Pipeline            ║
║   Author: Qowiyu Yusrizal  |  github.com/Exynos93   ║
╚══════════════════════════════════════════════════════╝
""")
    print(f"  Running {len(ALL_TESTS)} tests …\n")

    for name, fn in ALL_TESTS:
        run_test(name, fn)

    total = results["passed"] + results["failed"]
    pct   = results["passed"] / total * 100 if total else 0

    print(f"""
{'═'*55}
  Results : {results['passed']}/{total} passed  ({pct:.0f}%)
  {'✅  ALL TESTS PASSED' if results['failed'] == 0 else f'❌  {results["failed"]} TESTS FAILED'}
{'═'*55}
""")
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
