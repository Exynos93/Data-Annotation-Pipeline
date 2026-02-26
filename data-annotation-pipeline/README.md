# 🏷️ Automated Data Annotation Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-42%2F42%20passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Label Studio](https://img.shields.io/badge/Label%20Studio-Compatible-orange)](https://labelstud.io)

> **Production-grade semi-automated pipeline for labeling text data** — sentiment, intent, and content moderation tasks. Includes rule-based auto-labeling, interactive human review, quality assurance, and Label Studio export.

**Author:** Qowiyu Yusrizal · [GitHub](https://github.com/Exynos93) · [hihakai123@gmail.com](mailto:hihakai123@gmail.com)  
**Degree:** B.Sc. Computer Science · Universitas Indonesia · 2024

---

## 🎯 Why This Exists

Companies like **TikTok, Appen, Telus Digital, and Lionbridge** hire Data Annotators and AI Data Analysts to do exactly what this pipeline automates:

- Label thousands of text samples for sentiment, intent, and content safety
- Ensure consistent quality across annotators
- Export data in formats ready for ML training (CSV, JSONL, Label Studio)

This project demonstrates you can build the **infrastructure behind annotation**, not just perform it manually.

---

## ✨ Features

### 🤖 annotator.py — Annotation Engine
| Feature | Detail |
|---------|--------|
| **Rule-based auto-labeling** | Regex keyword engine with confidence scoring |
| **3 built-in tasks** | Sentiment · Intent · Content Category (Trust & Safety) |
| **Custom rule configs** | Load your own JSON rule files for any domain |
| **3 annotation modes** | `auto` · `interactive` · `batch` |
| **Interactive review UI** | Terminal UI with colour-coded labels and confidence |
| **Smart confidence** | Each prediction scored 0–100%, low-confidence flagged |
| **4 export formats** | CSV · JSONL · Label Studio JSON · Summary JSON |
| **Audit trail** | Every record stores annotator, timestamp, override flag |
| **Multilingual** | Works on any language (regex patterns adapt) |

### 🔍 quality_checker.py — QA Module
| Check | What It Does |
|-------|-------------|
| Completeness | Missing labels, empty texts, null fields |
| Confidence Analysis | Distribution stats, low-confidence count, buckets |
| Label Distribution | Class imbalance detection, Gini coefficient |
| Duplicate Detection | Exact + near-duplicate texts + label conflicts |
| Text Quality | Length extremes, encoding issues, noise |
| Override Analysis | Human correction rate by confidence band |
| Golden Set Validation | Accuracy, precision, recall, F1 per label vs reference |
| Inter-Annotator Agreement | Cohen's Kappa with interpretation |
| HTML Report | Standalone self-contained quality report |

---

## 🗂️ Project Structure

```
data-annotation-pipeline/
│
├── annotator.py              ← Main annotation engine (run this)
├── quality_checker.py        ← QA module (run after annotating)
├── guidelines.md             ← Annotation guidelines for human annotators
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
├── LICENSE                   ← MIT
│
├── config/
│   └── ecommerce_intent.json ← Example custom rule config
│
├── data/
│   ├── raw/                  ← Input files + sample/golden sets (auto-created)
│   ├── annotated/            ← Output CSV + JSONL files (auto-created)
│   └── exports/              ← Label Studio JSON + QA reports (auto-created)
│
├── logs/                     ← Session logs with timestamps (auto-created)
│
└── tests/
    └── test_pipeline.py      ← 42-test suite (all passing ✅)
```

---

## 🚀 Quick Start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Run a demo (generates sample data automatically)

```bash
# Sentiment analysis on 50 sample reviews:
python annotator.py --demo --task sentiment

# Intent classification:
python annotator.py --demo --task intent

# Content moderation (Trust & Safety):
python annotator.py --demo --task content_category
```

### 3 · Label your own data

```bash
# Your CSV just needs a column named: text, content, review, comment, or body
python annotator.py --input your_file.csv --task sentiment --mode auto
```

### 4 · Check annotation quality

```bash
# Run after annotating — generates HTML report
python quality_checker.py --input data/annotated/your_file_sentiment_*.csv

# Generate a golden reference set first, then validate:
python quality_checker.py --gen-golden sentiment
python quality_checker.py --input data/annotated/your_file.csv \
                          --golden data/raw/golden_sentiment.csv
```

### 5 · Run all tests

```bash
python tests/test_pipeline.py
# Expected: 42/42 passed ✅
```

---

## 📖 Detailed Usage

### Annotation Modes

#### `--mode auto` (default)
Labels every item with the rule engine. No human input needed. Best for large batches.

```bash
python annotator.py \
  --input data/raw/reviews.csv \
  --task sentiment \
  --mode auto \
  --export all
```

#### `--mode interactive`
Presents each item for human review. Shows auto-label and confidence. You can accept or override.

```bash
python annotator.py \
  --input data/raw/reviews.csv \
  --task sentiment \
  --mode interactive \
  --annotator "Qowiyu Yusrizal"
```

Terminal UI looks like:
```
──────────────────────────────────────────────────────────────────────
[12/50]  ID: review_0012

TEXT:
  The product arrived damaged and customer service never responded.

AUTO LABEL  : Negative
CONFIDENCE  : 85.7%
SIGNALS     : Negative: 6  |  Positive: 0  |  Neutral: 0

  [1] Positive
  [2] Negative  ← auto
  [3] Neutral
  [Enter] Accept auto-label
  [s]     Skip this item
  [q]     Quit and save progress

  Your choice: _
```

#### `--mode batch`
Same as auto but also exports Label Studio JSON. Use for bulk processing + upload to Label Studio.

```bash
python annotator.py \
  --input data/raw/posts.csv \
  --task content_category \
  --mode batch \
  --export ls
```

---

### Export Formats

| Flag | Output | Best For |
|------|--------|----------|
| `--export csv` | CSV with all fields | Excel, further analysis |
| `--export jsonl` | One JSON per line | ML training pipelines |
| `--export ls` | Label Studio JSON | Import to Label Studio for team review |
| `--export all` | All three above | Full documentation |

---

### Custom Rules

Create a JSON file mapping labels to regex patterns:

```json
{
  "Urgent": [
    "\\b(asap|urgent|immediately|emergency|critical|now)\\b",
    "!!+"
  ],
  "Normal": [
    "\\b(please|when possible|no rush|whenever|fine|okay)\\b"
  ]
}
```

Then use it:
```bash
python annotator.py \
  --input tickets.csv \
  --task custom \
  --config config/my_rules.json \
  --mode auto
```

---

### Quality Checker Options

```bash
# Basic check + HTML report:
python quality_checker.py \
  --input data/annotated/sample.csv \
  --report html

# Full check with golden validation + inter-annotator:
python quality_checker.py \
  --input annotator_a.csv \
  --golden data/raw/golden_sentiment.csv \
  --annotator2 annotator_b.csv \
  --report html \
  --threshold 0.20

# Generate a new golden reference set for any task:
python quality_checker.py --gen-golden content_category
```

---

## 🔢 Output Fields Explained

Every annotated record (CSV/JSONL) contains:

| Field | Type | Description |
|-------|------|-------------|
| `record_id` | UUID | Unique ID for this annotation event |
| `source_id` | str | Original ID from input file |
| `text` | str | Original text content |
| `task` | str | Annotation task (sentiment / intent / content_category) |
| `auto_label` | str | Label assigned by the rule engine |
| `auto_confidence` | float | Confidence score 0.0–1.0 |
| `auto_signals` | dict | Count of matched patterns per label |
| `final_label` | str | Final label (auto or human-overridden) |
| `annotator` | str | Who labeled it (auto or human name) |
| `is_human_reviewed` | bool | Whether a human reviewed this item |
| `human_override` | bool | Whether human changed the auto-label |
| `annotation_time_sec` | float | Seconds spent on this item (human review) |
| `created_at` | ISO datetime | When this annotation was created |

---

## 📊 Quality Metrics Explained

### Cohen's Kappa (κ) — Inter-Annotator Agreement
| κ Value | Interpretation |
|---------|---------------|
| < 0 | Poor (worse than chance) |
| 0.01–0.20 | Slight |
| 0.21–0.40 | Fair |
| 0.41–0.60 | Moderate |
| 0.61–0.80 | **Substantial ← industry target** |
| 0.81–1.00 | Almost Perfect |

### Confidence Score
- **0%** = No keyword signals matched — model is guessing — **always review manually**
- **1–20%** = Very weak signal — review recommended
- **20–50%** = Moderate — spot-check
- **50–80%** = Good — accept with occasional spot-check
- **80–100%** = High — safe to accept

### Override Rate
- **< 15%** = Rules are very well-tuned
- **15–35%** = Normal, healthy range
- **> 35%** = Rules need improvement for flagged labels

---

## 🏭 Label Studio Integration

This pipeline is designed to work seamlessly with [Label Studio](https://labelstud.io) (free, open source).

**Workflow:**

```
1. annotator.py --export ls    → generates label_studio.json
2. Label Studio → Projects → Import → upload JSON
3. Team annotators review and correct predictions
4. Export corrected labels from Label Studio
5. quality_checker.py → run QA on the corrected labels
6. Use final labels for ML model training
```

**Label Studio config for sentiment:**
```xml
<View>
  <Text name="text" value="$text"/>
  <Choices name="label" toName="text" choice="single">
    <Choice value="Positive"/>
    <Choice value="Negative"/>
    <Choice value="Neutral"/>
  </Choices>
</View>
```

---

## 🔮 Extension Ideas

- [ ] Add `--model` flag to use a Hugging Face transformer instead of rule engine
- [ ] Implement active learning: prioritize low-confidence items for human review
- [ ] Add Bahasa Indonesia / Malay keyword dictionaries
- [ ] Build a Streamlit web UI for the annotation session
- [ ] Connect to Label Studio REST API for direct upload/pull
- [ ] Add Bengali, Arabic, or other multilingual rule packs

---

## 📦 Deploy on Any Machine

```bash
# Clone
git clone https://github.com/Exynos93/data-annotation-pipeline.git
cd data-annotation-pipeline

# Install (no GPU needed, no heavy ML libs)
pip install -r requirements.txt

# Verify
python tests/test_pipeline.py

# Run
python annotator.py --demo --task sentiment
```

No internet connection required after install. Works on Windows, macOS, and Linux.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built as Portfolio Project #3 · Personal Branding Series · by Qowiyu Yusrizal (Universitas Indonesia, CS 2024)*
