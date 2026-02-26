# 📋 Data Annotation Guidelines

**Project:** Semi-Automated Data Annotation Pipeline  
**Author:** Qowiyu Yusrizal — [hihakai123@gmail.com](mailto:hihakai123@gmail.com)  
**GitHub:** [github.com/Exynos93](https://github.com/Exynos93)  
**Version:** 1.0.0  
**Last Updated:** 2024

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [General Principles](#2-general-principles)
3. [Task A: Sentiment Analysis](#3-task-a-sentiment-analysis)
4. [Task B: Intent Classification](#4-task-b-intent-classification)
5. [Task C: Content Category (Trust & Safety)](#5-task-c-content-category-trust--safety)
6. [Handling Edge Cases](#6-handling-edge-cases)
7. [Quality Standards](#7-quality-standards)
8. [Annotator Workflow](#8-annotator-workflow)
9. [Label Definitions Quick Reference](#9-label-definitions-quick-reference)
10. [FAQ](#10-faq)

---

## 1. Introduction

This document defines labeling standards for the **Data Annotation Pipeline** built to train and evaluate AI/ML models used in:

- **Content moderation** (TikTok-style platforms)
- **Search quality rating** (Google/Bing-style search)
- **Customer service automation** (e-commerce, CRM)
- **Trust & safety systems** (social platforms)

Consistent, high-quality labels are the foundation of reliable AI. A model is only as good as its training data. **Your judgment as an annotator is the most important signal in this pipeline.**

---

## 2. General Principles

### 2.1 The Cardinal Rules

> **Rule 1:** Label what the text **IS**, not what you think the author **intended**.  
> **Rule 2:** When in doubt, choose the label that fits the **majority** of the content.  
> **Rule 3:** Context matters — but only context **within the text**, not assumed external context.  
> **Rule 4:** Speed ≠ quality. A wrong label takes 30 seconds to give and weeks to fix downstream.

### 2.2 Label Independence

Each text must be labeled **independently**. Do not:
- Compare texts to each other
- Look at previous labels to decide the next one
- Let batch patterns influence individual decisions

### 2.3 Confidence Signal

The system assigns an **auto-confidence score (0–100%)** to every prediction.

| Score Range | What It Means | Your Action |
|-------------|---------------|-------------|
| 0%          | No signals matched — model is guessing | **Always review** |
| 1–20%       | Very weak signal | **Review recommended** |
| 20–50%      | Moderate confidence | Review if time allows |
| 50–80%      | Good confidence | Accept or spot-check |
| 80–100%     | High confidence | Accept (spot-check 5%) |

### 2.4 What Counts as "Text"

Label the **primary communicative content**. Ignore:
- Metadata (timestamps, user IDs, URLs unless relevant)
- HTML tags (label the visible text intent)
- Typos and spelling errors (label the clear meaning, not the error)

---

## 3. Task A: Sentiment Analysis

**Goal:** Determine the overall emotional tone of a customer-written text.

### Labels

#### ✅ Positive
The text expresses **satisfaction, happiness, praise, or approval**.

**Key signals:**
- Explicit praise words: *amazing, love, excellent, perfect, fantastic*
- Satisfaction with outcome: *works well, exactly as described, fast delivery*
- Recommendation language: *would recommend, five stars, best purchase*
- Positive emojis: 😍 👍 ❤️ 💯 🎉 ✨

**Examples:**
```
✅  "Absolutely love this product! Best purchase I've made all year. 😍"
✅  "Fast shipping, great quality, exactly what I expected. 5 stars!"
✅  "Customer service was incredibly helpful. Problem solved in minutes."
```

---

#### ❌ Negative
The text expresses **disappointment, frustration, complaint, or dissatisfaction**.

**Key signals:**
- Complaint language: *broken, defective, terrible, worst, disappointed*
- Refund/return intent: *want my money back, returning this, demand refund*
- Unmet expectations: *not as described, completely different, waste of money*
- Negative emojis: 😠 😡 👎 💔 😤

**Examples:**
```
❌  "Terrible product. Broke after 2 days. Complete waste of money."
❌  "Never buying from this store again. Item arrived 3 weeks late."
❌  "Worst customer service experience. No response for 2 weeks!"
```

---

#### ➖ Neutral
The text is **informational, ambivalent, or does not lean clearly positive or negative**.

**Key signals:**
- Factual statements without emotional language
- Mixed positive/negative in roughly equal balance
- Questions or requests for information
- Hedging language: *okay, alright, average, not sure, neither*

**Examples:**
```
➖  "The product arrived. It does what it says on the box."
➖  "Shipping was slow but the quality is decent."  ← MIXED = Neutral
➖  "Can anyone tell me if this fits a size 10 foot?"
```

### Sentiment Edge Cases

| Situation | Label | Reasoning |
|-----------|-------|-----------|
| Sarcasm: *"Oh great, another delayed order."* | Negative | Tone is clearly negative in context |
| Mixed: *"Product is good but service was terrible"* | Neutral | Roughly balanced |
| Question only: *"Does this come in red?"* | Neutral | No sentiment expressed |
| Emojis only: *"😡😡😡"* | Negative | Clear negative signal |
| Hyperbole positive: *"Best thing since sliced bread!"* | Positive | Despite being unrealistic, intent is positive |
| Review of a bad product in positive tone: *"Works exactly as described — and it described rubbish."* | Negative | Final evaluation is negative |

---

## 4. Task B: Intent Classification

**Goal:** Identify the **primary information need** behind a query or message.

Use cases: search quality rating, chatbot routing, recommendation systems.

### Labels

#### 🔍 Informational
The user wants to **learn** something — a fact, explanation, or background knowledge.

**Key signals:**
- Question words: *what, why, how, when, where, who, which*
- Learning intent: *explain, define, history of, causes of, difference between*
- Research terms: *statistics, data, overview, guide, tutorial*

**Examples:**
```
🔍  "What is reinforcement learning from human feedback?"
🔍  "History of the Internet"
🔍  "Why does the sky appear blue?"
🔍  "How do vaccines work?"
```

---

#### 🛒 Transactional
The user wants to **do** something — buy, download, sign up, register.

**Key signals:**
- Purchase intent: *buy, purchase, order, checkout, price, cost*
- Action words: *download, install, subscribe, register, book, hire*
- Offer language: *discount, coupon, promo, deal, free trial*

**Examples:**
```
🛒  "Buy cheap laptop online free shipping"
🛒  "Download Python 3.12 installer"
🛒  "Sign up for Netflix free trial"
🛒  "20% off promo code for Shopee"
```

---

#### 🧭 Navigational
The user wants to **go to** a specific website, page, or platform.

**Key signals:**
- Login/access terms: *login, sign in, account, dashboard, profile*
- Platform references: *website, site, app, portal, official page*
- Direction words: *go to, open, visit, find, where is*

**Examples:**
```
🧭  "Facebook login"
🧭  "Toyota official website Malaysia"
🧭  "Gmail inbox"
🧭  "Apple support contact page"
```

---

#### 💬 Conversational
The user is **engaging in dialogue** — greetings, opinions, small talk, recommendations.

**Key signals:**
- Greetings: *hi, hello, hey, good morning*
- Social language: *thanks, sorry, please, how are you*
- Opinion requests: *what do you think, recommend, suggest, your opinion*

**Examples:**
```
💬  "Hey! Can you recommend a Python book for beginners?"
💬  "Thanks for your help yesterday!"
💬  "What's your opinion on electric cars?"
💬  "Good morning! How's the weather today?"
```

### Intent Edge Cases

| Situation | Label | Reasoning |
|-----------|-------|-----------|
| *"Best laptop price comparison"* | Transactional | Price comparison is purchase-oriented |
| *"Wikipedia machine learning"* | Navigational | User wants to go to the Wikipedia page |
| *"How to lose weight fast"* | Informational | Primary need is knowledge, not a product |
| *"Buy a guide to Python"* | Transactional | Action (buy) dominates, not the learning |
| *"What is the best VPN?"* | Informational | Seeking knowledge, not taking action yet |

---

## 5. Task C: Content Category (Trust & Safety)

**Goal:** Classify content that may violate platform policies. Used for content moderation at scale.

> ⚠️ **Important:** This task may expose you to harmful content. If you feel distressed, stop and take a break. You are not required to review graphic content beyond what is needed to classify it.

### Labels

#### 🚫 Spam
Commercial or deceptive content intended to manipulate the reader into clicking, buying, or engaging artificially.

**Key signals:**
- Urgency manipulation: *act now, limited offer, last chance*
- Unrealistic claims: *earn $5,000/week, guaranteed results, 100% free*
- Engagement bait: *follow4follow, sub4sub, like4like*
- Multiple irrelevant links (3+)

**Examples:**
```
🚫  "EARN $5000 A WEEK FROM HOME! Click here NOW! Limited offer!"
🚫  "Win a free iPhone! You've been SELECTED! Act within 24 hours!"
🚫  "F4F follow back instantly — 100k followers in 24 hours!"
```

---

#### ⚠️ Hate Speech
Content that **targets people** based on protected characteristics (race, religion, gender, ethnicity, sexual orientation, disability, national origin) with language that degrades, dehumanizes, or incites hatred.

**Key signals:**
- Dehumanising language applied to a group
- Calls to exclude or harm a demographic
- Slurs used as attacks (not quoted or educational)

> **Note:** Discussion of hate speech for educational or journalistic purposes is **NOT** hate speech. Label the intent and context.

---

#### 💣 Violence
Content containing **threats, instructions, or glorification of physical harm**.

**Key signals:**
- Credible threats against individuals or groups
- Instructions for creating weapons or carrying out attacks
- Glorification of assault, torture, or murder

> **Note:** News reports about violence, fictional violence in stories, or historical accounts are typically **Safe** or **Misinformation** — not Violence.

---

#### ❗ Misinformation
False or misleading claims presented as fact, particularly in health, science, politics, or current events.

**Key signals:**
- Conspiracy language: *they don't want you to know, hidden truth, cover-up*
- False health claims: *miracle cure, 100% effective, no side effects, doctors hate*
- Sensationalist framing: *shocking truth, exposed, secret revealed*

> **Calibration:** If you're unsure if something is factually false, label it **Safe** and add a note. Do not label something Misinformation just because you disagree with it.

---

#### 🔞 NSFW
Sexually explicit content or content appropriate only for adult audiences.

**Key signals:**
- Explicit tags: *nsfw, 18+, adult content, x-rated*
- Graphic sexual language

---

#### ✅ Safe
Normal, acceptable content that does not violate any policy.

**Key signals:**
- Day-to-day communication, news, updates
- Educational content, tutorials, reviews
- General social posts, community content

---

## 6. Handling Edge Cases

### 6.1 Mixed Language Text

Label based on the **dominant language**. If text is 70% English and 30% Bahasa Indonesia, apply English criteria. If you cannot understand enough to label, mark as **Unknown** and leave a note.

### 6.2 Very Short Texts

Texts under 10 words should still be labeled if intent is clear.
- *"Worst product ever"* → Negative
- *"Buy now!"* → Spam
- *"Hello!"* → Conversational

For truly ambiguous short texts (single emoji, single word with no context), use:
- The final label that requires the lowest confidence to justify
- Add a note: `[SHORT TEXT — ambiguous]`

### 6.3 Non-English Text

The pipeline supports multilingual annotation. Apply the **same criteria** — the emotional and communicative intent transcends language.

*"Produk ini sangat bagus!"* → Positive (very clear)  
*"Terrible, terrible, benar-benar rugi"* → Negative (mixed but clear)

### 6.4 Quoted Text

If text **quotes** something harmful for informational/critical purposes, label based on the **framing**, not the quoted content.

*"The report noted that the suspect said 'I will harm you' — police have made an arrest."*
→ Label: **Safe** (news report, not a threat)

*"Share this everywhere: [harmful claim]"*
→ Label based on the claim being amplified.

---

## 7. Quality Standards

### 7.1 Acceptance Criteria

| Metric | Minimum Standard | Target |
|--------|-----------------|--------|
| Inter-Annotator Agreement (Cohen's Kappa) | κ ≥ 0.60 (Substantial) | κ ≥ 0.75 |
| Golden Set Accuracy | ≥ 70% | ≥ 85% |
| Completion Rate | ≥ 95% (no skips without reason) | 100% |
| Zero-signal Review Rate | < 30% | < 15% |
| Override Rate (human vs auto) | N/A (natural) | 15–35% is healthy |

### 7.2 Common Annotation Errors to Avoid

| Error | Description | Correct Approach |
|-------|-------------|-----------------|
| **Label leakage** | Using knowledge of true label to guide annotation | Label independently every time |
| **Central tendency bias** | Always picking the "safe" middle label (Neutral, Safe) | Apply criteria strictly; don't default to middle |
| **Halo effect** | Rating all aspects of a text highly because one aspect is good | Each text gets one label based on its dominant signal |
| **Speed bias** | Rushing → accepting auto-labels without reading | Read every text before confirming |
| **Anchoring** | Letting the auto-label influence your judgment too heavily | Form your own opinion first, then compare |

### 7.3 Calibration Sessions

At the start of each annotation batch, complete a **10-item calibration set** of pre-labeled examples. If your agreement with the gold labels is < 80%, re-read these guidelines before continuing.

---

## 8. Annotator Workflow

### 8.1 Session Setup

```
1. Run:  python annotator.py --demo --task <task>
2. Complete the 10-item warm-up (first 10 items in demo)
3. Check your calibration score in the output summary
4. If score < 80%, re-read Section 3/4/5 for your task
5. Begin your full annotation session
```

### 8.2 Per-Item Workflow

```
For each item:
  1. READ the full text
  2. IDENTIFY the dominant signal
  3. FORM your own label (before looking at auto-label)
  4. COMPARE with auto-label
  5. If you agree → press Enter
  6. If you disagree → select your label (1, 2, 3...)
  7. If genuinely ambiguous → select most defensible label
  8. If severely ambiguous → skip [s] and add to review list
```

### 8.3 End-of-Session Checklist

- [ ] All items have a final label (no blanks)
- [ ] Override notes added for unusual decisions
- [ ] Session exported to `data/annotated/`
- [ ] Run `python quality_checker.py --input <your_file>` and check score

### 8.4 When to Escalate

Escalate to a senior annotator or task lead if:
- You encounter a text you believe is genuinely illegal content
- A text pattern appears frequently that no label covers well
- Your calibration score drops below 70% for two sessions in a row
- You have a policy interpretation question not covered in these guidelines

---

## 9. Label Definitions Quick Reference

### Sentiment

| Label | One-line definition | Example |
|-------|--------------------|-|
| **Positive** | Expresses satisfaction, praise, or approval | *"Absolutely love it! Highly recommend."* |
| **Negative** | Expresses disappointment, frustration, or complaint | *"Terrible. Broke on day one."* |
| **Neutral** | Factual, ambivalent, or mixed signals | *"Product arrived. Does its job."* |

### Intent

| Label | One-line definition | Example |
|-------|--------------------|-|
| **Informational** | User wants to learn something | *"What is quantum computing?"* |
| **Transactional** | User wants to do something | *"Buy discounted AirPods"* |
| **Navigational** | User wants to go somewhere | *"Gmail login page"* |
| **Conversational** | User is engaging in dialogue | *"Any book recommendations?"* |

### Content Category

| Label | One-line definition | Example |
|-------|--------------------|-|
| **Spam** | Commercial manipulation / engagement bait | *"Win free iPhone! Act NOW!"* |
| **Hate Speech** | Degrades group based on identity | *(see section 5)* |
| **Violence** | Threats or glorification of harm | *(see section 5)* |
| **Misinformation** | False claims presented as fact | *"Miracle cure doctors don't want you to know"* |
| **NSFW** | Adult/explicit content | *"18+ explicit content"* |
| **Safe** | Normal, acceptable content | *"Had a great day at work today!"* |

---

## 10. FAQ

**Q: What if I read the same text twice and would label it differently?**  
A: This is normal — your first impression is usually more reliable. If you notice inconsistency in your own labels, that's a sign to slow down and re-read the guidelines for the problematic label.

**Q: The auto-label looks obviously wrong. Should I always override it?**  
A: Yes, always override if you're confident the auto-label is wrong. The system learns from your overrides. High override rates on a specific label are actually valuable data for improving the rules.

**Q: Can I use external knowledge (Google, Wikipedia) to decide a label?**  
A: For **Misinformation** task only — yes, checking facts is appropriate. For other tasks, label only based on the text itself.

**Q: The text is in a language I don't know. What do I do?**  
A: If you cannot determine meaning even with tools, use **Unknown** and flag for native-speaker review. Do not guess.

**Q: What if the auto-label confidence is 0%?**  
A: This means the rule engine found no matching patterns. The label is essentially a default. You **must** review these manually.

**Q: How many items can I annotate per hour?**  
A: Quality benchmarks vary by task:
- Sentiment: 200–400 items/hour
- Intent: 150–300 items/hour  
- Content Category: 100–200 items/hour (slower due to complexity)

Going faster than these rates consistently suggests you may be skimming, not reading.

---

*These guidelines are a living document. If you find cases not covered here, document them and raise them with the project lead. Your feedback makes the guidelines better for everyone.*

---

**Document maintained by:** Qowiyu Yusrizal  
**Contact:** [hihakai123@gmail.com](mailto:hihakai123@gmail.com)  
**GitHub:** [github.com/Exynos93](https://github.com/Exynos93)
