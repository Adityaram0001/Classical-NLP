# Handoff Context — Classical NLP Deep Dive Project

## Project Location
`/Users/adityaram/2026/classical_NLP`

## What This Project Is
A structured ML learning project where the user learns Classical NLP algorithms in depth across multiple Kaggle datasets. Each dataset lives in its own folder under `datasets/`. Shared documentation and utilities live outside.

Key design requirement: **Every non-trivial step must explain WHY this approach, WHAT alternatives exist, and WHAT impact the decision has.** This "decision-first" philosophy runs through every notebook and documentation file.

---

## Files Already Created ✅

### Root Level
- `datasets_roadmap.json` — All 6 planned datasets with learning goals and complexity ratings.
- `CONTINUATION_CONTEXT.md` — This file.
- `README.md` & Phase Q&A files — Pre-existing learning documentation.

### shared/
- `shared/__init__.py`
- `shared/text_utils.py` — Reusable text cleaning and tokenization logic.
- `shared/evaluation_utils.py` — Reusable metrics for classification, similarity, and NER.

### documentation/
- `documentation/decision_logs/master_decision_log.md` — To keep track of major decisions across projects.

### datasets/01_sms_spam_collection/
- Fully implemented. Code generation scripts and notebooks executed.

### datasets/02_bbc_news_classification/
- Fully implemented. Code generation scripts and notebooks executed. Models saved.

### datasets/03_imdb_movie_reviews/
- Folder structure created.
- Dataset downloaded (`dataset.csv`) using Hugging Face `datasets`.
- `clean_html_text` util added to `shared/`.
- Notebook generation scripts and execution completed.

---

## What Still Needs to Be Done ❌

### 1. Create Notebooks for SMS Spam Collection
Location: `/Users/adityaram/2026/classical_NLP/datasets/01_sms_spam_collection/notebooks/`

Notebooks to create:
1. `01_eda_and_cleaning.ipynb`
2. `02_text_vectorization.ipynb`
3. `03_naive_bayes_classifier.ipynb`
4. `04_model_comparison.ipynb`

*Note*: Use `gen_01_sms_spam_notebooks.py` to generate these programmatically to avoid token limits.

### 2. Download the Dataset
```bash
# Needs to be run inside the appropriate environment
# kaggle datasets download -d uciml/sms-spam-collection-dataset -p datasets/01_sms_spam_collection/data/raw
```

---

## Notebook Content Guidelines (CRITICAL — DO NOT SKIP)

### Cell Structure Pattern
Each notebook follows: **Markdown title → imports → section markdown → code → section markdown → code...**

### Markdown Decision-Note Pattern
For every non-trivial step, include a markdown cell formatted like:

```markdown
> **📌 Decision Note — Why [step]?**
>
> **Chosen approach:** [what we do]
>
> **Why this works:** [explanation]
>
> **Alternatives we could have used:**
> | Option | Pros | Cons |
> |--------|------|------|
> | Option A | ... | ... |
> | Option B | ... | ... |
>
> **Why we chose this over alternatives:** [reasoning]
```

This pattern applies to: cleaning method, stopword usage, stemming vs lemmatization, vectorization choice (BoW vs TF-IDF), classifier choice, metric selection.
