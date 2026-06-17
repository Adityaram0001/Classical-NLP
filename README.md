# Classical NLP Deep Dive

This repository is a structured machine learning learning project where we implement Classical Natural Language Processing (NLP) algorithms in depth across multiple datasets.

## Project Structure

### 1. `datasets/`
Contains the practical implementation of various NLP datasets, exploring algorithms ranging from simple Naive Bayes to more complex models like CRFs.
Each dataset folder includes:
- `data/raw/` - Raw datasets.
- `data/processed/` - Cleaned datasets and vectorizers.
- `notebooks/` - Explanatory Jupyter notebooks featuring our "Decision Note" design philosophy.
- `models/` & `results/` - Saved models and metrics.

Currently implemented projects:
- **[01_sms_spam_collection](datasets/01_sms_spam_collection/)**: Text cleaning, Bag-of-Words vs TF-IDF, and Naive Bayes baseline.

### 2. `shared/`
Contains reusable utility functions shared across datasets:
- Text cleaning functions (`text_utils.py`)
- Standardized evaluation metrics and plots (`evaluation_utils.py`)

### 3. `documentation/`
Contains theoretical material and decision logs:
- **`learning_materials/`**: Extensive Q&A files covering 6 phases of classical NLP (Preprocessing, Metadata, Vectorization, Modeling, Evaluation, Real-World Apps).
- **`decision_logs/`**: Logs of architectural decisions made during the practical projects.

---

## 🚀 Getting Started

If you want to review the theoretical concepts, head over to `documentation/learning_materials/original_README.md`.

If you want to dive into the practical code, start with the notebooks in `datasets/01_sms_spam_collection/notebooks/`.

### Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
