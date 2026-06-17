# Master Decision Log — Classical NLP Deep Dive

This file tracks major architectural or conceptual decisions made across the various projects.

## Project 01: SMS Spam Collection

### 1. Vectorization Method
*   **Decision:** Compare Bag-of-Words (BoW) and TF-IDF.
*   **Alternatives:** Word2Vec/Embeddings (Overkill for spam detection).
*   **Reasoning:** TF-IDF usually performs slightly better on spam as it downweights common English words automatically. Comparing both empirically provides the best learning outcome.

### 2. Model Selection: Naive Bayes
*   **Decision:** Use `MultinomialNB` as the baseline.
*   **Alternatives:** `GaussianNB` (Expects normal distribution, poor for sparse text), `BernoulliNB` (Ignores word frequency).
*   **Reasoning:** Multinomial Naive Bayes is the industry standard baseline for text classification tasks. It works exceptionally well with discrete features like word counts and handles high dimensionality gracefully.

### 3. Model Comparison
*   **Decision:** Compare NB with Logistic Regression and Support Vector Machines (Linear Kernel).
*   **Alternatives:** Tree-based models like Random Forest.
*   **Reasoning:** Tree-based models often underperform on high-dimensional sparse text compared to linear models. For spam detection, SVM (Linear) typically provides the absolute best linear separation and minimizes False Positives (legitimate texts marked as spam).

---

## Project 02: BBC News Classification

### 1. Morphological Reduction
*   **Decision:** Compare Stemming (Porter) vs Lemmatization (WordNet).
*   **Alternatives:** No reduction (Keeps vocabulary too large).
*   **Reasoning:** Stemming is fast and aggressive but can create non-words (e.g., 'organization' -> 'organ'), causing semantic collisions. Lemmatization preserves real words and meaning but is slower. Comparing their impact on vocabulary size empirically is the best way to choose.

### 2. Vectorization Tuning
*   **Decision:** TF-IDF with `min_df=5` and `max_features=5000`.
*   **Alternatives:** Default vectorization with no limits (creates a massive, highly sparse matrix prone to overfitting).
*   **Reasoning:** A `min_df` of 5 drops words appearing in less than 5 documents, effectively pruning typos and rare entities, while `max_features` limits memory footprint while capturing the most important signals.

### 3. Multi-Class Strategy
*   **Decision:** Use MultinomialNB and LogisticRegression.
*   **Alternatives:** Binary Classifiers (Complex to orchestrate for 5 classes manually 1v1).
*   **Reasoning:** Both chosen models handle multi-class problems natively or via One-Vs-Rest seamlessly. They are the golden standard for BoW/TF-IDF text classification across multiple categories.
