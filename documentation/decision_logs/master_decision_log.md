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

---

## Project 03: IMDB Movie Reviews Dataset

### 1. Stopword Removal Strategy
*   **Decision:** Do **NOT** remove stopwords during cleaning.
*   **Alternatives:** Standard stopword removal (reduces feature space).
*   **Reasoning:** For Sentiment Analysis, negations (e.g. 'not', 'very', 'too') are critical. Standard stopword lists typically strip these words out. Removing them turns "not good" into "good", completely flipping the sentiment. We leave them in to preserve context.

### 2. N-Gram Vectorization
*   **Decision:** Use `CountVectorizer(ngram_range=(1, 2))` (Unigrams + Bigrams).
*   **Alternatives:** Unigrams only (Misses context/negations), Trigrams (Explodes matrix size).
*   **Reasoning:** Bigrams strike the perfect balance. They capture local context like "not bad" or "very good" without the exponential matrix explosion and extreme sparsity that trigrams introduce.

### 3. Handling Massive Feature Spaces
*   **Decision:** Use L2 Regularized Linear Models (`LogisticRegression` and `RidgeClassifier`).
*   **Alternatives:** Unregularized models, Tree-based models.
*   **Reasoning:** With Bigrams on 50,000 documents, the feature space grows to over 100,000 columns. Tree-based models fail completely on massive sparse data. Unregularized models will overfit instantly (the curse of dimensionality). L2 Regularization aggressively shrinks noisy coefficients, forcing the model to rely only on truly generalizable sentiment indicators.

---

## Project 04: A Million News Headlines (Topic Modeling)

### 1. Dataset Sub-Sampling
*   **Decision:** Sample the 1.2M rows down to 100,000 rows.
*   **Alternatives:** Use full dataset.
*   **Reasoning:** Topic modeling (especially LDA) is highly computationally expensive. Sampling to 100k ensures models train in minutes instead of hours on a local machine, while still providing a robust document size to discover meaningful topics.

### 2. Aggressive Text Preprocessing
*   **Decision:** Heavy Stopword Removal + Lemmatization + Drop words < 3 chars.
*   **Alternatives:** Basic cleaning, Stemming.
*   **Reasoning:** Topic modeling relies purely on word co-occurrence. If noisy words (like 'the', 'is', 'to') or short artifacts aren't removed, they will completely dominate the topics. Lemmatization is strongly preferred over stemming because unsupervised models must output human-readable words.

### 3. Model Comparison: LDA vs NMF
*   **Decision:** Compare Latent Dirichlet Allocation (LDA) with Bag-of-Words against Non-Negative Matrix Factorization (NMF) with TF-IDF.
*   **Alternatives:** Embeddings / BERTopic.
*   **Reasoning:** LDA assumes documents are generated by probability distributions of word counts, hence it requires CountVectorizer. NMF decomposes a matrix via linear algebra and works best with TF-IDF to penalize ubiquitous words. Often, NMF combined with TF-IDF extracts much more coherent topics on extremely short texts like news headlines compared to LDA.
