# 📚 The Classical NLP Roadmap for Data Scientists (2026 Edition)

Before moving to Transformers and LLMs, a Data Scientist must master these "Classic" pillars to handle data cleaning, efficient feature engineering, and interpretable modeling.

---

## 📍 Phase 1: Text Preprocessing (The Cleanup)

**Goal:** Transform noisy, unstructured text into a clean, standardized format.

* **Noise Removal:** Stripping HTML tags, URLs, emojis, and special characters using **Regex** (Regular Expressions).
* **Normalization:** * **Case Folding:** Lowercasing for consistency.
  * **Stemming vs. Lemmatization:** Reducing words to their root form (e.g., "running" → "run"). Know when to use the faster Porter Stemmer vs. the more accurate WordNet Lemmatizer.
* **Tokenization:** Breaking text into words, sentences, or sub-words (using `nltk` or `spaCy`).
* **Stopword Filtering:** Removing common words (a, an, the) that carry little semantic value for classification.

---

## 🔍 Phase 2: Linguistic Metadata (Linguistic Analysis)

**Goal:** Extract structural meaning without using complex neural networks.

* **POS Tagging (Parts of Speech):** Identifying nouns, verbs, and adjectives. Crucial for understanding the "intent" of a sentence.
* **NER (Named Entity Recognition):** Extracting specific entities like Names, Organizations, Dates, and Locations.
* **Dependency Parsing:** Understanding the grammatical relationship between words (e.g., which adjective describes which noun).
* **Stopword Customization:** Learning to build domain-specific stopword lists (e.g., in medical NLP, "patient" might be a stopword if it appears in every document).

---

## 🔢 Phase 3: Text Vectorization (Feature Engineering)

**Goal:** Convert text into numerical "vectors" that Machine Learning models can understand.

### Traditional Count-Based Methods
* **Bag of Words (BoW):** Simple frequency counting. Great for basic spam filters.
* **TF-IDF (Term Frequency-Inverse Document Frequency):** Weighting words by how unique they are to a specific document. This is the "Gold Standard" for classical search and classification.
* **N-Grams:** Capturing context by looking at word pairs (Bigrams) or triplets (Trigrams) (e.g., "not good" vs "good").
* **Hashing Vectorizer:** A memory-efficient way to vectorize massive datasets that don't fit in RAM.

### Word Embeddings (Pre-Transformer Era)
* **Word2Vec:** Learning semantic word representations using neural networks.
  * **Skip-gram:** Predicting context words from a target word.
  * **CBOW (Continuous Bag of Words):** Predicting a target word from context words.
* **GloVe (Global Vectors):** Combining global matrix factorization and local context windows for word representations.
* **FastText:** Extending Word2Vec to handle out-of-vocabulary words by using subword information.
* **Practical Use:** Pre-trained embeddings (Google News, GloVe) for transfer learning before Transformers existed.

---

## 🤖 Phase 4: Classical Modeling & Topic Discovery

**Goal:** Applying statistical models to the vectors created in Phase 3.

### Classification Models
* **Naive Bayes:** The industry standard for speed in text classification.
* **Logistic Regression / SVM:** Higher accuracy for high-dimensional text data.
* **Decision Trees / Random Forest:** For interpretable text classification with feature importance.

### Sequence Models
* **Hidden Markov Models (HMMs):** For POS tagging and sequence prediction tasks.
* **Conditional Random Fields (CRFs):** For sequence labeling tasks like NER (better than HMMs for structured prediction).

### Topic Modeling (Unsupervised)
* **LDA (Latent Dirichlet Allocation):** Discovering hidden "themes" in a collection of documents without labels.
* **LSA (Latent Semantic Analysis):** Using SVD to find latent topics and reduce dimensionality.
* **NMF (Non-negative Matrix Factorization):** Alternative to LDA with more interpretable components.

### Similarity & Distance Metrics
* **Cosine Similarity:** Finding how related two documents are (useful for "Similar Article" recommendations).
* **Euclidean Distance:** Measuring straight-line distance between vectors.
* **Jaccard Similarity:** For set-based comparisons (good for comparing document vocabularies).
* **Levenshtein Distance (Edit Distance):** For spell checking and fuzzy string matching.

---

## 🛠️ Phase 5: Implementation & Evaluation

**Goal:** Moving from a script to a usable pipeline.

### Pipeline Development
* **Pipelines:** Using `scikit-learn` Pipelines to chain Preprocessing → Vectorization → Modeling into a single object.
* **Custom Transformers:** Building your own preprocessing steps that fit into scikit-learn pipelines.

### Evaluation Metrics
* **Classification:**
  * **F1-Score:** Crucial for text because classes are often imbalanced (e.g., 99% Not-Spam vs. 1% Spam).
  * **Precision, Recall, Accuracy:** Understanding trade-offs between false positives and false negatives.
  * **Confusion Matrix:** Identifying exactly which categories are being confused.
* **Sequence Labeling (NER, POS):**
  * **Token-level F1:** Evaluating each token classification.
  * **Entity-level F1:** Evaluating complete entity extraction.
* **Text Generation & Summarization:**
  * **BLEU Score:** For machine translation and text generation quality.
  * **ROUGE Score:** For summarization evaluation (overlap with reference summaries).
* **Topic Modeling:**
  * **Coherence Score:** Measuring how interpretable discovered topics are.
  * **Perplexity:** Lower is better for probabilistic models.

### Cross-Validation Strategies
* **Stratified K-Fold:** Maintaining class distribution across folds.
* **Time-based Splits:** For temporal text data (e.g., news articles, tweets).
* **Group-based Splits:** Avoiding data leakage (e.g., keeping all documents from the same author in the same fold).

### Libraries to Master
* **NLTK:** Best for education and complex linguistics.
* **spaCy:** Best for production-grade speed and NER.
* **Scikit-Learn:** Best for vectorization and classical ML.
* **Gensim:** Best for topic modeling (LDA) and word embeddings (Word2Vec).

---

## 🚀 Phase 6: Real-World Applications

**Goal:** Applying classical NLP techniques to solve practical problems.

### Sentiment Analysis
* **Lexicon-Based Approaches:** Using pre-built sentiment dictionaries (VADER, TextBlob, SentiWordNet).
* **Machine Learning Approaches:** Training classifiers on labeled sentiment data.
* **Aspect-Based Sentiment:** Identifying sentiment towards specific aspects (e.g., "food was great but service was slow").

### Text Summarization
* **Extractive Summarization:**
  * **TextRank:** Graph-based ranking algorithm (similar to PageRank).
  * **TF-IDF Ranking:** Selecting sentences with highest TF-IDF scores.
  * **LSA-Based:** Using latent semantic analysis to identify key sentences.
* **Evaluation:** ROUGE scores for comparing against reference summaries.

### Information Extraction
* **Named Entity Recognition (NER):** Extracting persons, organizations, locations, dates.
* **Relation Extraction:** Finding relationships between entities (e.g., "Obama was born in Hawaii").
* **Coreference Resolution:** Understanding what pronouns and references point to.
* **Event Extraction:** Identifying events and their participants from text.
* **Regular Expression Patterns:** Advanced pattern matching for structured data extraction.

### Spell Checking & Autocomplete
* **Edit Distance Algorithms:** Finding closest correct words using Levenshtein distance.
* **N-gram Language Models:** Predicting next word probabilities for autocomplete.
* **Phonetic Algorithms:** Soundex and Metaphone for sound-based matching.
* **Context-Aware Correction:** Using surrounding words to disambiguate corrections.

### Language Models (Classical)
* **N-gram Models:** Computing probabilities of word sequences.
* **Smoothing Techniques:** Handling unseen n-grams (Laplace, Good-Turing, Kneser-Ney).
* **Perplexity:** Evaluating how well a model predicts text.
* **Applications:** Autocomplete, spell checking, text generation.

### Document Classification at Scale
* **Multi-label Classification:** Documents belonging to multiple categories.
* **Hierarchical Classification:** Organizing documents into taxonomies.
* **Online Learning:** Updating models as new documents arrive.
* **Handling Imbalanced Data:** Techniques for rare classes.

---

### 🎓 Summary Table for Daily Use

| Step                      | Key Technique                  | Tool of Choice                      |
| :------------------------ | :----------------------------- | :---------------------------------- |
| **Cleaning**              | Regex / Normalization          | `re`, `string`                      |
| **Structure**             | POS Tagging / NER              | `spaCy`, `nltk`                     |
| **Vectorizing (Classic)** | TF-IDF / N-Grams               | `scikit-learn`                      |
| **Vectorizing (Modern)**  | Word2Vec / GloVe / FastText    | `gensim`, `spaCy`                   |
| **Topic Modeling**        | LDA / LSA / NMF                | `scikit-learn`, `gensim`            |
| **Sequence Models**       | HMM / CRF                      | `sklearn-crfsuite`, `nltk`          |
| **Classification**        | Naive Bayes / SVM              | `scikit-learn`                      |
| **Similarity**            | Cosine / Edit Distance         | `scikit-learn`, `nltk`              |
| **Sentiment**             | VADER / TextBlob               | `vaderSentiment`, `textblob`        |
| **Summarization**         | TextRank / LSA                 | `gensim`, `sumy`                    |
| **Evaluation**            | F1 / BLEU / ROUGE              | `scikit-learn`, `nltk`              |
