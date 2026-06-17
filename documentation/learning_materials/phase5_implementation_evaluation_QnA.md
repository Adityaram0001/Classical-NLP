# Phase 5: Implementation & Evaluation - Q&A

> **Goal:** Moving from a script to a usable pipeline and properly evaluating results.

---

## Part A: Pipeline Development

### Q1: What is a scikit-learn Pipeline and why should you use it?
**A:** A Pipeline chains preprocessing and modeling steps into a single object, preventing data leakage and making code cleaner.

**Without Pipeline (prone to errors):**
```python
# Training
X_train_tfidf = vectorizer.fit_transform(X_train)  # ← Fit on train
model.fit(X_train_tfidf, y_train)

# Testing - EASY TO FORGET transform!
X_test_tfidf = vectorizer.transform(X_test)  # ← transform only!
predictions = model.predict(X_test_tfidf)
```

**With Pipeline (automatic):**
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Training - fit_transform applied automatically
pipeline.fit(X_train, y_train)

# Testing - transform applied automatically (no fit!)
predictions = pipeline.predict(X_test)
```

**Benefits:**
- ✅ **Prevents data leakage:** Can't accidentally fit on test data
- ✅ **Cleaner code:** One object instead of multiple
- ✅ **Easy grid search:** Tune all parameters at once
- ✅ **Reproducible:** Save entire pipeline with `joblib`
- ✅ **Production-ready:** Deploy single object

### Q2: How do you build custom transformers for text preprocessing?
**A:** Extend `BaseEstimator` and `TransformerMixin` to create transformers that work in pipelines.

**Example: Custom text cleaner**
```python
from sklearn.base import BaseEstimator, TransformerMixin
import re

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, remove_html=True, remove_urls=True):
        self.remove_html = remove_html
        self.remove_urls = remove_urls
    
    def fit(self, X, y=None):
        # No fitting needed for deterministic preprocessing
        return self
    
    def transform(self, X):
        X_clean = []
        for text in X:
            if self.remove_html:
                text = re.sub(r'<[^>]+>', '', text)
            if self.remove_urls:
                text = re.sub(r'http\S+', '', text)
            text = text.lower().strip()
            X_clean.append(text)
        return X_clean

# Use in pipeline
pipeline = Pipeline([
    ('cleaner', TextCleaner()),
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])
```

**Example: Custom feature extractor**
```python
class TextStats(BaseEstimator, TransformerMixin):
    """Extract statistical features from text"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            features.append([
                len(text),                    # Character count
                len(text.split()),            # Word count
                text.count('!'),              # Exclamation marks
                text.count('?'),              # Question marks
                sum(1 for c in text if c.isupper()) / len(text)  # Caps ratio
            ])
        return np.array(features)

# Combine with text features using FeatureUnion
from sklearn.pipeline import FeatureUnion

features = FeatureUnion([
    ('tfidf', TfidfVectorizer()),
    ('stats', TextStats())
])

pipeline = Pipeline([
    ('features', features),
    ('classifier', RandomForestClassifier())
])
```

**Key methods:**
- `fit(X, y)`: Learn from training data (often just `return self`)
- `transform(X)`: Apply transformation
- `fit_transform(X, y)`: Combined (automatically provided by `TransformerMixin`)

### Q3: How do you do hyperparameter tuning with text pipelines?
**A:** Use `GridSearchCV` or `RandomizedSearchCV` with parameter naming convention `step__parameter`.

**Example:**
```python
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

param_grid = {
    'vectorizer__max_features': [1000, 5000, 10000],
    'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'vectorizer__min_df': [1, 5, 10],
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best F1 score:", grid_search.best_score_)

# Use best model
best_pipeline = grid_search.best_estimator_
```

**RandomizedSearchCV for large search spaces:**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    'vectorizer__max_features': randint(1000, 20000),
    'vectorizer__min_df': randint(1, 20),
    'classifier__C': uniform(0.01, 100),
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=50,  # Try 50 random combinations
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
```

---

## Part B: Evaluation Metrics

### Q4: Why is accuracy often misleading for text classification?
**A:** Text classification often has **imbalanced classes**, making accuracy a poor metric.

**Example: Spam detection**
```
Dataset: 990 ham emails, 10 spam emails (99% ham)

Naive classifier: Always predict "ham"
Accuracy = 990/1000 = 99%

But it's completely useless! Misses all spam!
```

**The problem:**
- Accuracy treats all errors equally
- Doesn't distinguish which class you're wrong about
- Dominated by majority class

**Better metrics for imbalanced data:**
- **Precision:** Of predicted spam, how many are actually spam?
- **Recall:** Of actual spam, how many did we catch?
- **F1 Score:** Harmonic mean of precision and recall

### Q5: What's the difference between Precision, Recall, and F1?
**A:**

**Definitions:**

**Precision:** Of items we labeled positive, how many are truly positive?
```
Precision = TP / (TP + FP)
"How accurate are our positive predictions?"
```

**Recall:** Of all actual positives, how many did we find?
```
Recall = TP / (TP + FN)
"How complete is our detection?"
```

**F1 Score:** Harmonic mean (balances precision and recall)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Example: Spam detection**
```
100 emails: 10 spam, 90 ham

Our classifier predicts: 8 as spam
- 6 are actually spam (TP = 6)
- 2 are actually ham (FP = 2)
- Missed 4 spam (FN = 4)

Precision = 6 / (6 + 2) = 6/8 = 75%
(Of 8 predicted spam, 6 are correct)

Recall = 6 / (6 + 4) = 6/10 = 60%
(Of 10 actual spam, caught 6)

F1 = 2 × (0.75 × 0.60) / (0.75 + 0.60) = 0.67
```

**Trade-offs:**

**High Precision, Low Recall:**
```
Only flag as spam if very confident
→ Few false positives (ham labeled as spam)
→ But miss many spam emails
```

**High Recall, Low Precision:**
```
Flag anything suspicious as spam
→ Catch most spam
→ But many ham emails incorrectly flagged
```

**F1 balances both:**
- Good when you care equally about precision and recall
- Penalizes extreme trade-offs

### Q6: What is Macro vs Micro vs Weighted F1?
**A:** Multi-class classification needs to aggregate scores across classes.

**Example: 3-class sentiment (Positive, Neutral, Negative)**

**Macro F1: Average of per-class F1**
```
F1(Positive) = 0.90
F1(Neutral) = 0.60
F1(Negative) = 0.85

Macro F1 = (0.90 + 0.60 + 0.85) / 3 = 0.78
```
- **Treats all classes equally** (even rare ones)
- Good when all classes matter equally
- Sensitive to poor performance on rare classes

**Micro F1: Pool all decisions**
```
Total TP, FP, FN across all classes
Then compute F1 from pooled counts

Micro F1 = 2 × (Micro_P × Micro_R) / (Micro_P + Micro_R)
```
- **Weighted by class frequency**
- Dominated by common classes
- Equals accuracy in multi-class

**Weighted F1: Average weighted by class support**
```
Weighted F1 = (0.90 × 100 + 0.60 × 500 + 0.85 × 100) / 700
(If Positive=100, Neutral=500, Negative=100 examples)
```
- **Accounts for class imbalance**
- Most common choice in practice

**When to use each:**
- **Macro:** All classes equally important (rare disease detection)
- **Micro:** Care about overall accuracy
- **Weighted:** Most realistic (accounts for imbalance)

### Q7: What is a Confusion Matrix and how to interpret it?
**A:** Confusion Matrix shows exact breakdown of predictions vs true labels.

**Binary classification:**
```
                Predicted
                Neg    Pos
Actual  Neg     TN     FP
        Pos     FN     TP
```

**Multi-class example: Sentiment (Pos/Neu/Neg)**
```
               Predicted
               Pos  Neu  Neg
Actual  Pos    85   10    5
        Neu    15   60   25
        Neg     5   15   80

Interpretation:
- Positive class: 85 correct, 10 confused as Neutral, 5 as Negative
- Main confusion: Neutral ↔ Negative (25 + 15 = 40 errors)
- Positive class performs best (85% accuracy)
```

**What to look for:**

**1. Diagonal (correct predictions):**
```
Higher values on diagonal = better
```

**2. Systematic confusions:**
```
If Negative often predicted as Neutral:
→ Model has weak negative indicators
→ Need better negative features
```

**3 Class-specific performance:**
```
If one class has low diagonal value:
→ Needs more training data
→ Or better features
```

**Python:**
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Pos', 'Neu', 'Neg'])
disp.plot()
```

### Q8: What are BLEU and ROUGE scores?
**A:** BLEU and ROUGE measure quality of generated text against reference texts.

**BLEU (Bilingual Evaluation Understudy):**
- **Use case:** Machine translation, text generation
- **Idea:** How many n-grams in generated text match reference?

**Formula (simplified):**
```
BLEU = Precision of n-grams in generated text

For unigrams (1-gram):
Generated: "the cat sat"
Reference: "the cat sat on mat"

Unigram precision = 3/3 = 100% (all words in reference)
```

**Higher order n-grams:**
```
BLEU-4 considers unigrams, bigrams, trigrams, 4-grams
Perfect match with reference = 1.0
No overlap = 0.0
```

**Problems:**
- ❌ Only measures precision (not recall)
- ❌ Doesn't capture meaning, just word overlap
- ❌ Multiple references needed for reliability

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**
- **Use case:** Text summarization
- **Idea:** How much of reference is covered by generated text?

**ROUGE-N: N-gram recall**
```
Generated summary: "the cat sat"
Reference summary: "the cat sat on the mat"

ROUGE-1 (unigrams) = 3/6 = 50% (3 of 6 reference words covered)
```

**ROUGE-L: Longest Common Subsequence**
```
Finds longest sequence of matching words (order matters, but gaps allowed)

Generated: "cat sat mat"
Reference: "the cat sat on the mat"
LCS: "cat sat mat" (length 3)
ROUGE-L = 3/6 = 50%
```

**When to use:**
- **BLEU:** Translation quality
- **ROUGE:** Summarization quality
- Both require reference texts (gold standard)

### Q9: How do you evaluate Named Entity Recognition?
**A:** NER evaluation is tricky because predictions are sequences with boundaries.

**Token-level evaluation:**
```
True:  [B-PER, I-PER, O, B-LOC]
Pred:  [B-PER, I-PER, O, B-ORG]

3 out of 4 tokens correct = 75% accuracy

Problem: Doesn't account for entity boundaries!
```

**Entity-level evaluation (standard):**

**Exact match:** Boundary AND type must match
```
True:  "Barack Obama" (PERSON), "New York" (LOCATION)
Pred:  "Barack" (PERSON), "New York" (ORGANIZATION)

TP: 0 (no exact matches!)
FP: 2 (both predictions wrong)
FN: 2 (missed both true entities)

Precision: 0/2 = 0%
Recall: 0/2 = 0%
F1: 0%
```

**Partial match variants:**

**1. Type match only:**
```
Pred: "New York" (GPE) for gold "New York City" (GPE)
Credit: Correct type, wrong boundary
```

**2. Boundary match only:**
```
Pred: "Barack Obama" (LOC) for gold "Barack Obama" (PER)
Credit: Correct span, wrong type
```

**Standard in research: Exact match F1**

**Python:**
```python
from seqeval.metrics import classification_report

# Use IOB format
y_true = [['B-PER', 'I-PER', 'O', 'B-LOC']]
y_pred = [['B-PER', 'I-PER', 'O', 'B-ORG']]

print(classification_report(y_true, y_pred))
# Gives entity-level precision, recall, F1 per entity type
```

### Q10: What are Coherence and Perplexity for topic models?
**A:**

**Perplexity: How well model predicts held-out documents**
```
Perplexity = exp(-log likelihood / word count)

Lower perplexity = better predictive power

Example:
Model A: Perplexity = 500
Model B: Perplexity = 300  ← Better at predicting text!
```

**Problem with perplexity:**
- Doesn't correlate well with human interpretability
- Can decrease with more topics (overfitting)
- Not meaningful in absolute terms

**Coherence: How semantically similar are top words in each topic?**

**Measuring coherence:**
```
Topic: {car, vehicle, auto, drive, road}

For each word pair, measure co-occurrence in documents:
- How often do "car" and "vehicle" appear in same document?
- Average across all pairs

High coherence → words frequently co-occur → interpretable topic
```

**Types of coherence:**

**C_V (best for LDA):**
- Based on word co-occurrence in sliding window
- Correlates well with human judgment

**U_Mass:**
- Based on document co-occurrence
- Faster to compute

**Example:**
```
Topic A: {election, vote, campaign, president}
→ High coherence (political terms co-occur)

Topic B: {cat, economy, blue, computer, vote}
→ Low coherence (unrelated terms)
```

**In practice:**
```python
from gensim.models.coherencemodel import CoherenceModel

coherence_model = CoherenceModel(
    model=lda_model,
    texts=texts,
    dictionary=dictionary,
    coherence='c_v'
)

coherence_score = coherence_model.get_coherence()
# Aim for 0.4+ for good topics
```

---

## Part C: Cross-Validation Strategies

### Q11: Why is standard K-Fold not always appropriate for text?
**A:** Text data has special characteristics that violate K-Fold assumptions.

**Problems:**

**1. Imbalanced classes:**
```
1000 documents: 900 negative, 100 positive

Random K-Fold might create:
Fold 1: 95 negative, 5 positive
Fold 2: 87 negative, 13 positive  ← Different distributions!

Model performance varies wildly across folds
```

**2. Temporal order (news, tweets, emails):**
```
Train on 2020 data, test on 2019 data ← Data leakage!
Model sees future information
```

**3. Grouped data (same author, same source):**
```
Train on paragraphs 1, 3, 5 from a document
Test on paragraphs 2, 4, 6 from same document

→ Train and test too similar!
→ Overly optimistic performance
```

**Solutions:**
- Stratified K-Fold (for imbalance)
- Time-based splits (for temporal data)
- Group K-Fold (for grouped data)

### Q12: What is Stratified K-Fold and when to use it?
**A:** Stratified K-Fold preserves class distribution in each fold.

**How it works:**
```
Dataset: 80% negative, 20% positive

Standard K-Fold (random):
Fold 1: 85% neg, 15% pos
Fold 2: 75% neg, 25% pos  ← Varying distributions

Stratified K-Fold:
Fold 1: 80% neg, 20% pos
Fold 2: 80% neg, 20% pos  ← Consistent!
All folds: 80% neg, 20% pos
```

**Why it matters:**
- More reliable performance estimates
- Reduces variance across folds
- Essential for imbalanced datasets

**Python:**
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Each fold maintains class distribution
```

**Use Stratified K-Fold when:**
- ✅ Classification tasks (almost always!)
- ✅ Imbalanced classes
- ✅ Multi-class with varying frequencies

**Don't use for:**
- ❌ Regression (no classes to stratify)
- ❌ Temporal data (use time-based splits)

### Q13: How do you handle temporal text data in cross-validation?
**A:** Use time-based splits to avoid data leakage from future → past.

**Problem with shuffle:**
```
Email timestamps:
Jan: "Obama elected"
Feb: "Obama president"
Mar: "Obama policy"

Random split:
Train: Jan, Mar
Test: Feb

Model learns "elected" → "president" from future (Mar)!
```

**Time-based split:**
```
Train: Jan, Feb (past)
Test: Mar (future)

Models what would happen in production: predict future unseen data
```

**Python:**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    # Ensures test_idx always > train_idx (temporally)
    X_train, X_test = X[train_idx], X[test_idx]
```

**Visualization:**
```
Fold 1: Train [----]     Test [--]
Fold 2: Train [--------] Test [--]
Fold 3: Train [------------] Test [--]
Fold 4: Train [----------------] Test [--]

Training set grows, test set is always future
```

**Best for:**
- News articles (predict tomorrow's topics)
- Social media (trending topics, sentiment shifts)
- Customer support (evolving issues)
- Email (spam patterns change over time)

### Q14: What is Group K-Fold and when do you need it?
**A:** Group K-Fold prevents data leakage when samples are grouped (same author, same conversation, same source).

**Problem:**
```
Dataset: Customer reviews, multiple reviews per customer

Standard K-Fold:
Train: Customer A's reviews 1, 3, 5
Test: Customer A's reviews 2, 4

→ Model learns Customer A's writing style
→ Overly optimistic performance!
```

**Group K-Fold solution:**
```
Keep all samples from same group together

Fold 1:
Train: Customers A, B, C
Test: Customer D

Fold 2:
Train: Customers A, B, D
Test: Customer C

No customer appears in both train and test!
```

**Python:**
```python
from sklearn.model_selection import GroupKFold

# Groups: which customer each review belongs to
groups = [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4]

gkf = GroupKFold(n_splits=3)

for train_idx, test_idx in gkf.split(X, y, groups=groups):
    # Guarantees no group overlap
    X_train, X_test = X[train_idx], X[test_idx]
```

**When to use:**
- Multiple documents from same author
- Conversation threads (group by conversation)
- Medical records (group by patient)
- Product reviews (group by reviewer)
- Multi-paragraph documents (group by document)

---

## Part D: Libraries and Tools

### Q15: NLTK vs spaCy: When to use which?
**A:**

**NLTK (Natural Language Toolkit):**

**Pros:**
- ✅ Educational: Great for learning NLP concepts
- ✅ Comprehensive: 50+ corpora, many algorithms
- ✅ Flexible: Access to low-level components
- ✅ Well-documented: Extensive tutorials

**Cons:**
- ❌ Slow: Not optimized for production
- ❌ Requires manual pipeline assembly
- ❌ Older codebase

**Best for:**
- Learning and education
- Prototyping and experiments
- When you need specific algorithms NLTK has

**spaCy:**

**Pros:**
- ✅ Fast: Cython-optimized, production-ready
- ✅ Batteries included: Pre-trained models
- ✅ Modern: Industrial-strength NLP
- ✅ Easy to use: Unified API

**Cons:**
- ❌ Less flexible: Harder to customize
- ❌ Fewer algorithms: Focused on production use cases
- ❌ Larger models (but more accurate)

**Best for:**
- Production systems
- Real-time processing
- Pre-trained NER, POS tagging
- When speed matters

**Decision matrix:**

| Use Case | NLTK | spaCy |
|----------|------|-------|
| **Learning NLP** | ✅ | ❌ |
| **Production NER** | ❌ | ✅ |
| **Custom algorithms** | ✅ | ❌ |
| **Speed critical** | ❌ | ✅ |
| **POS tagging** | ⚠️ | ✅ |
| **Sentiment (lexicon)** | ✅ | ⚠️ |

### Q16: What about Gensim and when to use it?
**A:** Gensim specializes in topic modeling and word embeddings.

**Best features:**

**1. Topic Modeling:**
```python
from gensim.models import LdaModel

lda = LdaModel(corpus, num_topics=10)
```
- Best LDA implementation
- Handles large corpora (streaming)
- Coherence metrics built-in

**2. Word Embeddings:**
```python
from gensim.models import Word2Vec

model = Word2Vec(sentences, vector_size=100)
```
- Word2Vec, FastText, Doc2Vec
- Load pre-trained embeddings easily

**3. Similarity queries:**
```python
from gensim.similarities import MatrixSimilarity

index = MatrixSimilarity(corpus)
sims = index[query]  # Fast similarity search
```

**When to use Gensim:**
- ✅ Topic modeling (LDA, LSA, NMF)
- ✅ Word embeddings (Word2Vec, Doc2Vec, FastText)
- ✅ Document similarity at scale
- ✅ Large corpora (memory-efficient streaming)

**When NOT to use:**
- ❌ Text classification (use scikit-learn)
- ❌ NER, POS tagging (use spaCy)
- ❌ Preprocessing (use NLTK or spaCy)

**Perfect combo:**
```
spaCy: Preprocessing, POS, NER
↓
Gensim: Topic modeling, embeddings
↓
scikit-learn: Classification, clustering
```

### Q17: How do you save and load models for production?
**A:**

**scikit-learn (Joblib):**
```python
from joblib import dump, load

# Save
dump(pipeline, 'model.joblib')

# Load
pipeline = load('model.joblib')
predictions = pipeline.predict(new_data)
```

**spaCy:**
```python
# Save
nlp.to_disk('/path/to/model')

# Load
import spacy
nlp = spacy.load('/path/to/model')
```

**Gensim:**
```python
# Save
model.save('lda_model')

# Load
from gensim.models import LdaModel
model = LdaModel.load('lda_model')
```

**Best practices:**
1. **Version your models:** `model_v1.2.3.joblib`
2. **Save preprocessing artifacts too:** Vectorizers, encoders, etc.
3. **Document dependencies:** scikit-learn version, Python version
4. **Test loaded model:** Ensure it produces same predictions
5. **Include metadata:** Date trained, metrics, dataset info

---

## 🎯 Key Takeaways

1. **Pipelines prevent data leakage** - fit on train, transform on test automatically
2. **Custom transformers** extend pipelines with domain logic
3. **F1 score** crucial for imbalanced text classification
4. **Confusion matrix** reveals systematic errors
5. **BLEU for generation, ROUGE for summarization**
6. **Entity-level F1** for NER (not token-level)
7. **Coherence > Perplexity** for topic model quality
8. **Stratified K-Fold** for classification (preserves class balance)
9. **Time-based splits** for temporal data (avoid future leakage)
10. **Group K-Fold** prevents author/source leakage
11. **spaCy for production, NLTK for learning, Gensim for topics/embeddings**
12. **Always save entire pipeline**, not just model
