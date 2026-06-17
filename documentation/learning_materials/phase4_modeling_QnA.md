# Phase 4: Classical Modeling & Topic Discovery - Q&A

> **Goal:** Applying statistical models to the vectors created in Phase 3.

---

## Part A: Classification Models

### Q1: Why is Naive Bayes the "industry standard" for text classification?
**A:** Naive Bayes is incredibly fast, simple, and surprisingly effective for text despite its "naive" assumptions.

**Key advantages:**
- ⚡ **Speed:** Training is O(n) - linear in data size
- 📊 **Works with small data:** Needs fewer examples than other methods
- 🎯 **Handles high dimensionality:** 10,000+ features is no problem
- 💾 **Low memory:** Just stores probabilities, not data
- 🔍 **Interpretable:** Can see which words indicate which class

**The "naive" assumption:**
```
Assumes all features (words) are independent given the class.

P("spam" has "free" AND "money") = P("spam" has "free") × P("spam" has "money")

Obviously wrong! "free" and "money" often appear together in spam.
But... it works anyway!
```

**Why it works despite being "wrong":**
- High dimensionality makes independence less wrong
- We only care about ranking, not exact probabilities
- Errors often cancel out

**When Naive Bayes excels:**
- Spam detection (classic use case)
- Sentiment analysis (simple binary tasks)
- Topic classification
- Real-time classification (speed critical)

### Q2: How does Naive Bayes actually work for text?
**A:** Uses Bayes' theorem to compute probability of each class given the document.

**Bayes' Theorem:**
```
P(Class | Document) = P(Document | Class) × P(Class) / P(Document)

We want: P(spam | "free money!")
```

**Step-by-step:**

**1. Prior probability:** P(Class)
```
P(spam) = # spam emails / # total emails
P(ham) = # ham emails / # total emails
```

**2. Likelihood:** P(Document | Class)
```
P("free money" | spam) = P("free" | spam) × P("money" | spam)

P("free" | spam) = (count of "free" in spam + 1) / (total words in spam + vocab size)
(+1 is Laplace smoothing for unseen words)
```

**3. Prediction:**
```
For each class, compute: P(Class) × P(Document | Class)
Pick class with highest value
```

**Example:**
```
Training:
- Spam emails: "free money", "buy now", "free offer"
- Ham emails: "meeting agenda", "project update"

New email: "free project"

P(spam) × P("free"|spam) × P("project"|spam) = 0.5 × 0.6 × 0.01 = 0.003
P(ham) × P("free"|ham) × P("project"|ham) = 0.5 × 0.01 × 0.4 = 0.002

Prediction: SPAM (higher score)
```

**Variants:**
- **Multinomial NB:** For word counts (most common for text)
- **Bernoulli NB:** For binary presence/absence
- **Gaussian NB:** For continuous features (not typical for text)

### Q3: When should you use Logistic Regression or SVM instead of Naive Bayes?
**A:**

**Logistic Regression advantages:**
- ✅ **Better with correlated features:** Doesn't assume independence
- ✅ **Probabilistic output:** Get calibrated probabilities
- ✅ **Feature weights:** Know which words most important
- ✅ **Regularization:** L1/L2 to prevent overfitting
- ✅ **Still fast and interpretable**

**SVM (Support Vector Machine) advantages:**
- ✅ **High accuracy:** Often best for high-dimensional text
- ✅ **Robust to outliers:** Only cares about support vectors
- ✅ **Kernel trick:** Can capture non-linear patterns
- ✅ **Works well with limited data**

**Trade-offs:**

| Metric | Naive Bayes | Logistic Regression | SVM |
|--------|-------------|---------------------|-----|
| **Speed** | Fastest | Medium | Slower |
| **Accuracy** | Good | Better | Best |
| **Interpretability** | High | High | Medium |
| **Probability estimates** | Poor (uncalibrated) | Good | Possible (slower) |
| **Small data (<1K)** | Best | Good | Good |
| **Large data (>100K)** | Best | Good | Slower |

**Best practices:**
1. Start with Naive Bayes (baseline)
2. Try Logistic Regression (usually improves)
3. Try SVM with linear kernel (marginal gains)
4. Try SVM with RBF kernel (if you have time/compute)

### Q4: What about tree-based models (Random Forest) for text?
**A:** Random Forests work for text but aren't typically the first choice.

**Advantages:**
- ✅ **Feature importance:** Easily see which words matter
- ✅ **Handle non-linearity:** Capture complex patterns
- ✅ **No feature scaling needed:** Works with raw counts
- ✅ **Robust to outliers**

**Disadvantages:**
- ❌ **Slow on high-dimensional sparse data:** TF-IDF creates 10K+ features
- ❌ **Large model size:** Many trees × many splits
- ❌ **Less interpretable than linear models:** Hard to explain predictions

**When RF works well for text:**
- Combined with feature reduction (top 500 features)
- With embeddings (300 dense dimensions, not 10K sparse)
- Metadata-heavy tasks (text + user features + time features)

**Example use case:**
```
Predict product rating from review:
- Text features: TF-IDF (top 500 words)
- Metadata: Review length, user history, product category
- RF can capture interactions: "long review" + "but" → likely negative
```

---

## Part B: Sequence Models

### Q5: What are Hidden Markov Models (HMMs) and when are they used?
**A:** HMMs model sequences where we observe outputs but underlying states are hidden.

**Core idea:**
```
Hidden states: POS tags (NOUN, VERB, ADJ)
Observations: Words ("cat", "runs", "fast")

Task: Given words, infer hidden POS sequence
```

**Components:**

**1. Transition probabilities:** P(State_t | State_{t-1})
```
P(VERB | NOUN) = 0.3  (noun often followed by verb)
P(NOUN | VERB) = 0.4  (verb often followed by noun)
P(ADV | VERB) = 0.2   (adverb often after verb)
```

**2. Emission probabilities:** P(Word | State)
```
P("cat" | NOUN) = 0.01
P("cat" | VERB) = 0.0001  (rare)
```

**3. Inference:** Viterbi algorithm finds most likely state sequence
```
Words: "The cat runs fast"
Best POS: DET NOUN VERB ADV
```

**Applications:**
- POS tagging (traditional method)
- Speech recognition (phonemes → words)
- Gene sequence analysis

**Limitations:**
- Assumes Markov property (only previous state matters)
- Can't capture long-distance dependencies
- Requires labeled data to train
- Superseded by CRFs and neural methods for most tasks

### Q6: What are Conditional Random Fields (CRFs) and why are they better than HMMs?
**A:** CRFs are discriminative sequence models that fix HMM's main limitations.

**HMM problems:**

**1. Independence assumptions:**
```
HMM: Each observation independent given state
Problem: "New York" - both words should inform decision together
```

**2. Generative vs Discriminative:**
```
HMM: Models P(words, tags) - generates both
CRF: Models P(tags | words) - only what we need!
```

**CRF advantages:**

**1. Rich feature sets:**
```
For "Apple" (NER task), CRF can use:
- Word itself: "Apple"
- Previous word: "CEO"
- Next word: "announced"
- Capitalization: True
- Position: Start of sentence
- Word shape: Xxxxx
- In company gazetteer: True

HMM limited to just the word!
```

**2. No independence assumption:**
```
Features can look at multiple observations simultaneously
"New York" - both words inform entity decision
```

**3. Better performance:**
```
NER task:
- HMM: ~85% F1
- CRF: ~90-92% F1
- BiLSTM-CRF: ~95% F1 (modern)
```

**When to use CRFs:**
- Named Entity Recognition (classic choice)
- POS tagging (better than HMMs)
- Chunking (finding phrases)
- Information extraction from structured text

**Tools:**
```python
from sklearn_crfsuite import CRF

# Define features for each token
def word_features(sent, i):
    return {
        'word': sent[i],
        'is_capitalized': sent[i][0].isupper(),
        'prev_word': sent[i-1] if i > 0 else 'BOS',
        'next_word': sent[i+1] if i < len(sent)-1 else 'EOS',
    }

crf = CRF()
crf.fit(X_train, y_train)
```

---

## Part C: Topic Modeling

### Q7: What is topic modeling and what problems does it solve?
**A:** Topic modeling automatically discovers abstract "topics" in a collection of documents without labels.

**Problem:**
```
You have 10,000 news articles but no labels.
Want to organize them into themes: Politics, Sports, Technology, etc.
Manual reading is impossible.
```

**Topic modeling solution:**
```
Algorithm discovers patterns:
- Topic 1: "election", "vote", "president", "campaign" → Politics
- Topic 2: "game", "team", "player", "score" → Sports
- Topic 3: "data", "algorithm", "AI", "technology" → Tech
```

**Real-world applications:**
- **Content recommendation:** Group similar articles
- **Trend analysis:** What topics are growing over time?
- **Document organization:** Automatic categorization
- **Exploratory analysis:** Understand large corpus quickly
- **Feature engineering:** Topic distributions as ML features

**Unsupervised:** No manual labeling needed!

### Q8: How does LDA (Latent Dirichlet Allocation) work?
**A:** LDA assumes documents are mixtures of topics, and topics are mixtures of words.

**Generative story (how LDA imagines documents are created):**

```
For each document:
  1. Pick a distribution over topics (e.g., 70% Politics, 20% Economics, 10% Sports)
  
  For each word in document:
    2. Pick a topic according to distribution (e.g., Politics)
    3. Pick a word from that topic (e.g., "election")
```

**Example:**
```
Document: "The president announced new economic policy in campaign speech"

LDA might infer:
- "president", "campaign", "speech" from Politics topic
- "economic", "policy" from Economics topic
- Document is mix: 60% Politics + 40% Economics
```

**What LDA learns:**

**Topic-word distributions:**
```
Politics topic: {president: 0.05, election: 0.04, vote: 0.03, ...}
Sports topic: {game: 0.06, team: 0.05, player: 0.04, ...}
```

**Document-topic distributions:**
```
Doc 1: {Politics: 0.7, Economics: 0.2, Sports: 0.1}
Doc 2: {Politics: 0.1, Sports: 0.8, Economics: 0.1}
```

**Key hyperparameters:**
- **n_topics:** How many topics to discover (must choose manually)
- **alpha:** Document-topic density (higher = more topics per doc)
- **beta:** Topic-word density (higher = more words per topic)

### Q9: What is LSA (Latent Semantic Analysis) and how does it differ from LDA?
**A:** LSA uses SVD (Singular Value Decomposition) on term-document matrix to find latent topics.

**Approach:**

**1. Create term-document matrix:**
```
           Doc1  Doc2  Doc3  ...
president    5     0     2
economy      3     1     4
team         0    10     0
...
```

**2. Apply SVD:**
```
Matrix = U × Σ × V^T

U: Term-topic relationships (words → topics)
Σ: Importance of each topic
V: Document-topic relationships (docs → topics)
```

**3. Dimensionality reduction:**
```
Keep top k topics (e.g., k=100)
Reduces dimensionality while preserving most information
```

**LDA vs LSA:**

| Aspect | LDA | LSA |
|--------|-----|-----|
| **Method** | Probabilistic (Bayesian) | Linear algebra (SVD) |
| **Interpretability** | Better (word probabilities) | Harder (abstract dimensions) |
| **Scalability** | Slower | Faster |
| **Output** | Probabilities (sum to 1) | Real-valued weights |
| **Sparsity** | Can be sparse | Dense representations |

**When to use each:**
- **LDA:** Need interpretable topics, willing to wait
- **LSA:** Speed critical, dimensionality reduction main goal

### Q10: What is NMF(Non-negative Matrix Factorization) for topic modeling?
**A:** NMF factorizes term-document matrix into non-negative components, often yielding more interpretable topics than LSA.

**Core idea:**
```
V ≈ W × H

V: Term-document matrix (n_terms × n_docs)
W: Term-topic matrix (n_terms × n_topics)  ← Non-negative!
H: Topic-document matrix (n_topics × n_docs)  ← Non-negative!
```

**Why non-negativity helps:**
- Topics are additive combinations (no subtraction)
- More interpretable: "Politics topic = president + election + vote"
- Each feature (word) contributes positively or not at all

**Comparison:**

**LSA (can be negative):**
```
Topic = 0.5*president + 0.3*election - 0.4*sports - 0.2*game
(Hard to interpret negative weights!)
```

**NMF (non-negative):**
```
Topic = 0.5*president + 0.3*election + 0.1*vote
(Clear: topic is about these words, not absence of other words)
```

**Advantages of NMF:**
- ✅ More interpretable than LSA
- ✅ Parts-based representation (additive)
- ✅ Faster than LDA
- ✅ Deterministic (same input → same output)

**Disadvantages:**
- ❌ Less principled than LDA (no probabilistic interpretation)
- ❌ Harder to choose number of topics
- ❌ Sensitive to initialization

**When to use:**
```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=10, random_state=42)
doc_topics = nmf.fit_transform(tfidf_matrix)
topic_words = nmf.components_

# Get top words per topic
for topic_idx, topic in enumerate(topic_words):
    top_words = [words[i] for i in topic.argsort()[-10:]]
    print(f"Topic {topic_idx}: {top_words}")
```

### Q11: How do you choose the number of topics?
**A:** No perfect method, but several heuristics help.

**Methods:**

**1. Coherence score (best):**
```
Measures how semantically similar top words in a topic are

High coherence: {car, drive, vehicle, road, auto}
Low coherence: {car, happiness, algorithm, banana, politics}
```

```python
from gensim.models.coherencemodel import CoherenceModel

coherence_scores = []
for n_topics in range(5, 50, 5):
    lda = train_lda(n_topics)
    cm = CoherenceModel(model=lda, texts=texts, coherence='c_v')
    coherence_scores.append(cm.get_coherence())

# Pick n_topics with highest coherence
```

**2. Perplexity (LDA-specific):**
```
How well model predicts held-out documents
Lower perplexity = better

BUT: Doesn't always correlate with human interpretability!
```

**3. Topic diversity:**
```
Are topics distinct or redundant?
Measure overlap in top words across topics
```

**4. Elbow method:**
```
Plot metric vs n_topics
Look for "elbow" where gains diminish
```

**5. Domain knowledge:**
```
If organizing news: might know there are ~10 major categories
Medical records: might expect 20-30 disease categories
```

**Best practice:**
- Try range of values (e.g., 10, 20, 30, 40, 50)
- Use coherence score + manual inspection
- Pick interpretable topics over marginal metric improvements

---

## Part D: Similarity & Distance Metrics

### Q12: What is cosine similarity and why is it preferred for text?
**A:** Cosine similarity measures the angle between two vectors, ignoring magnitude.

**Formula:**
```
cosine_sim(A, B) = (A · B) / (||A|| × ||B||)

Dot product / (length of A × length of B)

Range: [-1, 1]
- 1: Identical direction
- 0: Orthogonal (unrelated)
- -1: Opposite direction
```

**Why perfect for text:**

**Problem with Euclidean distance:**
```
Doc A: "cat cat cat" → [3, 0, 0]
Doc B: "cat cat cat cat cat cat" → [6, 0, 0]

Euclidean distance: sqrt((6-3)² + 0² + 0²) = 3 (far apart!)

But they're the same document, just different lengths!
```

**Cosine similarity solution:**
```
Same docs:
cosine_sim([3, 0, 0], [6, 0, 0]) = 1.0 (identical!)

Focuses on proportion of words, not absolute counts
```

**Example:**
```
Doc A: "dog cat" → [1, 1, 0, 0]
Doc B: "dog dog cat cat" → [2, 2, 0, 0]
Doc C: "bird fish" → [0, 0, 1, 1]

cosine_sim(A, B) = 1.0 (same content, different length)
cosine_sim(A, C) = 0.0 (no shared words)
```

**Applications:**
- Document similarity
- Recommendation systems
- Duplicate detection
- Information retrieval

### Q13: When should you use Euclidean distance vs cosine similarity?
**A:**

**Use Cosine Similarity when:**
- ✅ **Document length varies:** Focus on content, not length
- ✅ **Sparse vectors:** TF-IDF, BoW (text is inherently sparse)
- ✅ **Proportions matter:** "50% politics, 50% sports" vs "60% politics, 40% sports"
- ✅ **High dimensions:** Less affected by curse of dimensionality

**Use Euclidean Distance when:**
- ✅ **Magnitude matters:** Dense embeddings where scale is meaningful
- ✅ **Same-length vectors:** All vectors normalized
- ✅ **Low dimensions:** 2D/3D space
- ✅ **Clustering:** K-means often uses Euclidean

**Example:**

**TF-IDF vectors (use cosine):**
```
Doc A (short): [0.5, 0.5, 0, 0]
Doc B (long): [0.5, 0.5, 0, 0]

Euclidean: Depends on normalization
Cosine: 1.0 (identical) ← Better!
```

**Word embeddings (either works):**
```
vec(cat): [0.2, 0.3, -0.1, ...]
vec(dog): [0.3, 0.2, -0.1, ...]

Both Euclidean and Cosine work well (vectors dense, similar scale)
```

### Q14: What is Jaccard similarity and when to use it?
**A:** Jaccard measures overlap between sets, ignoring frequency.

**Formula:**
```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|

Intersection / Union

Range: [0, 1]
- 0: No overlap
- 1: Identical sets
```

**Example:**
```
Doc A: "the cat sat on the mat" → {the, cat, sat, on, mat}
Doc B: "the dog sat on the rug" → {the, dog, sat, on, rug}

Intersection: {the, sat, on} = 3 words
Union: {the, cat, dog, sat, on, mat, rug} = 7 words

Jaccard = 3/7 ≈ 0.43
```

**Note: Ignores frequency!**
```
"cat cat cat cat" and "cat" have Jaccard = 1.0
(Both just contain "cat")
```

**When to use:**
- 🎯 **Short texts:** Tweets, product titles (frequency less meaningful)
- 🎯 **Binary presence:** Does document contain term? (yes/no)
- 🎯 **Vocabulary comparison:** What words do documents share?
- 🎯 **Deduplication:** Finding exact or near-duplicate documents
- 🎯 **Tag similarity:** Comparing sets of tags/labels

**Not ideal for:**
- Long documents (set-based view loses info)
- When frequency matters ("very very good" vs "very good")

### Q15: What is Levenshtein (Edit) Distance and its applications?
**A:** Levenshtein distance counts minimum edits (insert/delete/replace) to transform one string into another.

**Examples:**
```
"cat" → "hat"
- Replace 'c' with 'h': 1 edit
Distance = 1

"kitten" → "sitting"
- Replace 'k' with 's': kitten → sitten
- Replace 'e' with 'i': sitten → sittin
- Insert 'g': sittin → sitting
Distance = 3
```

**Applications:**

**1. Spell checking:**
```
User types: "recieve"
Dictionary: "receive"
Distance = 2 (swap 'ie' to 'ei')

Find closest dictionary word within distance 1-2
```

**2. Fuzzy matching:**
```
Database: "John Smith"
Query: "Jon Smith"
Distance = 1 → Likely same person!
```

**3. OCR error correction:**
```
Scanned: "Th1s" (OCR mistook 'i' for '1')
Original: "This"
Distance = 1
```

**4. DNA sequence alignment:**
```
Measuring similarity between genetic sequences
```

**5. Plagiarism detection:**
```
Finding slightly modified copies
```

**Normalized Levenshtein:**
```
distance / max(len(str1), len(str2))

Gives range [0, 1] for easier comparison
```

**Performance:**
```
O(m × n) time complexity (m, n = string lengths)
Can be slow for long strings

Optimizations:
- Stop if distance exceeds threshold
- Use BK-trees for multi-word search
```

**Python:**
```python
import Levenshtein

dist = Levenshtein.distance("cat", "hat")  # 1
ratio = Levenshtein.ratio("cat", "hat")    # 0.67 (normalized)
```

---

## 5. Putting It All Together

### Q16: How do you choose which model to use?
**A:**

**Decision tree:**

```
Do you have labels?
├─ YES → Classification
│   ├─ Need speed? → Naive Bayes
│   ├─ Need accuracy? → Logistic Regression / SVM
│   └─ Have metadata? → Random Forest
│
└─ NO → Unsupervised
    ├─ Find topics? → Topic Modeling
    │   ├─ Need interpretability? → LDA or NMF
    │   └─ Need speed? → LSA or NMF
    │
    └─ Find similar docs? → Similarity Metrics
        ├─ Vary-length docs? → Cosine Similarity
        ├─ Short texts/sets? → Jaccard Similarity
        └─ String matching? → Levenshtein Distance
```

**Is it a sequence labeling task?**
```
├─ POS tagging → CRF (or pre-trained spaCy)
├─ NER → CRF (or pre-trained spaCy)
└─ Chunking → CRF
```

### Q17: How do you evaluate topic models?
**A:**

**Automated metrics:**

**1. Coherence:**
```
Measures semantic similarity of top words in topics

For topic: {car, vehicle, drive, automobile, road}
Check if these words co-occur in documents

Higher coherence = better topics
```

**2. Perplexity (LDA):**
```
How surprised is model by held-out data?
Lower perplexity = better predictive power

BUT: May not align with human interpretability!
```

**Manual evaluation:**

**3. Top words inspection:**
```
Topic 1: {president, election, vote, campaign} ← Coherent!
Topic 2: {cat, economy, run, happy, government} ← Incoherent!
```

**4. Topic labeling:**
```
Can you assign meaningful labels to each topic?
If not, topics aren't interpretable
```

**5. Document inspection:**
```
Look at documents assigned to each topic
Do they actually belong together?
```

**6. Topic diversity:**
```
Are topics distinct or overlapping?
Too similar → reduce n_topics
Too diverse → increase n_topics
```

**Best practice:**
- Use coherence as automatic metric
- Always manually inspect top words
- Sample documents from each topic
- Iterate on n_topics based on interpretability

### Q18: How do you build a document recommendation system?
**A:**

**Approach 1: Content-based (similarity)**

```python
# 1. Vectorize documents
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
doc_vectors = vectorizer.fit_transform(documents)

# 2. Compute similarity
from sklearn.metrics.pairwise import cosine_similarity

def recommend(doc_id, top_n=5):
    # Get similarities to all other docs
    sims = cosine_similarity(doc_vectors[doc_id], doc_vectors)[0]
    
    # Get top-N most similar (excluding itself)
    similar_indices = sims.argsort()[-top_n-1:-1][::-1]
    
    return similar_indices

# "Users who read Doc 5 also liked..."
recommend(doc_id=5, top_n=5)
```

**Approach 2: Topic-based**

```python
# 1. Train topic model
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=20)
doc_topics = lda.fit_transform(tfidf_matrix)

# Each document represented by topic distribution

# 2. Find similar topic distributions
from sklearn.metrics.pairwise import cosine_similarity
topic_sims = cosine_similarity(doc_topics)

# Recommend docs with similar topic mixtures
```

**Approach 3: Hybrid**

```
Combine multiple signals:
- Content similarity (TF-IDF cosine)
- Topic similarity (LDA)
- Collaborative filtering (if you have user behavior)
- Metadata (same author, category, time period)

Final score = 0.4*content + 0.3*topic + 0.2*collab + 0.1*meta
```

---

## 🎯 Key Takeaways

1. **Naive Bayes is the baseline** - fast, simple, surprisingly effective
2. **Logistic Regression and SVM** - better accuracy, still interpretable
3. **CRFs beat HMMs** - for sequence labeling tasks (NER, POS)
4. **LDA is the standard topic model** - probabilistic, interpretable
5. **LSA for speed, NMF for interpretability** - alternatives to LDA
6. **Cosine similarity for documents** - handles varying lengths well
7. **Jaccard for sets, Levenshtein for strings** - task-specific choices
8. **Always establish baselines** - start simple, add complexity as needed
9. **Evaluation strategy matters** - metrics should match your goals
10. **Combine methods strategically** - hybrid approaches often work best
