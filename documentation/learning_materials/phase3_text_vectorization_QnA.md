# Phase 3: Text Vectorization - Q&A

> **Goal:** Convert text into numerical "vectors" that Machine Learning models can understand.

---

## Part A: Traditional Count-Based Methods

### Q1: Why can't machine learning models work directly with text?
**A:** ML models require numerical input because:
- They perform mathematical operations (addition, multiplication, gradients)
- Need fixed-size inputs (but documents have varying lengths)
- Need to compute distances/similarities between inputs
- Optimization algorithms work in vector spaces

**The vectorization challenge:**
- "cat" and "dog" are both animals (semantically similar)
- But as strings, they share no characters
- Need to convert to numbers that preserve meaning

### Q2: What is Bag of Words (BoW) and when should you use it?
**A:** BoW represents text as a vector of word counts, ignoring grammar and word order.

**How it works:**
```
Vocabulary: ["cat", "dog", "sat", "mat", "on", "the"]

Document 1: "the cat sat on the mat"
→ [2, 1, 0, 1, 1, 1, 1]  (counts: the=2, cat=1, dog=0, sat=1, mat=1, on=1)

Document 2: "the dog sat on the mat"
→ [2, 0, 1, 1, 1, 1, 1]
```

**Advantages:**
- ✅ Simple and interpretable
- ✅ Fast to compute
- ✅ Works well for short texts (tweets, product reviews)
- ✅ Good baseline for many tasks

**Disadvantages:**
- ❌ Loses word order: "dog bites man" = "man bites dog"
- ❌ Loses context: "not good" = "good"
- ❌ High dimensionality (vocabulary size)
- ❌ Sparse vectors (most values are 0)
- ❌ Frequent words dominate

**When to use:**
- Text classification with simple vocabulary
- Spam detection
- Document clustering
- As baseline before trying complex methods

### Q3: What is TF-IDF and why is it better than simple counts?
**A:** TF-IDF (Term Frequency - Inverse Document Frequency) weights words by their importance.

**Problem with BoW:**
- "the" appears 1000 times → huge weight
- "quantum" appears 1 time → tiny weight
- But "quantum" is more informative!

**TF-IDF solution:**
- **TF (Term Frequency):** How often term appears in document
- **IDF (Inverse Document Frequency):** How rare term is across all documents

**Formula:**
```
TF-IDF(term, doc) = TF(term, doc) × IDF(term)

IDF(term) = log(Total documents / Documents containing term)
```

**Example:**
```
Corpus: 1000 documents
- "the" appears in 999 documents → IDF ≈ log(1000/999) ≈ 0.001 (low)
- "quantum" appears in 10 documents → IDF ≈ log(1000/10) ≈ 2.0 (high)

If both appear 5 times in a document:
- TF-IDF("the") = 5 × 0.001 = 0.005
- TF-IDF("quantum") = 5 × 2.0 = 10.0
```

**What TF-IDF captures:**
- High TF-IDF: Term frequent in this document, rare in corpus (discriminative!)
- Low TF-IDF: Term infrequent in this document OR common in corpus (not discriminative)

**When to use:**
- Document search and retrieval
- Document classification
- Feature engineering for ML
- Topic modeling preprocessing

**Still the gold standard for classical NLP!**

### Q4: What are N-grams and why do they matter?
**A:** N-grams are contiguous sequences of N words, capturing local word patterns.

**Types:**
- **Unigrams (1-gram):** Single words ["the", "cat", "sat"]
- **Bigrams (2-gram):** Word pairs ["the cat", "cat sat", "sat on"]
- **Trigrams (3-gram):** Word triples ["the cat sat", "cat sat on"]

**Why they matter:**

**1. Capture word order (locally):**
```
Without bigrams:
"not good" → ["not", "good"] (same as "good, not bad")

With bigrams:
"not good" → ["not", "good", "not good"] (captures negation!)
```

**2. Capture phrases:**
```
"New York" as bigram is different from "New" and "York" separately
"machine learning" is different from "machine" + "learning"
```

**3. Improve sentiment analysis:**
```
"not bad" → negative + negative = actually POSITIVE!
Bigram "not bad" can be learned as positive phrase
```

**Trade-offs:**
- **Bigrams:** 10x more features, moderate improvement
- **Trigrams:** 100x more features, diminishing returns
- **4-grams+:** Usually not worth it (too sparse, overfitting)

**Best practice:** Use unigrams + bigrams for most tasks.

### Q5: What's the curse of dimensionality in text vectorization?
**A:** As vocabulary grows, vectors become high-dimensional and sparse, causing multiple problems.

**Example:**
```
10,000 unique words in corpus
→ Each document is a 10,000-dimensional vector
→ Most values are 0 (sparse)
→ Only 20-50 words per document typically
```

**Problems:**

**1. Memory explosion:**
- 1M documents × 10K features × 8 bytes = 80 GB!
- With bigrams: 100M features possible → 800 TB!

**2. Computational cost:**
- Matrix operations slow on large sparse matrices
- Training time increases quadratically

**3. Overfitting:**
- More features than samples → model memorizes training data
- Rare words become noise

**4. Sparsity:**
- Most word co-occurrences never observed
- Hard to estimate probabilities accurately

**Solutions:**
- **Feature selection:** Keep top N most informative features
- **Dimensionality reduction:** PCA, SVD, LDA
- **Min/Max document frequency:** `min_df=5, max_df=0.8`
- **Sparse matrix formats:** Only store non-zero values
- **Hashing vectorizer:** Fixed-size hash table

### Q6: What is the Hashing Vectorizer and when should you use it?
**A:** Hashing Vectorizer uses a hash function to map words to a fixed-size vector space.

**Traditional vectorization problem:**
```
1. Need to store vocabulary dictionary
2. Vocabulary grows with data
3. Requires two passes: one to build vocab, one to vectorize
```

**Hashing approach:**
```python
hash("cat") % 10000 → 4523
hash("dog") % 10000 → 8721

# No vocabulary storage needed!
# Fixed size (10000 in this example)
# Single pass over data
```

**Advantages:**
- ✅ **Online learning:** Can process streaming data
- ✅ **Low memory:** No vocabulary dictionary stored
- ✅ **Fixed size:** Only need to specify n_features
- ✅ **Fast:** Single pass, no dictionary lookup

**Disadvantages:**
- ❌ **Hash collisions:** Different words → same index
- ❌ **Not invertible:** Can't map back to words
- ❌ **No IDF weighting:** Just counts (though can be added)

**When to use:**
- Very large datasets that don't fit in memory
- Online/streaming learning scenarios
- When vocabulary is extremely large
- Don't need to interpret individual features

**Example use case:**
```
Processing 100M tweets:
- Traditional: Need to store millions of words in vocabulary
- Hashing: Fix size at 2^20 features, process in constant memory
```

---

## Part B: Word Embeddings (Pre-Transformer Era)

### Q7: What are word embeddings and how do they differ from count-based methods?
**A:** Word embeddings are dense, low-dimensional, learned representations where semantically similar words have similar vectors.

**Count-based (BoW, TF-IDF):**
```
Vocabulary size: 10,000 words
Vector dimension: 10,000
Sparsity: 99.5% zeros
Relationship: "cat" and "dog" are orthogonal (no similarity)

Vector for "cat": [0, 0, 0, ..., 1, ..., 0, 0]  (one 1, rest zeros)
```

**Embeddings (Word2Vec, GloVe):**
```
Vector dimension: 300 (dense!)
All values non-zero
Similarity: cosine("cat", "dog") = 0.8 (highly similar)

Vector for "cat": [0.2, -0.5, 0.8, 0.1, ...]  (all meaningful values)
Vector for "dog": [0.3, -0.4, 0.7, 0.2, ...]  (similar to cat!)
```

**Key differences:**

| Aspect | Count-Based | Embeddings |
|--------|-------------|------------|
| **Dimensionality** | Vocabulary size (10K-100K) | Fixed (50-300) |
| **Density** | Sparse (99%+ zeros) | Dense (all non-zero) |
| **Semantics** | No similarity captured | Semantic similarity |
| **Learning** | Counting (unsupervised) | Neural network (unsupervised) |
| **Interpretability** | High (features = words) | Low (dimensions abstract) |

### Q8: How does Word2Vec work?
**A:** Word2Vec learns embeddings by predicting context words from target words (Skip-gram) or vice versa (CBOW).

**Two architectures:**

**1. Skip-gram:** Predict context from word
```
Input: "cat"
Task: Predict words around "cat"

"The cat sat on the mat"
     ↑
Given "cat", predict: ["the", "sat", "on"]

If "dog" predicts similar contexts → "cat" and "dog" get similar embeddings!
```

**2. CBOW (Continuous Bag of Words):** Predict word from context
```
Input: ["the", "_", "sat", "on"]
Task: Predict the missing word

Answer: "cat" (or "dog")

If "cat" and "dog" appear in similar contexts → similar embeddings!
```

**Key insight: Distributional hypothesis**
> "You shall know a word by the company it keeps"
> — J.R. Firth

Words appearing in similar contexts have similar meanings.

**Training:**
- Shallow neural network (1 hidden layer)
- Hidden layer weights become the embeddings
- Trained on large corpus (Wikipedia, Google News)

**Tricks for efficiency:**
- **Negative sampling:** Don't update all words, just positive + random negatives
- **Subsampling frequent words:** Downsample "the", "is", etc.
- **Hierarchical softmax:** Faster than full softmax

### Q9: What is GloVe and how does it differ from Word2Vec?
**A:** GloVe (Global Vectors) combines count-based and prediction-based methods.

**Word2Vec approach:**
- Local context windows
- Predicts one context word at a time
- Never looks at global statistics

**GloVe approach:**
1. **Build co-occurrence matrix:** Count how often words appear together
2. **Factorize matrix:** Find vectors that reconstruct co-occurrence
3. **Optimize:** Minimize difference between dot product and log of co-occurrence

**Example:**
```
"ice" and "solid" co-occur 100 times
"ice" and "cold" co-occur 150 times

GloVe learns vectors where:
vec(ice) · vec(solid) ≈ log(100)
vec(ice) · vec(cold) ≈ log(150)
```

**Advantages over Word2Vec:**
- Uses global corpus statistics (more information)
- Often better performance on word analogy tasks
- Deterministic (same corpus → same result)

**Word2Vec advantages:**
- Faster to train on large corpora
- Online learning possible
- Better for rare words (GloVe needs co-occurrence counts)

**In practice:** Both work well, choice is often based on available pre-trained models.

### Q10: What is FastText and how does it handle unknown words?
**A:** FastText extends Word2Vec by representing words as bags of character n-grams.

**Word2Vec problem:**
```
Vocabulary: {"cat", "dog", "mouse"}
New word: "cats"
Word2Vec: Unknown! No embedding available!
```

**FastText solution:**
```
"cats" = <ca + cat + ats + cats>  (character n-grams)

If seen similar words:
- "cat" = <ca + cat + at>
- "catch" = <ca + cat + atc + tch + atch>

Can construct "cats" from "cat" substrings!
```

**Advantages:**
- ✅ **Handle OOV (out-of-vocabulary) words:** Construct from subwords
- ✅ **Better for rare words:** Share substrings with common words
- ✅ **Morphology-aware:** "teach", "teacher", "teaching" related
- ✅ **Great for morphologically rich languages:** German, Turkish, Finnish

**Example:**
```
"unbelievable" = un + believe + able

Even if "unbelievable" is rare:
- "un-" appears in "unhappy", "unclear"
- "believe" is common
- "-able" appears in "readable", "doable"

FastText combines these → good embedding!
```

**Disadvantages:**
- Slower to train and use (more computations)
- Larger model size
- Less interpretable

**When to use:**
- Rare words in your domain
- Morphologically rich languages
- User-generated content (typos, slang)
- Medical/scientific text (compound words)

### Q11: What are the famous word embedding analogies?
**A:** Word embeddings capture semantic and syntactic relationships through vector arithmetic.

**Semantic analogies:**
```
king - man + woman ≈ queen

vec(king) - vec(man) + vec(woman) ≈ vec(queen)

This pattern works for:
- Paris - France + Germany ≈ Berlin
- big - bigger + small ≈ smaller
```

**Syntactic analogies:**
```
walk - walking + swim ≈ swimming

Present - gerund transformation
```

**Why this works:**
- "king" and "queen" differ mainly in gender
- "man" and "woman" encode the gender difference
- Subtracting "man" and adding "woman" performs gender transformation
- Embeddings learn these consistent relationships

**Limitations:**
- Not perfect (returns nearest neighbor, not exact)
- Biased: can encode societal biases
  - "doctor - man + woman ≈ nurse" (gender bias!)
- Only works for clear, consistent relationships

### Q12: How do you use pre-trained embeddings?
**A:** Pre-trained embeddings save training time and work better on small datasets.

**Popular pre-trained models:**
- **Word2Vec:** Google News (3M words, 300 dimensions)
- **GloVe:** Wikipedia + Gigaword (400K words, 50-300 dim)
- **FastText:** Common Crawl (2M words, 300 dim)

**How to use:**

**1. Load pre-trained embeddings:**
```python
import gensim.downloader as api

# Load pre-trained Word2Vec
model = api.load("word2vec-google-news-300")

# Get embedding for word
vec = model['cat']  # 300-dimensional vector

# Find similar words
similar = model.most_similar('cat', topn=5)
# Output: [('dog', 0.85), ('kitten', 0.82), ('feline', 0.78), ...]
```

**2. Convert documents to vectors:**
```python
def document_vector(doc, model):
    # Remove out-of-vocabulary words
    vectors = [model[word] for word in doc if word in model]
    
    # Average word vectors
    return np.mean(vectors, axis=0)

# Now use as features for ML
X = [document_vector(doc, model) for doc in documents]
```

**3. Fine-tuning (optional):**
```python
# Start with pre-trained, continue training on your data
model.build_vocab(your_sentences, update=True)
model.train(your_sentences, total_examples=len(your_sentences), epochs=5)
```

**Benefits:**
- Leverages knowledge from billions of words
- Works well with small training data
- Captures general language patterns

### Q13: How do you evaluate word embeddings?
**A:**

**Intrinsic evaluation (word-level):**

**1. Word similarity tasks:**
```
Human judgment: similarity("cat", "dog") = 8/10
Embedding: cosine_similarity(vec(cat), vec(dog)) = 0.78

Compare correlation with human judgments
```

**2. Word analogy tasks:**
```
king - man + woman = ?
Correct if nearest neighbor is "queen"

Accuracy on benchmark datasets:
- Google analogies (19K questions)
- BATS analogies
```

**3. Visualization:**
```
t-SNE or PCA to 2D
Check if semantic clusters form:
- Animals cluster together
- Countries cluster together
- Verbs cluster together
```

**Extrinsic evaluation (task-level):**

**Test on downstream tasks:**
- Text classification accuracy
- Named entity recognition F1
- Sentiment analysis accuracy

**Better embeddings → Better task performance**

**Which matters more?**
- Intrinsic: Quick check, doesn't always correlate with task performance
- Extrinsic: What actually matters, but slower to evaluate

**Rule of thumb:** Optimize for your specific task, not analogy benchmarks.

### Q14: How do you combine TF-IDF with word embeddings?
**A:** You can weight word embeddings by TF-IDF scores to emphasize important words.

**Problem with simple averaging:**
```python
doc_vec = mean([vec(w1), vec(w2), vec(w3), ...])

Problem: "the" and "quantum" weighted equally!
```

**TF-IDF weighted embeddings:**
```python
def tfidf_weighted_doc_vector(doc, model, tfidf_weights):
    vectors = []
    weights = []
    
    for word in doc:
        if word in model and word in tfidf_weights:
            vectors.append(model[word])
            weights.append(tfidf_weights[word])
    
    # Weighted average
    return np.average(vectors, weights=weights, axis=0)
```

**Effect:**
- Important words (high TF-IDF) contribute more
- Stopwords (low TF-IDF) contribute less
- Combines semantic similarity (embeddings) with discriminative power (TF-IDF)

**empirically:**
- Often performs better than simple averaging
- Especially good for document classification

---

## 5. Practical Considerations

### Q15: BoW/TF-IDF vs Embeddings: When to use which?
**A:**

**Use BoW/TF-IDF when:**
- ✅ Need interpretability (which words matter?)
- ✅ Small dataset (< 1000 documents)
- ✅ Simple task (spam detection, topic classification)
- ✅ Limited compute resources
- ✅ Vocabulary is well-defined and stable

**Use Embeddings when:**
- ✅ Need semantic understanding
- ✅ Handling synonyms ("car" = "automobile")
- ✅ Out-of-vocabulary words expected
- ✅ Smaller feature space needed (300 vs 10,000)
- ✅ Have pre-trained embeddings available

**Example comparison:**

**Task: Sentiment classification**
```
Review: "This phone is terrible!"

TF-IDF approach:
- Stores: "terrible" → negative
- Fails on: "horrible", "awful" (if not in training)

Embedding approach:
- Learns: negative emotion region in vector space
- "terrible", "horrible", "awful" all cluster nearby
- Generalizes better!
```

### Q16: What are common pitfalls with word embeddings?
**A:**

**1. Averaging loses information:**
```
"great movie" and "not a great movie"
Might average to similar vectors!
```

**2. Out-of-vocabulary words:**
```
Word2Vec/GloVe: Unknown words → skipped entirely
Solution: Use FastText or fallback strategy
```

**3. Polysemy (multiple meanings):**
```
"bank" (financial) vs "bank" (river)
→ Same embedding for both meanings!

Solution: Contextual embeddings (ELMo, BERT) - but that's beyond classical NLP
```

**4. Bias in embeddings:**
```
Trained on human text → learns human biases
"doctor" - "man" + "woman" ≈ "nurse"
"programmer" - "man" + "woman" ≈ "homemaker"

Serious issue for production systems!
```

**5. Domain mismatch:**
```
Pre-trained on news → poor on medical text
"positive" (sentiment) vs "positive" (test result)

Solution: Train domain-specific embeddings or fine-tune
```

### Q17: How do you handle documents of different lengths?
**A:**

**Problem:**
```
Document 1: 10 words → average 10 embeddings
Document 2: 1000 words → average 1000 embeddings

ML needs fixed-size inputs!
```

**Solutions:**

**1. Average pooling (most common):**
```python
doc_vec = np.mean([vec(word) for word in doc], axis=0)
```
- Simple, works reasonably well
- Loses length information

**2. Max pooling:**
```python
doc_vec = np.max([vec(word) for word in doc], axis=0)
```
- Captures most salient features
- Can be noisy

**3. Weighted average (TF-IDF):**
```python
doc_vec = np.average(vectors, weights=tfidf_scores, axis=0)
```
- Better than simple average
- Emphasizes important words

**4. Concatenate [average, max, min]:**
```python
doc_vec = np.concatenate([
    np.mean(vectors, axis=0),
    np.max(vectors, axis=0),
    np.min(vectors, axis=0)
])
```
- Richer representation (900-dim if embeddings are 300-dim)
- More features to learn from

**5. Doc2Vec (advanced):**
- Learn document-level embeddings directly
- Each document gets its own vector during training

### Q18: Memory and performance considerations?
**A:**

**Sparse matrices (BoW, TF-IDF):**
```python
from scipy.sparse import csr_matrix

# Only stores non-zero values
# 10,000 features, but only 50 non-zero per document
# Memory: 50 values vs 10,000 values → 200x savings!
```

**Dense embeddings:**
```python
# 300-dimensional vectors
# ALL values stored (no sparsity savings)
# But: Much smaller than 10K feature TF-IDF!

10K documents:
- TF-IDF: 10K docs × 100K features × 8 bytes = 8GB (sparse format smaller)
- Embeddings: 10K docs × 300 dim × 8 bytes = 24MB (dense)
```

**Performance tips:**

**For BoW/TF-IDF:**
- Use sparse matrix operations (scipy)
- Set max_features to limit vocabulary
- Use min_df and max_df to filter
- HashingVectorizer for massive scale

**For embeddings:**
- Load pre-trained once, reuse
- Use dimensionality reduction if needed (PCA)
- Batch process documents
- Consider approximate nearest neighbors (for similarity search)

---

## 🎯 Key Takeaways

1. **BoW and TF-IDF are interpretable and fast** - excellent baselines
2. **TF-IDF is the gold standard** for classical text vectorization
3. **N-grams capture local context** - use unigrams + bigrams
4. **Sparsity is a major challenge** - most values are zero
5. **Word embeddings capture semantics** - similar words have similar vectors
6. **Word2Vec, GloVe, FastText** are the classical embedding methods
7. **Pre-trained embeddings save time** - leverage large-scale training
8. **Choose method based on task** - interpretability vs semantics
9. **Combine methods when helpful** - TF-IDF weighted embeddings
10. **Always handle out-of-vocabulary words** - FastText or fallback strategies
