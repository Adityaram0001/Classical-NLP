# Phase 6: Real-World Applications - Q&A

> **Goal:** Applying classical NLP techniques to solve practical problems.

---

## Part A: Sentiment Analysis

### Q1: What's the difference between lexicon-based and machine learning sentiment analysis?
**A:**

**Lexicon-based approach:**
- Uses pre-built dictionaries of words with sentiment scores
- Rule-based: No training needed
- Fast and interpretable

**Example (VADER):**
```
"amazing" → +3.1 (very positive)
"good" → +1.9 (positive)
"bad" → -2.5 (negative)
"terrible" → -3.1 (very negative)

Text: "This movie is amazing and good"
Score: (+3.1) + (+1.9) = +5.0 → POSITIVE
```

**Machine Learning approach:**
- Learns from labeled training data
- Uses features (TF-IDF, embeddings)
- Can capture domain-specific patterns

**Example:**
```
Training: 1000 reviews labeled positive/negative
Learn: "unpredictable" is positive for movies, negative for products
```

**Comparison:**

| Aspect | Lexicon-Based | ML-Based |
|--------|---------------|----------|
| **Training data** | Not needed | Requires labeled data |
| **Domain adaptation** | Poor | Excellent |
| **Speed** | Very fast | Slower |
| **Accuracy** | 70-75% | 80-90%+ |
| **Interpretability** | High | Medium |
| **Handles new words** | Poor | Better (if in training) |

**When to use lexicon:**
- ✅ No training data available
- ✅ General sentiment (social media, reviews)
- ✅ Need real-time processing
- ✅ Want interpretable scores

**When to use ML:**
- ✅ Have labeled data
- ✅ Domain-specific language
- ✅ Need highest accuracy
- ✅ Can afford training time

### Q2: How does VADER work and when is it appropriate?
**A:** VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based tool specialized for social media text.

**Key features:**

**1. Lexicon with intensity:**
```
"good" → +1.9
"great" → +2.5
"fantastic" → +3.0
```

**2. Handles modifiers:**
```
"good" → +1.9
"very good" → +2.3 (intensifier boost)
"not good" → -1.9 (negation flip)
"GOOD" → +2.3 (ALL CAPS boost)
"good!!!" → +2.2 (punctuation emphasis)
```

**3. Emoji and emoticon aware:**
```
":)" → +1.5
"😊" → +1.5
":(" → -1.5
```

**4. Compound score:**
```
Returns: {
    'pos': 0.75,     # Positive proportion
    'neu': 0.20,     # Neutral proportion
    'neg': 0.05,     # Negative proportion
    'compound': 0.85 # Overall score [-1, +1]
}

Use compound for classification:
  compound ≥ 0.05 → Positive
  compound ≤ -0.05 → Negative
  Else → Neutral
```

**Example:**
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

texts = [
    "This is GREAT!!!",
    "This is not good",
    "This is good but expensive"
]

for text in texts:
    scores = analyzer.polarity_scores(text)
    print(f"{text}: {scores['compound']}")
```

**Best for:**
- Social media (Twitter, Reddit)
- Product reviews
- Customer feedback
- When you have no training data

**Limitations:**
- No domain adaptation
- Doesn't learn from data
- Fixed rules (can't improve)
- Limited context understanding

### Q3: What is aspect-based sentiment analysis?
**A:** Aspect-based sentiment identifies sentiment towards specific features/aspects of an entity.

**Problem with document-level sentiment:**
```
Review: "Food was excellent but service was terrible"

Overall sentiment: Mixed/Neutral
→ Loses nuance!
```

**Aspect-based solution:**
```
Aspects identified:
- Food: POSITIVE ("excellent")
- Service: NEGATIVE ("terrible")

More actionable insights!
```

**How it works:**

**Step 1: Aspect extraction**
```
Identify aspects (nouns/noun phrases):
- "battery life", "screen", "camera", "price"
```

**Step 2: Opinion extraction**
```
Find opinion words (adjectives) near aspects:
- "battery life is great"
- "screen is terrible"
```

**Step 3: Sentiment assignment**
```
Determine sentiment of opinion towards aspect:
- battery life → great → POSITIVE
- screen → terrible → NEGATIVE
```

**Classical approach:**
```python
# 1. Dependency parsing to find aspect-opinion pairs
doc = nlp("The battery life is great")

# Parse finds: "battery life" ← nsubj ← "is" → "great"
# Aspect: "battery life"
# Opinion: "great"

# 2. Look up sentiment of opinion word
sentiment = lexicon["great"]  # → POSITIVE

# 3. Assign to aspect
aspects["battery life"] = POSITIVE
```

**Applications:**
- Product reviews (Amazon, Yelp)
- Hotel reviews (Booking.com)
- Restaurant reviews
- Customer feedback analysis

**Output example:**
```
Review: "Room was spacious and clean but WiFi was slow and breakfast was overpriced"

Aspects:
✅ Room: POSITIVE (spacious, clean)
❌ WiFi: NEGATIVE (slow)
❌ Breakfast: NEGATIVE (overpriced)
```

---

## Part B: Text Summarization

### Q4: What's the difference between extractive and abstractive summarization?
**A:**

**Extractive summarization:**
- **Selects** existing sentences from document
- No new text generated
- Classical NLP can do this

**Example:**
```
Original (3 paragraphs):
[1] "The president announced a new policy today..."
[2] "The policy focuses on healthcare reform..."
[3] "Critics have raised concerns about cost..."

Extractive summary:
Sentences [1] and [2] (selected, not modified)
```

**Abstractive summarization:**
- **Generates** new sentences
- Paraphrases and reformulates
- Requires neural models (beyond classical NLP)

**Example:**
```
Original: "The president announced a new healthcare policy aimed at reform, though critics worry about costs"

Abstractive summary:
"President unveils controversial healthcare reform plan"
(New sentence, not in original!)
```

**Classical NLP can do extractive, not abstractive.**

### Q5: How does TextRank work for extractive summarization?
**A:** TextRank adapts Google's PageRank algorithm to rank sentence importance.

**Algorithm:**

**Step 1: Build sentence graph**
```
Nodes: Sentences
Edges: Similarity between sentences (cosine similarity)

         S1
        /  \
      0.8   0.3
      /      \
    S2 ---- S3
        0.6
```

**Step 2: Run PageRank**
```
Iteratively compute importance scores:
- Important sentences link to other important sentences
- Scores converge after iterations

Scores:
S1: 0.45 (highest)
S2: 0.35
S3: 0.20 (lowest)
```

**Step 3: Select top-K sentences**
```
Take top 3 sentences → Summary
Return in original document order (not rank order)
```

**Implementation:**
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def textrank_summarize(sentences, top_n=3):
    # 1. Vectorize sentences
    vectors = vectorizer.fit_transform(sentences)
    
    # 2. Build similarity matrix
    sim_matrix = cosine_similarity(vectors)
    
    # 3. Apply PageRank
    scores = np.array([1.0] * len(sentences))
    for _ in range(50):  # Iterations
        new_scores = 0.15 + 0.85 * sim_matrix.T @ scores
        if np.allclose(scores, new_scores):
            break
        scores = new_scores
    
    # 4. Get top sentences
    top_indices = scores.argsort()[-top_n:][::-1]
    top_indices = sorted(top_indices)  # Original order
    
    return [sentences[i] for i in top_indices]
```

**Advantages:**
- ✅ Unsupervised (no training data)
- ✅ Domain-independent
- ✅ Preserves original phrasing (no errors introduced)

**Limitations:**
- ❌ Just selection (can be choppy)
- ❌ No compression (long sentences stay long)
- ❌ May select redundant sentences

### Q6: What is LSA-based summarization?
**A:** LSA (Latent Semantic Analysis) finds key sentences using topic decomposition.

**Approach:**

**Step 1: Create term-sentence matrix**
```
           S1   S2   S3   S4
president  5    0    2    0
policy     3    4    0    1
healthcare 2    5    0    0
...
```

**Step 2: Apply SVD**
```
Matrix = U × Σ × V^T

V: Sentence-topic matrix
Each sentence represented by topic weights
```

**Step 3: Select representative sentences**
```
For each topic, pick sentence with highest weight
Ensures coverage of all major topics
```

**Example:**
```
Topic 1 (Politics): S1 (0.8), S2 (0.3)
Topic 2 (Healthcare): S2 (0.9), S4 (0.4)
Topic 3 (Economy): S3 (0.7)

Select: S1 (Topic 1), S2 (Topic 2), S3 (Topic 3)
→ Summary covers all topics
```

**Advantages:**
- ✅ Topic-aware (covers all themes)
- ✅ Reduces redundancy
- ✅ Mathematically principled

**Disadvantages:**
- ❌ Sensitive to topic number
- ❌ May select strange sentences if topics unclear

### Q7: How do you evaluate summarization quality?
**A:**

**Automatic metrics (need reference summaries):**

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**

**ROUGE-1: Unigram overlap**
```
Generated: "The president announced new policy"
Reference: "President announces healthcare policy reform"

Shared unigrams: {president, policy}
ROUGE-1 = 2/4 = 0.5 (recall)
```

**ROUGE-2: Bigram overlap**
```
Generated bigrams: [the president, president announced, announced new, new policy]
Reference bigrams: [president announces, announces healthcare, healthcare policy, policy reform]

Overlap: 0
ROUGE-2 = 0
```

**ROUGE-L: Longest common subsequence**
```
Finds longest matching word sequence (gaps allowed)

Generated: "president announced policy"
Reference: "president announces healthcare policy reform"
LCS: "president ... policy" (length 2)
```

**Manual evaluation:**

**1. Content coverage:**
```
Does summary capture main points?
Are important details included?
```

**2. Coherence:**
```
Do sentences flow logically?
Is summary readable?
```

**3. Conciseness:**
```
Is it appropriately short?
No redundancy?
```

**4. Factual accuracy:**
```
Are facts from source preserved correctly?
No hallucinations?
```

**Typical evaluation process:**
1. Run automatic metrics (ROUGE)
2. Manual review by humans
3. Compare multiple summarization methods
4. User studies (which summary is better?)

---

## Part C: Information Extraction

### Q8: How do you extract relationships between entities?
**A:** Relation extraction finds semantic relationships between entities mentioned in text.

**Task:**
```
Input: "Steve Jobs founded Apple in Cupertino"
Output: 
  - FOUNDER(Steve Jobs, Apple)
  - LOCATED_IN(Apple, Cupertino)
```

**Classical approaches:**

**1. Pattern-based (handcrafted rules):**
```
Pattern: <PERSON> founded <ORGANIZATION>
→ FOUNDER(PERSON, ORGANIZATION)

Pattern: <ORGANIZATION> in <LOCATION>
→ LOCATED_IN(ORGANIZATION, LOCATION)

Regex examples:
- X founded Y
- X is founder of Y
- X, who started Y
- Y, founded by X
```

**2. Dependency parsing:**
```
"Steve Jobs founded Apple"

Dependency parse:
founded (ROOT)
 ├─ Jobs (nsubj) [PERSON]
 └─ Apple (dobj) [ORGANIZATION]

Rule: If verb="found/establish/create" and nsubj=PERSON and dobj=ORG
→ FOUNDER(PERSON, ORG)
```

**3. Machine learning:**
```
Features for each entity pair:
- Words between entities
- Entity types (PERSON, ORG)
- POS tags between
- Dependency path
- Distance between entities

Train classifier: Relation vs No-Relation
```

**Challenges:**
- Same relation, many ways to express
- Ambiguity: "X works for Y" (employee or contractor?)
- Long-distance dependencies
- Co-reference: "He founded it" (who is "he"?)

**Tools:**
```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define pattern: PERSON + "founded" + ORG
pattern = [
    {"ENT_TYPE": "PERSON"},
    {"LEMMA": "found"},
    {"ENT_TYPE": "ORG"}
]

matcher.add("FOUNDER", [pattern])

doc = nlp("Steve Jobs founded Apple")
matches = matcher(doc)
# Extract: (Steve Jobs, Apple)
```

### Q9: What is coreference resolution and why is it hard?
**A:** Coreference resolution identifies when different expressions refer to the same entity.

**Examples:**
```
"Barack Obama was born in Hawaii. He became president in 2009."
→ "He" refers to "Barack Obama"

"Apple released a new iPhone. The company expects strong sales."
→ "The company" refers to "Apple"

"The box is heavy because it is full of books."
→ "it" refers to "The box"
```

**Why it's hard:**

**1. Ambiguity:**
```
"John told Bill that he should leave"
Who should leave? John or Bill? (unclear!)
```

**2. World knowledge needed:**
```
"The trophy doesn't fit in the suitcase because it's too big"
"it" = trophy (too big to fit)

"The trophy doesn't fit in the suitcase because it's too small"
"it" = suitcase (too small to hold trophy)

→ Requires understanding size constraints!
```

**3. Long distance:**
```
"The company... [5 sentences later] ...it announced..."
Need to track across long context
```

**4. Multiple candidates:**
```
"Alice met Bob and Charlie. She told him..."
"She" = Alice? "him" = Bob or Charlie?
```

**Classical approach (features for ML):**
```
For each pronoun:
- Find candidate antecedents (nouns before pronoun)
- Extract features:
  * Gender match (he → male noun)
  * Number match (they → plural noun)
  * Proximity (closer is more likely)
  * Grammatical role (subject pronouns prefer subject antecedents)
  * Semantic compatibility

Train classifier to score candidates
```

**State-of-the-art:**
- Neural models (Stanford CoreNLP, Allen NLP)
- End-to-end learning
- But still challenging!

**Practical shortcut for classical NLP:**
- Use off-the-shelf tools (spaCy has basic coreference)
- Focus on simple cases (he/she/it in same sentence)
- Accept imperfection

### Q10: How do you build a regex-based information extractor?
**A:** Regular expressions are powerful for extracting structured patterns.

**Common extraction tasks:**

**1. Email addresses:**
```python
import re

pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

text = "Contact us at support@example.com or sales@company.org"
emails = re.findall(pattern, text)
# ['support@example.com', 'sales@company.org']
```

**2. Phone numbers:**
```python
# US format: (123) 456-7890 or 123-456-7890
pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'

text = "Call (555) 123-4567 or 555-987-6543"
phones = re.findall(pattern, text)
```

**3. Dates:**
```python
# Various formats
patterns = [
    r'\d{4}-\d{2}-\d{2}',           # 2024-01-15
    r'\d{1,2}/\d{1,2}/\d{2,4}',     # 1/15/2024 or 01/15/24
    r'[A-Z][a-z]+ \d{1,2}, \d{4}'   # January 15, 2024
]

for pattern in patterns:
    dates = re.findall(pattern, text)
```

**4. Domain-specific (medical, legal):**
```python
# Extract drug mentions with dosage
pattern = r'(\d+)\s*mg\s+of\s+([A-Za-z]+)'

text = "Patient received 500 mg of aspirin and 10 mg of lisinopril"
matches = re.findall(pattern, text)
# [('500', 'aspirin'), ('10', 'lisinopril')]
```

**5. Named patterns:**
```python
# Extract structured info
pattern = r'(?P<name>[A-Z][a-z]+ [A-Z][a-z]+) works at (?P<company>[A-Z][A-Za-z ]+)'

text = "John Smith works at Google Inc and Jane Doe works at Microsoft"
for match in re.finditer(pattern, text):
    print(f"Name: {match.group('name')}, Company: {match.group('company')}")
```

**Best practices:**
- Start simple, test thoroughly
- Use raw strings: `r'pattern'`
- Name capture groups for readability
- Combine multiple patterns for variations
- Validate extracted data

---

## Part D: Language Models & Applications

### Q11: How do N-gram language models work?
**A:** N-gram models estimate probability of word sequences based on preceding words.

**Unigram model (most basic):**
```
P(word) = count(word) / total_words

P("the") = 50,000 / 1,000,000 = 0.05
P("quantum") = 100 / 1,000,000 = 0.0001
```

**Bigram model:**
```
P(word | previous_word) = count(prev, word) / count(prev)

P("cat" | "the") = count("the cat") / count("the")
                 = 500 / 50,000 = 0.01

Sentence probability:
P("the cat sat") = P("the") × P("cat"|"the") × P("sat"|"cat")
```

**Trigram model:**
```
P(word | previous_2_words)

P("sat" | "the cat") = count("the cat sat") / count("the cat")
```

**The Markov assumption:**
```
P(word | all previous words) ≈ P(word | last N-1 words)

Makes computation tractable!
```

**Applications:**

**1. Next word prediction (autocomplete):**
```
User typed: "Hello how are"
Model predicts: P("you" | "how are") = 0.8 (highest)
Suggestion: "you"
```

**2. Sentence generation:**
```
Start: <START>
Sample next word from P(word | <START>)
Sample next from P(word | previous_word)
Continue until <END>
```

**3. Spell checking:**
```
Input: "I wnat to go"

Check: P("I want to go") vs P("I wnat to go")
"want" has much higher probability → suggest correction
```

### Q12: What are smoothing techniques and why are they needed?
**A:** Smoothing handles unseen N-grams that have zero count (and would get zero probability).

**The problem:**
```
Training corpus: "the cat sat on the mat"

Test: "the dog sat"

P("dog" | "the") = count("the dog") / count("the")
                 = 0 / 2 = 0

Problem: Entire sentence probability becomes 0!
P("the dog sat") = P("the") × 0 × P("sat"|"dog") = 0
```

**Smoothing techniques:**

**1. Add-one (Laplace) smoothing:**
```
Add 1 to all counts (pretend we saw everything once)

P("dog" | "the") = (count("the dog") + 1) / (count("the") + vocab_size)
                = (0 + 1) / (2 + 10000) = 1/10002

No longer zero!
```

**Problem with add-one:**
- Too much probability mass to unseen events
- Rare events over-smoothed

**2. Add-k smoothing:**
```
Add k instead of 1 (k < 1, e.g., k=0.01)
Less aggressive smoothing
```

**3. Good-Turing smoothing:**
```
Use count of counts to estimate unseen probabilities

Idea: Things we've seen once are like things we've never seen
```

**4. Kneser-Ney smoothing (best):**
```
Considers how diverse contexts a word appears in

"Francisco" appears 100 times, but only after "San"
"city" appears 100 times, after many different words

Kneser-Ney gives higher probability to "city" in novel contexts
```

**Example:**
```
Train: "San Francisco", "New York", "Los Angeles"

Test: "I visited _____"

Simple count: P("Francisco") = P("York") = P("Angeles")
Kneser-Ney: P("city") > P("Francisco")
(because "Francisco" only after "San", but "city" more flexible)
```

**Choice matters for:**
- Small training data → aggressive smoothing needed
- Large training data → light smoothing sufficient
- Domain-specific → may need custom smoothing

### Q13: What is perplexity and how do you use it?
**A:** Perplexity measures how "surprised" a language model is by test data.

**Formula:**
```
Perplexity = 2^(-average log probability)

Lower perplexity = better model (less surprised)
```

**Intuition:**
```
"On average, the model is choosing between N options"

Perplexity = 100 means model is as uncertain as random choice from 100 words
Perplexity = 10 means model narrowed down to ~10 likely words
```

**Example:**
```
Test sentence: "the cat sat"

Model predicts:
P("the") = 0.05 → log₂(0.05) = -4.32
P("cat"|"the") = 0.01 → log₂(0.01) = -6.64
P("sat"|"cat") = 0.10 → log₂(0.10) = -3.32

Average: (-4.32 - 6.64 - 3.32) / 3 = -4.76
Perplexity: 2^4.76 = 27
```

**Using perplexity:**

**1. Model comparison:**
```
Bigram model: Perplexity = 150
Trigram model: Perplexity = 120  ← Better!
```

**2. Track training:**
```
Epoch 1: Perplexity = 500
Epoch 2: Perplexity = 300
Epoch 3: Perplexity = 250
Epoch 4: Perplexity = 248  ← Converging
```

**3. Detect overfitting:**
```
Train perplexity: 50 (low)
Test perplexity: 300 (high)
→ Overfitting!
```

**Note:** Perplexity is dataset-specific (absolute values not comparable across datasets).

### Q14: How do you build an autocomplete system?
**A:**

**Approach 1: N-gram language model**

```python
from collections import defaultdict, Counter

class Autocomplete:
    def __init__(self, n=3):
        self.n = n
        self.ngrams = defaultdict(Counter)
    
    def train(self, texts):
        for text in texts:
            words = ['<START>'] * (self.n - 1) + text.split() + ['<END>']
            for i in range(len(words) - self.n + 1):
                context = tuple(words[i:i+self.n-1])
                next_word = words[i+self.n-1]
                self.ngrams[context][next_word] += 1
    
    def predict(self, context_words, top_k=5):
        context = tuple(context_words[-(self.n-1):])  # Last N-1 words
        candidates = self.ngrams[context]
        
        # Return top K most common
        return candidates.most_common(top_k)

# Usage
model = Autocomplete(n=3)  # Trigram
model.train(["hello how are you", "how are you doing", ...])

suggestions = model.predict(["how", "are"], top_k=3)
# [('you', 150), ('we', 20), ('they', 5)]
```

**Approach 2: Trie (prefix tree) + frequency**

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.frequency = 0

class TrieAutocomplete:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word, freq=1):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.frequency += freq
    
    def autocomplete(self, prefix, top_k=5):
        # Find node for prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Get all words with this prefix
        words = []
        self._collect_words(node, prefix, words)
        
        # Sort by frequency and return top K
        words.sort(key=lambda x: x[1], reverse=True)
        return words[:top_k]
    
    def _collect_words(self, node, prefix, words):
        if node.is_word:
            words.append((prefix, node.frequency))
        for char, child in node.children.items():
            self._collect_words(child, prefix + char, words)
```

**Approach 3: Hybrid (real-world)**

```
Combine:
1. N-gram context (what user is typing)
2. Frequency (popular completions)
3. Personalization (user's history)
4. Recency (trending words)

Score(word) = 
  0.4 × P(word|context) +
  0.3 × frequency(word) +
  0.2 × user_preference(word) +
  0.1 × recency(word)
```

---

## Part E: Putting It All Together

### Q15: How do you build a multi-class document classifier end-to-end?
**A:**

**Complete pipeline:**

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd

# 1. Load and explore data
df = pd.read_csv('documents.csv')
print(df['category'].value_counts())  # Check class balance

# 2. Preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove special chars
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize and lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split()]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess)

# 3. Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        class_weight='balanced'  # Handle imbalance
    ))
])

# 4. Cross-validation
from sklearn.model_selection import StratifiedKFold

cv_scores = cross_val_score(
    pipeline,
    df['clean_text'],
    df['category'],
    cv=StratifiedKFold(5),
    scoring='f1_macro'
)
print(f"CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# 5. Train final model
pipeline.fit(df['clean_text'], df['category'])

# 6. Evaluate
from sklearn.metrics import classification_report, confusion_matrix

predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions))

# 7. Save model
from joblib import dump
dump(pipeline, 'classifier.joblib')

# 8. Use in production
def classify_new_document(text):
    clean = preprocess(text)
    category = pipeline.predict([clean])[0]
    probas = pipeline.predict_proba([clean])[0]
    confidence = max(probas)
    return category, confidence
```

### Q16: How do you diagnose and improve a poorly performing classifier?
**A:**

**Diagnostic checklist:**

**1. Check data quality:**
```python
# Are labels correct?
df.sample(20)  # Manual inspection

# Class balance?
df['label'].value_counts()

# Text quality?
df['text'].str.len().describe()  # Length distribution
df['text'].str.split().str.len().describe()  # Word count
```

**2. Analyze errors:**
```python
from sklearn.metrics import confusion_matrix

# Where is model confused?
cm = confusion_matrix(y_true, y_pred)
print(cm)

# What are misclassified examples?
errors = df[(predictions != y_true)]
errors.sample(20)  # Read them!
```

**3. Feature analysis:**
```python
# What features matter?
tfidf = pipeline.named_steps['tfidf']
clf = pipeline.named_steps['clf']

feature_names = tfidf.get_feature_names_out()
coefs = clf.coef_[0]  # For binary

# Top positive features
top_positive = [feature_names[i] for i in coefs.argsort()[-20:]]

# Top negative features
top_negative = [feature_names[i] for i in coefs.argsort()[:20]]

print("Positive indicators:", top_positive)
print("Negative indicators:", top_negative)
```

**Improvement strategies:**

**1. More/better data:**
- Collect more examples of confused classes
- Balance dataset (oversample minority, undersample majority)
- Clean mislabeled examples

**2. Better preprocessing:**
- Try with/without stemming/lemmatization
- Adjust stopword list
- Domain-specific cleaning

**3. Better features:**
- Increase max_features
- Try different ngram_range
- Add bigrams/trigrams
- Use word embeddings instead of TF-IDF

**4. Better model:**
- Try SVM instead of Logistic Regression
- Ensemble multiple models
- Tune hyperparameters (GridSearchCV)

**5. Handle class imbalance:**
- class_weight='balanced'
- Oversample minority (SMOTE)
- Adjust decision threshold

### Q17: What's a complete workflow for a real NLP project?
**A:**

**Phase 1: Problem Definition**
```
1. Define task clearly
   - Classification? Extraction? Summarization?
   - Binary or multi-class?
   - What's success? (metric)

2. Understand data
   - How much data available?
   - Quality? Labels reliable?
   - Class balance?

3. Set baseline
   - Random guess accuracy?
   - Majority class accuracy?
   - Simple heuristic?
```

**Phase 2: Exploratory Data Analysis**
```
4. Data statistics
   - Distribution of labels
   - Text length distribution
   - Vocabulary size
   - Common words per class

5. Sample inspection
   - Read 50-100 examples
   - Look for patterns
   - Identify challenges

6. Initial preprocessing experiments
   - Try different cleaning strategies
   - Check impact on readable samples
```

**Phase 3: Modeling**
```
7. Start simple
   - BoW + Naive Bayes (baseline)
   - TF-IDF + Logistic Regression

8. Iterate
   - Add bigrams
   - Try different models
   - Hyperparameter tuning

9. Error analysis
   - Find systematic errors
   - Analyze confusion matrix
   - Read misclassified examples

10. Feature engineering
    - Based on error analysis
    - Add domain-specific features
    - Try embeddings
```

**Phase 4: Evaluation**
```
11. Cross-validation
    - Stratified K-Fold
    - Report mean and std

12. Multiple metrics
    - Accuracy, Precision, Recall, F1
    - Per-class breakdown
    - Confusion matrix

13. Test on held-out set
    - Final model evaluation
    - Compare to baseline
```

**Phase 5: Deployment**
```
14. Package pipeline
    - Save complete pipeline
    - Document preprocessing steps
    - Version control

15. Monitor performance
    - Track predictions
    - Catch data drift
    - Retrain periodically
```

---

## 🎯 Key Takeaways

1. **Lexicon-based sentiment** (VADER) is fast and needs no training data
2. **Aspect-based sentiment** gives actionable insights beyond document-level
3. **Extractive summarization** (TextRank, LSA) is feasible with classical NLP
4. **Relation extraction** combines NER, parsing, and patterns
5. **Coreference resolution** is hard but important for understanding
6. **Regex is powerful** for structured extraction (emails, dates, patterns)
7. **N-gram language models** enable autocomplete and spell checking
8. **Smoothing is essential** to handle unseen N-grams
9. **Perplexity measures** model quality (lower is better)
10. **End-to-end projects** require iteration, error analysis, and refinement
11. **Start simple, then improve** - don't jump to complex models first
12. **Error analysis drives improvement** - understand failures to fix them
