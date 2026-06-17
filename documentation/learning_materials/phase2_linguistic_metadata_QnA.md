# Phase 2: Linguistic Metadata - Q&A

> **Goal:** Extract structural meaning without using complex neural networks.

---

## 1. Parts of Speech (POS) Tagging

### Q1: What is POS tagging and why is it fundamental to NLP?
**A:** POS tagging is the process of assigning grammatical categories (noun, verb, adjective, etc.) to each word in a sentence based on its context.

**Why fundamental:**
- **Disambiguates meaning:** "book" can be a noun ("read a book") or verb ("book a flight")
- **Enables syntax understanding:** Identifies sentence structure
- **Prerequisite for advanced tasks:** Parsing, NER, relation extraction all depend on POS
- **Improves lemmatization:** "better" → "good" (adjective) vs "better" → "well" (adverb)

**Example:**
```
Input:  "The quick brown fox jumps over the lazy dog"
Output: DT  ADJ   ADJ   NN  VBZ   IN   DT  ADJ  NN
        (determiner, adjectives, noun, verb, preposition, etc.)
```

### Q2: What are the major POS tag sets?
**A:**

**Penn Treebank Tag Set (most common):**
- 36 tags total
- Examples: NN (noun), VB (verb base), VBZ (verb 3rd person), JJ (adjective), RB (adverb)
- Very detailed: distinguishes VB, VBD, VBG, VBN, VBP, VBZ

**Universal Dependencies (UD) Tag Set:**
- 17 simplified tags
- NOUN, VERB, ADJ, ADV, PRON, DET, ADP, etc.
- Better for cross-linguistic work
- More generalized but easier to work with

**Why multiple tag sets?**
- Trade-off between granularity and complexity
- Different languages need different distinctions
- Task-specific needs (detailed analysis vs broad categorization)

### Q3: How does POS tagging actually work?
**A:**

**Classical approach: Hidden Markov Models (HMMs)**
- Learns probability of tags following each other: P(VB|NN)
- Learns probability of words given tags: P("run"|VB)
- Finds most likely tag sequence using Viterbi algorithm

**Modern approach: Conditional Random Fields (CRFs)**
- Considers more context features
- Doesn't assume independence of observations
- Better accuracy than HMMs

**Deep learning approach:**
- BiLSTM + CRF layers
- Contextual embeddings (BERT)
- State-of-the-art but requires more data

**For classical NLP:** Use pre-trained models from spaCy or NLTK (already 95%+ accurate).

### Q4: When is POS tagging critical for your task?
**A:**

**Critical for:**
- **Lemmatization:** Need POS to lemmatize correctly
- **Relation extraction:** "Apple announced..." (ORG) vs "apple tastes..." (food)
- **Dependency parsing:** Understanding sentence structure
- **Information extraction:** Extracting noun phrases, verb phrases
- **Question answering:** Identifying question types based on verbs

**Less critical for:**
- **Spam detection:** Word presence matters more than grammar
- **Topic modeling:** Focus on content words, not structure
- **Simple sentiment:** "good" vs "bad" regardless of part of speech

### Q5: What are common POS tagging errors and how to handle them?
**A:**

**Common errors:**
1. **Ambiguous words:** "book" (NN vs VB), "close" (ADJ vs VB vs ADV)
2. **Unknown words:** Rare words or typos
3. **Domain-specific language:** Technical jargon, slang
4. **Long-distance dependencies:** Complex sentence structures

**Handling strategies:**
- **Use context-aware taggers:** spaCy performs better than simple taggers
- **Domain adaptation:** Retrain on domain-specific data if needed
- **Ensemble methods:** Combine multiple taggers
- **Post-processing rules:** Fix known systematic errors

---

## 2. Named Entity Recognition (NER)

### Q6: What is NER and what types of entities can you extract?
**A:** NER is identifying and classifying named entities (proper nouns) in text into predefined categories.

**Standard entity types:**
- **PERSON:** "Barack Obama", "Dr. Smith"
- **ORGANIZATION:** "Google", "United Nations"
- **LOCATION:** "New York", "Mount Everest", "Asia"
- **DATE:** "January 2023", "last week", "2020-01-01"
- **TIME:** "3:00 PM", "midnight"
- **MONEY:** "$100", "€50", "10 dollars"
- **PERCENT:** "15%", "one-third"
- **FACILITY:** "JFK Airport", "Stanford University"
- **GPE (Geo-Political Entity):** "United States", "California"

**Domain-specific entities:**
- Medical: DISEASE, DRUG, SYMPTOM, TREATMENT
- Legal: LAW, CASE, COURT
- Financial: STOCK_SYMBOL, CURRENCY, FISCAL_QUARTER

### Q7: Why is NER more challenging than POS tagging?
**A:**

**Challenges unique to NER:**

1. **Multi-word entities:** "New York City" (3 tokens, 1 entity)
2. **Nested entities:** "Bank of America" (organization) in "New York" (location)
3. **Ambiguity:** "Washington" (PERSON or GPE?) depends on context
4. **New entities constantly appear:** People, companies, products
5. **No clear boundaries:** Is it "iPhone" or "Apple iPhone"?
6. **Case sensitivity critical:** "apple" vs "Apple"
7. **Context-dependent:** "I'm going to the Apple Store" (ORG or PRODUCT?)

**POS tagging is easier:**
- Fixed, small tag set
- Words have typical POS patterns
- Less context-dependent
- Sentence-local (doesn't need world knowledge)

### Q8: How do classical NER systems work?
**A:**

**Rule-based approach:**
- **Gazetteers:** Lists of known entities (cities, company names)
- **Patterns:** "Mr./Mrs./Dr." followed by capitalized word → PERSON
- **Regex:** Phone numbers, emails, dates follow patterns
- **High precision, low recall:** Only finds what you explicitly program

**Machine Learning approach (classical):**
1. **Feature engineering:**
   - Word itself and surrounding words
   - POS tags
   - Capitalization patterns
   - Word shapes (Xxxxx, XXXXX, 123-456)
   - Prefixes/suffixes
   - Gazeteer membership

2. **Sequence labeling:**
   - Use CRF or HMM to tag sequences
   - IOB tagging scheme:
     - B-PER: Beginning of person
     - I-PER: Inside person
     - O: Outside any entity
   
**Example:**
```
Barack    → B-PERSON
Obama     → I-PERSON
visited   → O
New       → B-GPE
York      → I-GPE
```

**Modern approach:**
- BiLSTM-CRF with word embeddings
- Transformer models (BERT-based)
- 90%+ F1 on standard datasets

### Q9: What's the IOB (BIO) tagging scheme and why is it needed?
**A:** IOB is a way to represent multi-word entities in sequence labeling.

**Tags:**
- **B-TYPE:** Beginning of an entity type
- **I-TYPE:** Inside/continuation of an entity
- **O:** Outside any entity

**Why needed:**
Makes multi-word entities explicit and handles adjacent entities:

```
Example: "John Smith and Mary Jones work at Microsoft"

Without IOB (wrong):
John      PERSON
Smith     PERSON  ← How do we know this continues "John"?
and       O
Mary      PERSON
Jones     PERSON  ← Separate person or part of "Mary"?
work      O
at        O
Microsoft ORG
```

**With IOB (correct):**
```
John      B-PERSON  ← Start of entity
Smith     I-PERSON  ← Part of same entity
and       O
Mary      B-PERSON  ← New entity starts
Jones     I-PERSON
work      O
at        O
Microsoft B-ORG
```

**Variant: IOBES**
- B: Beginning
- I: Inside
- E: End
- S: Single-token entity
- O: Outside

More explicit but more complex.

### Q10: How do you evaluate NER systems?
**A:**

**Token-level evaluation:**
- Treats each token independently
- Easier but not reflective of real use

**Entity-level evaluation (standard):**
- Entity is correct only if:
  1. Boundaries are exact ("New York" not "New" or "New York City")
  2. Type is correct (PERSON not ORG)

**Metrics:**

**Strict matching:**
```
Precision = Correct entities / Predicted entities
Recall = Correct entities / True entities
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Partial matching:**
- Give partial credit for overlapping entities
- Separate scores for type and boundary

**Example:**
```
True:      "Barack Obama" (PERSON), "United Nations" (ORG)
Predicted: "Barack" (PERSON), "United Nations" (PERSON)

Strict entity-level:
- TP: 0 (neither is exact match)
- FP: 2 (both predictions wrong)
- FN: 2 (missed both true entities)
- Precision: 0%, Recall: 0%, F1: 0%

Partial credit might be more generous.
```

---

## 3. Dependency Parsing

### Q11: What is dependency parsing and how does it differ from constituency parsing?
**A:** Dependency parsing identifies grammatical relationships between words in a sentence.

**Dependency parsing:**
- Shows which words modify which
- Direct word-to-word relationships
- Creates a tree with words as nodes
- Language-independent structure

**Example:**
```
"The cat sat on the mat"

sat (ROOT)
 ├─ cat (subject)
 │   └─ The (determiner)
 ├─ on (preposition)
     └─ mat (object of preposition)
         └─ the (determiner)
```

**Constituency parsing:**
- Groups words into phrases (noun phrase, verb phrase)
- Hierarchical phrase structure
- Language-specific

**Example:**
```
              S
         _____|_____
       NP            VP
      __|__       ___|___
    DT    NN     VBD     PP
    |     |       |    ___|___
   The   cat     sat  IN      NP
                       |     __|__
                      on    DT   NN
                             |    |
                            the  mat
```

**Why use each:**
- **Dependency:** Simpler, better for relation extraction, multilingual NLP
- **Constituency:** Better for understanding phrase-level meaning, grammar checking

### Q12: What are dependency labels and why do they matter?
**A:** Dependency labels describe the grammatical relationship between connected words.

**Common labels (Universal Dependencies):**
- **nsubj:** Nominal subject ("cat" is subject of "sat")
- **dobj:** Direct object ("book" in "read a book")
- **amod:** Adjectival modifier ("red" modifying "car")
- **advmod:** Adverbial modifier ("quickly" modifying "ran")
- **det:** Determiner ("the", "a", "an")
- **prep:** Prepositional modifier
- **pobj:** Object of preposition
- **compound:** Compound words ("New York")
- **conj:** Conjunct in coordination ("and", "or")

**Why they matter:**
- **Semantic role labeling:** Who did what to whom?
- **Relation extraction:** "X founded Y" → (X, founded, Y) relationship
- **Question answering:** Understanding question structure
- **Coreference resolution:** "He said he" (which "he" is subject?)

### Q13: How can dependency parsing help with practical NLP tasks?
**A:**

**1. Information Extraction:**
Find all (PERSON, action, ORGANIZATION) tuples:
```
"Steve Jobs founded Apple in 1976"

Parse:
founded (ROOT)
 ├─ Jobs (nsubj) [PERSON]
 └─ Apple (dobj) [ORG]
 
Extract: (Steve Jobs, founded, Apple)
```

**2. Relation Extraction:**
```
"Obama was born in Hawaii"

born (ROOT)
 ├─ Obama (nsubjpass) [PERSON]
 └─ in (prep)
     └─ Hawaii (pobj) [LOCATION]

Relation: BIRTHPLACE(Obama, Hawaii)
```

**3. Negation Detection:**
```
"The product is not good"

good (ROOT)
 └─ not (neg)  ← Negation dependency

Action: Flip sentiment from positive to negative
```

**4. Aspect-Based Sentiment:**
```
"The battery life is great but the screen is terrible"

Find what modifies what:
- great → battery life (positive aspect)
- terrible → screen (negative aspect)
```

**5. Question Answering:**
```
"Who founded Microsoft?"

founded (ROOT)
 ├─ Who (nsubj) ← This is what we need to find
 └─ Microsoft (dobj)

Search for: "X founded Microsoft" where X is PERSON
```

### Q14: What are the limitations of dependency parsing in classical NLP?
**A:**

**Computational complexity:**
- Parsing is O(n³) or more expensive
- Too slow for real-time on very long documents
- May need to break into sentences first

**Accuracy issues:**
- 85-95% labeled attachment score (LAS) on standard data
- Lower on domain-specific or noisy text
- Complex sentences harder to parse correctly

**Ambiguity:**
- PP-attachment: "I saw the man with the telescope"
  - Did I use telescope to see? Or did man have telescope?
- Multiple valid parse trees possible

**Long-distance dependencies:**
- Hard to capture relationships across clauses
- "The book that I told you about yesterday is here"

**Solution in classical NLP:**
- Use dependency parsing selectively (not on every sentence)
- Focus on simple, high-confidence dependencies
- Combine with other features (POS, NER) for robustness

---

## 4. Stopword Customization

### Q15: Why would you create domain-specific stopword lists?
**A:** Generic stopword lists (NLTK's English stopwords) are designed for general text but may not suit specialized domains.

**Examples of domain-specific stopwords:**

**Medical texts:**
- Generic: "the", "is", "at"
- Domain: "patient", "doctor", "hospital", "treatment" (appear in nearly every document)

**Legal documents:**
- Generic: "the", "is"
- Domain: "court", "case", "law", "attorney", "hereby", "whereas"

**Product reviews:**
- Generic: "the", "a"
- Domain: "product", "item", "purchase", "bought"

**News articles:**
- Generic: "the", "and"
- Domain: "said", "according", "reported", "sources"

**Why customize:**
- More aggressive dimensionality reduction
- Remove uninformative words specific to corpus
- Improve topic modeling (remove terms that appear in all topics)
- Better classification (remove words that don't discriminate)

### Q16: How do you build a domain-specific stopword list?
**A:**

**Method 1: Frequency-based**
```python
# Remove words appearing in > 80% of documents
# Or remove top 5% most frequent words

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(max_df=0.8)  # Appears in > 80% of docs
vec.fit(documents)
# Words filtered out are effectively "stopwords"
```

**Method 2: TF-IDF based**
```python
# Words with very low IDF (inverse document frequency)
# IDF close to 0 means word appears in most documents

from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()
vec.fit(documents)

# Get words with lowest IDF
idf_scores = dict(zip(vec.get_feature_names_out(), vec.idf_))
low_idf_words = [word for word, idf in idf_scores.items() if idf < threshold]
```

**Method 3: Manual curation**
- Read sample documents
- Identify frequently appearing uninformative words
- Combine with generic stopword list
- Iteratively refine based on results

**Method 4: Statistical testing**
- Chi-squared test for word-class association
- Words with no discriminative power → stopwords

**Best practice:** Start with generic list, add domain-specific terms iteratively.

### Q17: Can aggressive stopword removal backfire?
**A:** **Yes, absolutely!**

**Dangers of over-aggressive removal:**

**1. Loss of discriminative features:**
```
Document classification task: Medical vs Legal documents

Removed as "domain stopwords": "patient", "doctor", "court", "attorney"

Problem: These are the MOST discriminative words!
```

**2. Destroying negations:**
```
"not good" → "good" (meaning reversed!)
"no problem" → "problem" (positive becomes negative)
```

**3. Losing important function words:**
```
"To be or not to be" → "" (nothing left!)
"Can you help me?" vs "You can help me" (question vs statement)
```

**4. Breaking phrases:**
```
"out of stock" → "stock" (loses "out of stock" concept)
"state of the art" → "state art"
```

**Guidelines:**
- Remove conservatively, test impact on performance
- Keep negation words: "not", "no", "never", "neither"
- Keep intensifiers in sentiment analysis: "very", "really", "extremely"
- Keep question words in QA: "who", "what", "when", "where", "why", "how"
- A/B test with and without stopword removal

---

## 5. Putting It All Together

### Q18: In what order should you apply linguistic analysis?
**A:**

**Recommended pipeline:**
```
1. Sentence segmentation (if needed)
2. Tokenization
3. POS tagging
4. NER (uses POS tags)
5. Lemmatization (uses POS tags)
6. Dependency parsing (uses POS tags)
7. Coreference resolution (uses NER and parsing)
```

**Why this order:**
- Parsing works on tokens
- NER often uses POS features
- Lemmatization needs POS to be accurate
- Dependency parsing builds on POS
- Coreference uses all of the above

**In practice with spaCy:**
```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("Apple CEO Tim Cook announced new products.")

# All done in one pass!
for token in doc:
    print(token.text, token.pos_, token.dep_, token.ent_type_)
```

### Q19: Which linguistic features should you use for different tasks?
**A:**

| Task | Essential Features | Optional Features |
|------|-------------------|-------------------|
| **Sentiment Analysis** | Tokens, negation detection | POS (adj/adv), dependency (intensifiers) |
| **Topic Modeling** | Lemmatized tokens, stopword removal | POS (keep only nouns/verbs) |
| **NER** | Tokens, POS, capitalization, word shapes | Dependency parsing, gazetteers |
| **Question Answering** | POS, NER, dependency parsing | Coreference, semantic roles |
| **Text Classification** | Tokens, possibly POS | NER (for some domains) |
| **Relation Extraction** | NER, dependency parsing | POS, coreference |
| **Summarization** | Sentence segmentation, NER | POS, dependency (for sentence importance) |

**Rule of thumb:**
- Start simple (just tokens)
- Add linguistic features if baseline is insufficient
- More features ≠ better (can overfit, slow down)

### Q20: How do you handle linguistic analysis errors propagating through pipeline?
**A:**

**Error propagation problem:**
```
Bad tokenization → Bad POS tagging → Bad parsing → Bad NER → Bad final results
```

**Mitigation strategies:**

**1. Use robust, pre-trained models:**
- spaCy (95%+ accuracy on POS, NER)
- Stanford CoreNLP
- Don't build from scratch unless you have domain-specific needs

**2. Handle uncertainty:**
- Use confidence scores if available
- Keep top-N parses rather than just best
- Ensemble multiple parsers

**3. Design robust downstream features:**
- Don't rely solely on parse trees
- Combine linguistic and distributional features
- Bag-of-words as fallback

**4. Error analysis:**
- Manually check sample output
- Identify systematic errors
- Add preprocessing rules to handle common cases

**5. End-to-end learning (modern approach):**
- Joint models that do tokenization + POS + NER together
- Share representations, reduce error propagation
- But requires more data and compute

---

## 🎯 Key Takeaways

1. **POS tagging disambiguates word meaning** - essential for lemmatization and parsing
2. **NER extracts structured information** - people, places, organizations from unstructured text
3. **Dependency parsing reveals relationships** - who did what to whom
4. **Domain-specific customization is critical** - one size doesn't fit all
5. **Use pre-trained models** - spaCy and NLTK have 95%+ accuracy
6. **Linguistic features are task-dependent** - not every task needs parsing
7. **Errors compound through pipelines** - start with high-quality base models
8. **Balance complexity with practicality** - more analysis isn't always better
