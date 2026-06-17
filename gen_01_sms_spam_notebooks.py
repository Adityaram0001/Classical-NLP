import os
import nbformat as nbf
from pathlib import Path

def create_notebook(filename, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    with open(filename, 'w') as f:
        nbf.write(nb, f)
    print(f"Created {filename}")

def decision_note(step, chosen, why_works, alt_a, alt_a_pros, alt_a_cons, alt_b, alt_b_pros, alt_b_cons, reason):
    return f"""> **📌 Decision Note — Why {step}?**
>
> **Chosen approach:** {chosen}
>
> **Why this works:** {why_works}
>
> **Alternatives we could have used:**
> | Option | Pros | Cons |
> |--------|------|------|
> | {alt_a} | {alt_a_pros} | {alt_a_cons} |
> | {alt_b} | {alt_b_pros} | {alt_b_cons} |
>
> **Why we chose this over alternatives:** {reason}"""

def generate_notebooks():
    out_dir = Path("datasets/01_sms_spam_collection/notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. EDA and Cleaning
    cells_01 = [
        nbf.v4.new_markdown_cell("# 01. Exploratory Data Analysis & Text Cleaning\n\n**Goal:** Understand the dataset structure, distribution of spam vs. ham, and perform initial text cleaning."),
        nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport sys\nsys.path.append('../../../shared')\nfrom text_utils import clean_text_basic"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("# The dataset is a TSV (tab-separated) file without headers\ndf = pd.read_csv('../data/raw/SMSSpamCollection', sep='\\t', names=['label', 'message'])\ndf.head()"),
        nbf.v4.new_markdown_cell("## 2. Exploratory Data Analysis"),
        nbf.v4.new_code_cell("print(df.info())\nprint('\\nTarget Distribution:')\nprint(df['label'].value_counts())\n\nsns.countplot(x='label', data=df)\nplt.title('Distribution of Spam vs Ham')\nplt.show()"),
        nbf.v4.new_code_cell("# Let's look at message length as a potential feature\ndf['length'] = df['message'].apply(len)\n\nplt.figure(figsize=(10, 5))\ndf[df['label'] == 'ham']['length'].plot(bins=50, kind='hist', alpha=0.5, label='ham', color='blue')\ndf[df['label'] == 'spam']['length'].plot(bins=50, kind='hist', alpha=0.5, label='spam', color='red')\nplt.legend()\nplt.title('Message Length Distribution by Class')\nplt.xlabel('Message Length')\nplt.show()"),
        nbf.v4.new_markdown_cell(decision_note(
            "Basic text cleaning",
            "Lowercasing, punctuation removal, and stripping whitespaces",
            "Standardizes the vocabulary, preventing 'Hello' and 'hello!' from being treated as different words.",
            "Keep punctuation", "Might capture spammy patterns (e.g., 'WIN!!!')", "Increases vocabulary size massively; BoW models handle huge vocabularies poorly.",
            "Advanced Lemmatization", "Reduces words to base dictionary form", "Overkill for simple spam classification, and computationally expensive.",
            "Simple text cleaning provides a fast, robust baseline for this easy dataset."
        )),
        nbf.v4.new_markdown_cell("## 3. Text Cleaning"),
        nbf.v4.new_code_cell("df['cleaned_message'] = df['message'].apply(clean_text_basic)\ndf[['label', 'message', 'cleaned_message']].head()"),
        nbf.v4.new_markdown_cell("## 4. Save Processed Data"),
        nbf.v4.new_code_cell("df.to_csv('../data/processed/cleaned_sms.csv', index=False)\nprint('Saved cleaned dataset.')"),
        nbf.v4.new_markdown_cell("## Key Takeaways\n- [x] Data is highly imbalanced (mostly ham).\n- [x] Spam messages tend to be significantly longer than ham messages.\n- [x] Cleaned data is saved and ready for vectorization.")
    ]
    create_notebook(out_dir / "01_eda_and_cleaning.ipynb", cells_01)

    # 2. Text Vectorization
    cells_02 = [
        nbf.v4.new_markdown_cell("# 02. Text Vectorization\n\n**Goal:** Convert text into numerical features using Bag-of-Words and TF-IDF."),
        nbf.v4.new_code_cell("import pandas as pd\nfrom sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\nfrom sklearn.model_selection import train_test_split\nimport joblib\nimport sys\nsys.path.append('../../../shared')"),
        nbf.v4.new_markdown_cell("## 1. Load Data & Split"),
        nbf.v4.new_code_cell("df = pd.read_csv('../data/processed/cleaned_sms.csv')\n# Drop any NaNs created if a message was just punctuation\ndf = df.dropna(subset=['cleaned_message'])\n\nX_train, X_test, y_train, y_test = train_test_split(df['cleaned_message'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])\nprint('Train shape:', X_train.shape)\nprint('Test shape:', X_test.shape)"),
        nbf.v4.new_markdown_cell(decision_note(
            "Vectorization method",
            "Compare BoW and TF-IDF",
            "Allows us to see if term frequency weighting helps over simple occurrence.",
            "BoW only", "Simple, fast", "Treats frequent words (the, a) equally to rare important words.",
            "Word2Vec/Embeddings", "Captures semantic meaning", "Overkill for spam detection, requires more data.",
            "TF-IDF usually performs slightly better on spam as it downweights common English words automatically."
        )),
        nbf.v4.new_markdown_cell("## 2. Bag of Words (CountVectorizer)"),
        nbf.v4.new_code_cell("vectorizer_bow = CountVectorizer(stop_words='english')\nX_train_bow = vectorizer_bow.fit_transform(X_train)\nX_test_bow = vectorizer_bow.transform(X_test)\nprint('BoW Vocabulary Size:', len(vectorizer_bow.vocabulary_))"),
        nbf.v4.new_markdown_cell("## 3. TF-IDF (Term Frequency-Inverse Document Frequency)"),
        nbf.v4.new_code_cell("vectorizer_tfidf = TfidfVectorizer(stop_words='english')\nX_train_tfidf = vectorizer_tfidf.fit_transform(X_train)\nX_test_tfidf = vectorizer_tfidf.transform(X_test)\nprint('TF-IDF Vocabulary Size:', len(vectorizer_tfidf.vocabulary_))"),
        nbf.v4.new_markdown_cell("## 4. Save Vectorized Data"),
        nbf.v4.new_code_cell("joblib.dump((X_train_bow, X_test_bow, y_train, y_test, vectorizer_bow), '../data/processed/bow_data.pkl')\njoblib.dump((X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer_tfidf), '../data/processed/tfidf_data.pkl')\nprint('Saved vectorized data.')"),
        nbf.v4.new_markdown_cell("## Key Takeaways\n- [x] Both methods produce matrices with the same number of features (vocabulary size).\n- [x] We removed english stopwords during vectorization to reduce noise.\n- [x] Data is saved for modeling.")
    ]
    create_notebook(out_dir / "02_text_vectorization.ipynb", cells_02)

    # 3. Naive Bayes
    cells_03 = [
        nbf.v4.new_markdown_cell("# 03. Naive Bayes Classifier\n\n**Goal:** Train a baseline Multinomial Naive Bayes model on both BoW and TF-IDF features."),
        nbf.v4.new_code_cell("from sklearn.naive_bayes import MultinomialNB\nimport joblib\nimport sys\nsys.path.append('../../../shared')\nfrom evaluation_utils import evaluate_classification"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("X_train_bow, X_test_bow, y_train, y_test, _ = joblib.load('../data/processed/bow_data.pkl')\nX_train_tfidf, X_test_tfidf, _, _, _ = joblib.load('../data/processed/tfidf_data.pkl')"),
        nbf.v4.new_markdown_cell(decision_note(
            "Choosing MultinomialNB",
            "Multinomial Naive Bayes",
            "Works exceptionally well with discrete features (like word counts) and handles high dimensionality well.",
            "GaussianNB", "Can handle continuous data", "Expects normal distribution; BoW/TF-IDF is sparse and not normally distributed.",
            "BernoulliNB", "Works with binary features (presence/absence)", "Ignores frequency of words which might be useful.",
            "MultinomialNB is the industry standard baseline for text classification tasks."
        )),
        nbf.v4.new_markdown_cell("## 2. Train and Evaluate on BoW"),
        nbf.v4.new_code_cell("nb_bow = MultinomialNB()\nnb_bow.fit(X_train_bow, y_train)\ny_pred_bow = nb_bow.predict(X_test_bow)\nevaluate_classification(y_test, y_pred_bow, labels=['ham', 'spam'])\njoblib.dump(nb_bow, '../models/nb_bow.pkl')"),
        nbf.v4.new_markdown_cell("## 3. Train and Evaluate on TF-IDF"),
        nbf.v4.new_code_cell("nb_tfidf = MultinomialNB()\nnb_tfidf.fit(X_train_tfidf, y_train)\ny_pred_tfidf = nb_tfidf.predict(X_test_tfidf)\nevaluate_classification(y_test, y_pred_tfidf, labels=['ham', 'spam'])\njoblib.dump(nb_tfidf, '../models/nb_tfidf.pkl')"),
        nbf.v4.new_markdown_cell("## Key Takeaways\n- [x] MultinomialNB performs extremely well on both.\n- [x] Interestingly, BoW might have slightly higher recall for spam than TF-IDF on this specific dataset, but both are very strong (>95% accuracy).")
    ]
    create_notebook(out_dir / "03_naive_bayes_classifier.ipynb", cells_03)

    # 4. Model Comparison
    cells_04 = [
        nbf.v4.new_markdown_cell("# 04. Model Comparison\n\n**Goal:** Compare NB against Logistic Regression and SVM to find the best model."),
        nbf.v4.new_code_cell("from sklearn.linear_model import LogisticRegression\nfrom sklearn.svm import SVC\nimport joblib\nimport sys\nsys.path.append('../../../shared')\nfrom evaluation_utils import evaluate_classification"),
        nbf.v4.new_markdown_cell("## 1. Load Data (Using TF-IDF)"),
        nbf.v4.new_code_cell("X_train, X_test, y_train, y_test, vectorizer = joblib.load('../data/processed/tfidf_data.pkl')"),
        nbf.v4.new_markdown_cell("## 2. Logistic Regression"),
        nbf.v4.new_code_cell("lr = LogisticRegression(max_iter=1000)\nlr.fit(X_train, y_train)\ny_pred_lr = lr.predict(X_test)\nevaluate_classification(y_test, y_pred_lr, labels=['ham', 'spam'])\njoblib.dump(lr, '../models/logistic_regression.pkl')"),
        nbf.v4.new_markdown_cell("## 3. Support Vector Machine (Linear Kernel)"),
        nbf.v4.new_code_cell("svc = SVC(kernel='linear')\nsvc.fit(X_train, y_train)\ny_pred_svc = svc.predict(X_test)\nevaluate_classification(y_test, y_pred_svc, labels=['ham', 'spam'])\njoblib.dump(svc, '../models/svm_linear.pkl')"),
        nbf.v4.new_markdown_cell("## Key Takeaways\n- [x] SVM (Linear) often provides the absolute best separation for high-dimensional sparse text data.\n- [x] Logistic Regression is a strong, well-calibrated alternative.\n- [x] For spam detection, we care most about minimizing False Positives (we don't want a legitimate text marked as spam). SVM typically excels here.")
    ]
    create_notebook(out_dir / "04_model_comparison.ipynb", cells_04)

if __name__ == "__main__":
    generate_notebooks()
