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
    out_dir = Path("datasets/03_imdb_movie_reviews/notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. EDA and Cleaning
    cells_01 = [
        nbf.v4.new_markdown_cell("# 01. Exploratory Data Analysis & Text Cleaning\n\n**Goal:** Understand the IMDB dataset, deal with HTML artifacts, and make a critical decision regarding stopword removal."),
        nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport sys\nimport os\nsys.path.append('../../../shared')\nfrom text_utils import clean_html_text"),
        nbf.v4.new_markdown_cell("## 1. Download & Load Data\nWe download the dataset via Hugging Face if it's not already present locally."),
        nbf.v4.new_code_cell("data_path = '../data/raw/dataset.csv'\nif not os.path.exists(data_path):\n    print('Downloading dataset via Hugging Face...')\n    from datasets import load_dataset\n    dataset = load_dataset('stanfordnlp/imdb')\n    df = pd.concat([dataset['train'].to_pandas(), dataset['test'].to_pandas()], ignore_index=True)\n    os.makedirs('../data/raw', exist_ok=True)\n    df.to_csv(data_path, index=False)\n\ndf = pd.read_csv(data_path)\nprint(df.head())\nprint(df.info())"),
        nbf.v4.new_markdown_cell("## 2. Target Distribution"),
        nbf.v4.new_code_cell("sns.countplot(x='label', data=df)\nplt.title('IMDB Sentiment Distribution (0: Negative, 1: Positive)')\nplt.show()"),
        nbf.v4.new_markdown_cell("## 3. Look at HTML Artifacts"),
        nbf.v4.new_code_cell("# Let's see an example of a dirty review\nprint(df['text'].iloc[0][:500])"),
        nbf.v4.new_markdown_cell(decision_note(
            "Text Cleaning Strategy",
            "Regex HTML Removal + Basic Cleaning, BUT NO Stopword Removal",
            "HTML tags like `<br />` add no semantic value. However, keeping stopwords ensures we don't lose crucial negations (like 'not', 'very', 'too').",
            "Standard Cleaning (Remove Stopwords)", "Reduces matrix size massively", "Destroys negations! 'This is not a good movie' becomes 'good movie'.",
            "Advanced Lemmatization", "Standardizes words", "For 50,000 reviews, lemmatizing is very slow and the sentiment is usually captured fine by raw words.",
            "In sentiment analysis, negations and modifiers are extremely important. Standard stopword removal is detrimental."
        )),
        nbf.v4.new_markdown_cell("## 4. Apply HTML Cleaning"),
        nbf.v4.new_code_cell("df['cleaned_text'] = df['text'].apply(clean_html_text)\ndf[['label', 'text', 'cleaned_text']].head()"),
        nbf.v4.new_code_cell("df.to_csv('../data/processed/cleaned_imdb.csv', index=False)\nprint('Saved cleaned dataset.')")
    ]
    create_notebook(out_dir / "01_eda_and_cleaning.ipynb", cells_01)

    # 2. N-Grams
    cells_02 = [
        nbf.v4.new_markdown_cell("# 02. N-Gram Vectorization\n\n**Goal:** Explore how Bigrams (N-grams where N=2) capture negations, and how this impacts vocabulary size."),
        nbf.v4.new_code_cell("import pandas as pd\nfrom sklearn.feature_extraction.text import CountVectorizer\nfrom sklearn.model_selection import train_test_split\nimport joblib"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("df = pd.read_csv('../data/processed/cleaned_imdb.csv')\ndf = df.dropna(subset=['cleaned_text'])\nX_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)"),
        nbf.v4.new_markdown_cell(decision_note(
            "N-Gram Selection",
            "Use CountVectorizer with ngram_range=(1, 2) (Unigrams + Bigrams)",
            "Captures two-word combos like 'not good' or 'very bad', drastically improving sentiment polarity detection.",
            "Unigrams only (1, 1)", "Small, fast matrix", "Misses all context and negations.",
            "Trigrams (1, 3)", "Even more context", "Causes exponential explosion in feature size (millions of columns), leading to massive sparsity and memory issues.",
            "Bigrams offer the best tradeoff between capturing crucial local context (negations) and keeping feature size manageable."
        )),
        nbf.v4.new_markdown_cell("## 2. Unigram vs Bigram Matrix Size Comparison"),
        nbf.v4.new_code_cell("vec_uni = CountVectorizer(ngram_range=(1,1), min_df=5)\nvec_uni.fit(X_train)\nprint('Unigram Vocabulary Size:', len(vec_uni.vocabulary_))\n\nvec_bi = CountVectorizer(ngram_range=(1,2), min_df=5)\nvec_bi.fit(X_train)\nprint('Unigram+Bigram Vocabulary Size:', len(vec_bi.vocabulary_))"),
        nbf.v4.new_markdown_cell("## 3. Save Bigram Vectorized Data"),
        nbf.v4.new_code_cell("X_train_vec = vec_bi.transform(X_train)\nX_test_vec = vec_bi.transform(X_test)\njoblib.dump((X_train_vec, X_test_vec, y_train, y_test, vec_bi), '../data/processed/bigram_data.pkl')\nprint('Saved Bigram matrices.')")
    ]
    create_notebook(out_dir / "02_ngram_vectorization.ipynb", cells_02)

    # 3. Regularized Models
    cells_03 = [
        nbf.v4.new_markdown_cell("# 03. Regularized Linear Models\n\n**Goal:** Train models on the massive Bigram sparse matrix. We will use L2 Regularization (Ridge / Logistic Regression) to prevent overfitting on this massive feature space."),
        nbf.v4.new_code_cell("from sklearn.linear_model import LogisticRegression, RidgeClassifier\nfrom sklearn.naive_bayes import MultinomialNB\nimport joblib\nimport sys\nsys.path.append('../../../shared')\nfrom evaluation_utils import evaluate_classification"),
        nbf.v4.new_markdown_cell("## 1. Load Bigram Data"),
        nbf.v4.new_code_cell("X_train, X_test, y_train, y_test, vectorizer = joblib.load('../data/processed/bigram_data.pkl')"),
        nbf.v4.new_markdown_cell(decision_note(
            "Model Selection for Massive Sparsity",
            "Logistic Regression and Ridge Classifier (L2 Regularization)",
            "With >100,000 features, models easily memorize the training data. L2 regularization drastically penalizes large coefficients, forcing the model to rely on truly generalizable sentiment indicators.",
            "Standard SVM / Unregularized LR", "Standard models", "Will heavily overfit the training data due to the curse of dimensionality.",
            "Decision Trees / Random Forest", "Handles non-linear", "Performs terribly and is extremely slow on 100k+ sparse features.",
            "Linear models with strong L2 regularization are incredibly fast and effective on massive N-gram matrices."
        )),
        nbf.v4.new_markdown_cell("## 2. Multinomial Naive Bayes (Baseline)"),
        nbf.v4.new_code_cell("nb = MultinomialNB()\nnb.fit(X_train, y_train)\ny_pred_nb = nb.predict(X_test)\nevaluate_classification(y_test, y_pred_nb)\njoblib.dump(nb, '../models/multinomial_nb.pkl')"),
        nbf.v4.new_markdown_cell("## 3. Logistic Regression (L2 Penalty)"),
        nbf.v4.new_code_cell("lr = LogisticRegression(max_iter=1000, C=0.5) # C=0.5 increases regularization\nlr.fit(X_train, y_train)\ny_pred_lr = lr.predict(X_test)\nevaluate_classification(y_test, y_pred_lr)\njoblib.dump(lr, '../models/logistic_regression_l2.pkl')"),
        nbf.v4.new_markdown_cell("## 4. Ridge Classifier"),
        nbf.v4.new_code_cell("ridge = RidgeClassifier(alpha=1.0)\nridge.fit(X_train, y_train)\ny_pred_ridge = ridge.predict(X_test)\nevaluate_classification(y_test, y_pred_ridge)\njoblib.dump(ridge, '../models/ridge_classifier.pkl')"),
        nbf.v4.new_markdown_cell("## Key Takeaways\n- [x] Bigrams allow us to catch negations, bumping accuracy nicely.\n- [x] L2 regularized linear models (Logistic/Ridge) excel in high dimensional spaces where $p > n$ or $p \\approx n$.")
    ]
    create_notebook(out_dir / "03_regularized_linear_models.ipynb", cells_03)

if __name__ == "__main__":
    generate_notebooks()
