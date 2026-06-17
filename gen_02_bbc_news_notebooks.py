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
    out_dir = Path("datasets/02_bbc_news_classification/notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. EDA and Cleaning
    cells_01 = [
        nbf.v4.new_markdown_cell("# 01. Exploratory Data Analysis & Text Cleaning\n\n**Goal:** Understand the 5 BBC news categories and perform basic cleaning."),
        nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport sys\nsys.path.append('../../../shared')\nfrom text_utils import clean_text_basic"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("df = pd.read_csv('../data/raw/dataset.csv', encoding='latin1')\ndf.rename(columns={'news': 'text', 'type': 'category'}, inplace=True)\nprint(df.head())\nprint(df.info())"),
        nbf.v4.new_markdown_cell("## 2. Category Distribution"),
        nbf.v4.new_code_cell("sns.countplot(x='category', data=df)\nplt.title('BBC News Category Distribution')\nplt.show()\nprint(df['category'].value_counts())"),
        nbf.v4.new_markdown_cell(decision_note(
            "Basic text cleaning",
            "Lowercasing and punctuation removal via custom shared utility",
            "Creates a clean base for advanced morphological analysis in the next step.",
            "Keep punctuation", "Helps with grammar parsing", "Irrelevant for simple text classification; increases vocabulary size.",
            "Remove all numbers", "Reduces noise", "In business/tech news, numbers (e.g. '5G', '2004') can be important features. We'll leave them for now.",
            "Basic cleaning is a necessary prerequisite before stemming or lemmatization."
        )),
        nbf.v4.new_markdown_cell("## 3. Apply Basic Cleaning"),
        nbf.v4.new_code_cell("df['cleaned_text'] = df['text'].apply(clean_text_basic)\ndf[['category', 'text', 'cleaned_text']].head()"),
        nbf.v4.new_code_cell("df.to_csv('../data/processed/cleaned_bbc.csv', index=False)\nprint('Saved cleaned dataset.')")
    ]
    create_notebook(out_dir / "01_eda_and_cleaning.ipynb", cells_01)

    # 2. Stemming vs Lemmatization
    cells_02 = [
        nbf.v4.new_markdown_cell("# 02. Stemming vs Lemmatization\n\n**Goal:** Compare the effects of PorterStemmer and WordNetLemmatizer on vocabulary reduction."),
        nbf.v4.new_code_cell("import pandas as pd\nfrom sklearn.feature_extraction.text import CountVectorizer\nimport sys\nsys.path.append('../../../shared')\nfrom text_utils import apply_stemming, apply_lemmatization"),
        nbf.v4.new_markdown_cell("## 1. Load Cleaned Data"),
        nbf.v4.new_code_cell("df = pd.read_csv('../data/processed/cleaned_bbc.csv')\n# We'll test on a subset to save time if needed, but 2225 rows is small enough.\ndf = df.dropna(subset=['cleaned_text'])"),
        nbf.v4.new_markdown_cell(decision_note(
            "Morphological Reduction",
            "Compare Stemming vs Lemmatization",
            "Allows us to see which method reduces vocabulary more effectively without losing meaning.",
            "Stemming (Porter)", "Fast, aggressive reduction", "Can create non-words (e.g., 'organization' -> 'organ'), causing semantic collision.",
            "Lemmatization (WordNet)", "Creates real words, preserves meaning better", "Slower, requires POS tags to be truly effective.",
            "Comparing them empirically is the best way to choose for a specific dataset."
        )),
        nbf.v4.new_markdown_cell("## 2. Apply Stemming"),
        nbf.v4.new_code_cell("df['stemmed'] = df['cleaned_text'].apply(apply_stemming)"),
        nbf.v4.new_markdown_cell("## 3. Apply Lemmatization"),
        nbf.v4.new_code_cell("df['lemmatized'] = df['cleaned_text'].apply(apply_lemmatization)"),
        nbf.v4.new_markdown_cell("## 4. Compare Vocabulary Sizes"),
        nbf.v4.new_code_cell("vec_clean = CountVectorizer()\nvec_clean.fit(df['cleaned_text'])\nprint('Original Vocab:', len(vec_clean.vocabulary_))\n\nvec_stem = CountVectorizer()\nvec_stem.fit(df['stemmed'])\nprint('Stemmed Vocab:', len(vec_stem.vocabulary_))\n\nvec_lem = CountVectorizer()\nvec_lem.fit(df['lemmatized'])\nprint('Lemmatized Vocab:', len(vec_lem.vocabulary_))"),
        nbf.v4.new_code_cell("df.to_csv('../data/processed/morph_bbc.csv', index=False)\nprint('Saved morphologically processed data.')")
    ]
    create_notebook(out_dir / "02_stemming_vs_lemmatization.ipynb", cells_02)

    # 3. Vectorization
    cells_03 = [
        nbf.v4.new_markdown_cell("# 03. Text Vectorization\n\n**Goal:** Apply TF-IDF and tune `max_features` and `min_df`."),
        nbf.v4.new_code_cell("import pandas as pd\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.model_selection import train_test_split\nimport joblib"),
        nbf.v4.new_markdown_cell("## 1. Load Lemmatized Data"),
        nbf.v4.new_code_cell("df = pd.read_csv('../data/processed/morph_bbc.csv')\ndf = df.dropna(subset=['lemmatized'])\nX_train, X_test, y_train, y_test = train_test_split(df['lemmatized'], df['category'], test_size=0.2, random_state=42)"),
        nbf.v4.new_markdown_cell(decision_note(
            "Vectorization Tuning",
            "TF-IDF with min_df=5 and max_features=5000",
            "Removes ultra-rare words (typos, obscure names) and limits memory footprint while capturing the most important signals.",
            "No limits (default)", "Captures all information", "Creates massive, highly sparse matrix. Prone to overfitting on rare words.",
            "High min_df (e.g. 50)", "Very dense matrix", "Might filter out words that strongly indicate a specific, rare topic.",
            "A `min_df` of 5 drops words appearing in less than 5 documents, effectively pruning typos and rare entities."
        )),
        nbf.v4.new_markdown_cell("## 2. TF-IDF Vectorization"),
        nbf.v4.new_code_cell("vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_features=5000)\nX_train_tfidf = vectorizer.fit_transform(X_train)\nX_test_tfidf = vectorizer.transform(X_test)\nprint('Final Vocabulary Size:', len(vectorizer.vocabulary_))"),
        nbf.v4.new_code_cell("joblib.dump((X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer), '../data/processed/tfidf_bbc.pkl')\nprint('Saved vectorized data.')")
    ]
    create_notebook(out_dir / "03_text_vectorization.ipynb", cells_03)

    # 4. Multi-class Modeling
    cells_04 = [
        nbf.v4.new_markdown_cell("# 04. Multi-Class Modeling\n\n**Goal:** Train and compare Logistic Regression and Naive Bayes on the 5-class dataset."),
        nbf.v4.new_code_cell("from sklearn.linear_model import LogisticRegression\nfrom sklearn.naive_bayes import MultinomialNB\nimport joblib\nimport sys\nsys.path.append('../../../shared')\nfrom evaluation_utils import evaluate_classification"),
        nbf.v4.new_markdown_cell("## 1. Load Vectorized Data"),
        nbf.v4.new_code_cell("X_train, X_test, y_train, y_test, vectorizer = joblib.load('../data/processed/tfidf_bbc.pkl')\nlabels = sorted(y_train.unique())"),
        nbf.v4.new_markdown_cell(decision_note(
            "Multi-Class Strategy",
            "MultinomialNB and LogisticRegression (multinomial/OVR)",
            "Both handle multi-class problems natively or via One-Vs-Rest seamlessly.",
            "Binary Classifiers", "Good for 1v1", "Complex to orchestrate for 5 classes manually.",
            "Tree-based (Random Forest)", "Handles non-linear", "Usually underperforms on high-dimensional sparse text compared to linear models.",
            "Linear models and Naive Bayes are the golden standard for BoW/TF-IDF text classification."
        )),
        nbf.v4.new_markdown_cell("## 2. Multinomial Naive Bayes"),
        nbf.v4.new_code_cell("nb = MultinomialNB()\nnb.fit(X_train, y_train)\ny_pred_nb = nb.predict(X_test)\nevaluate_classification(y_test, y_pred_nb, labels=labels)\njoblib.dump(nb, '../models/multinomial_nb.pkl')"),
        nbf.v4.new_markdown_cell("## 3. Logistic Regression"),
        nbf.v4.new_code_cell("lr = LogisticRegression(max_iter=1000)\nlr.fit(X_train, y_train)\ny_pred_lr = lr.predict(X_test)\nevaluate_classification(y_test, y_pred_lr, labels=labels)\njoblib.dump(lr, '../models/logistic_regression_ovr.pkl')"),
        nbf.v4.new_markdown_cell("## Key Takeaways\n- [x] Both models should achieve >95% accuracy.\n- [x] Look at the confusion matrix to see which categories get confused the most (e.g., Tech vs Business).")
    ]
    create_notebook(out_dir / "04_multiclass_modeling.ipynb", cells_04)

if __name__ == "__main__":
    generate_notebooks()
