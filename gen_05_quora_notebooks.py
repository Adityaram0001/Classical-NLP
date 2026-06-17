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
    out_dir = Path("datasets/05_quora_question_pairs/notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lexical Feature Engineering
    cells_01 = [
        nbf.v4.new_markdown_cell("# 01. Lexical Feature Engineering\n\n**Goal:** Explore the target distribution and engineer our first set of features based purely on raw word overlap. In Pairwise Text classification, we do NOT feed the raw text to the model! We feed *distance metrics* between the two texts."),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport sys\nsys.path.append('../../../shared')\nfrom text_utils import clean_text_basic"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("df = pd.read_csv('../data/raw/dataset.csv')\ndf['question1'] = df['question1'].fillna('')\ndf['question2'] = df['question2'].fillna('')\nprint(df.head())\nprint(f'Total pairs: {len(df)}')"),
        nbf.v4.new_markdown_cell("## 2. Target Distribution"),
        nbf.v4.new_code_cell("sns.countplot(x='is_duplicate', data=df)\nplt.title('Duplicate (1) vs Non-Duplicate (0)')\nplt.show()"),
        nbf.v4.new_markdown_cell("## 3. Basic Cleaning"),
        nbf.v4.new_code_cell("df['q1_clean'] = df['question1'].apply(clean_text_basic)\ndf['q2_clean'] = df['question2'].apply(clean_text_basic)"),
        nbf.v4.new_markdown_cell(decision_note(
            "Feature Engineering Approach",
            "Calculating distance metrics (e.g., Jaccard Similarity, Word Overlap)",
            "Creates dense numerical features that tabular models (like Random Forest) can easily split on.",
            "Raw TF-IDF Matrix Concatenation", "No engineering required", "Concatenating two 50k sparse vectors creates a massive 100k vector that is heavily prone to overfitting and extremely slow to train.",
            "Siamese Neural Networks (Deep Learning)", "State of the art", "Requires deep learning architectures (LSTMs or Transformers). Not suitable for Classical NLP constraints.",
            "Classical pairwise classification relies almost entirely on human-engineered distance metrics."
        )),
        nbf.v4.new_markdown_cell("## 4. Engineer Jaccard Similarity & Word Overlap"),
        nbf.v4.new_code_cell("def jaccard_similarity(row):\n    w1 = set(str(row['q1_clean']).split())\n    w2 = set(str(row['q2_clean']).split())\n    if len(w1) == 0 or len(w2) == 0:\n        return 0.0\n    intersection = len(w1.intersection(w2))\n    union = len(w1.union(w2))\n    return intersection / union\n\ndef word_overlap(row):\n    w1 = set(str(row['q1_clean']).split())\n    w2 = set(str(row['q2_clean']).split())\n    return len(w1.intersection(w2))\n\ndf['jaccard_sim'] = df.apply(jaccard_similarity, axis=1)\ndf['word_overlap'] = df.apply(word_overlap, axis=1)\n\n# Also compute absolute difference in length\ndf['len_diff'] = abs(df['q1_clean'].str.len() - df['q2_clean'].str.len())\ndf[['q1_clean', 'q2_clean', 'jaccard_sim', 'word_overlap', 'len_diff']].head()"),
        nbf.v4.new_code_cell("df.to_csv('../data/processed/features_part1.csv', index=False)\nprint('Saved initial features.')")
    ]
    create_notebook(out_dir / "01_lexical_feature_engineering.ipynb", cells_01)

    # 2. Semantic and String Distances
    cells_02 = [
        nbf.v4.new_markdown_cell("# 02. Semantic and String Distances\n\n**Goal:** Compute advanced distance metrics: TF-IDF Cosine Similarity and Character-level String Match Ratio (similar to Levenshtein distance)."),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.metrics.pairwise import paired_cosine_distances\nimport difflib"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("df = pd.read_csv('../data/processed/features_part1.csv')\ndf['q1_clean'] = df['q1_clean'].fillna('')\ndf['q2_clean'] = df['q2_clean'].fillna('')"),
        nbf.v4.new_markdown_cell(decision_note(
            "Semantic Similarity Metric",
            "TF-IDF Cosine Similarity",
            "Cosine similarity measures the angle between two vectors. It perfectly captures how semantically similar two sentences are, completely ignoring document length differences.",
            "Euclidean Distance", "Simple spatial distance", "Heavily penalized by length. A short sentence and a long sentence with similar meaning will have a huge Euclidean distance.",
            "Manhattan Distance", "Grid-based distance", "Same issue as Euclidean, highly sensitive to magnitude/length of the document.",
            "Cosine similarity normalizes for magnitude, making it the mathematical gold-standard for vector similarity."
        )),
        nbf.v4.new_markdown_cell("## 2. Compute TF-IDF Cosine Similarity"),
        nbf.v4.new_code_cell("# Fit a single TF-IDF on all questions to build a shared vocabulary\nall_questions = pd.concat([df['q1_clean'], df['q2_clean']])\ntfidf = TfidfVectorizer(max_features=10000)\ntfidf.fit(all_questions)\n\n# Transform Q1 and Q2 separately\ntfidf_q1 = tfidf.transform(df['q1_clean'])\ntfidf_q2 = tfidf.transform(df['q2_clean'])\n\n# Compute pairwise cosine *distance* (1 - similarity)\n# Similarity = 1 - Distance\ncos_distances = paired_cosine_distances(tfidf_q1, tfidf_q2)\ndf['tfidf_cosine_sim'] = 1.0 - cos_distances"),
        nbf.v4.new_markdown_cell("## 3. Compute Character-Level String Ratio"),
        nbf.v4.new_code_cell("def string_ratio(row):\n    s1 = str(row['q1_clean'])\n    s2 = str(row['q2_clean'])\n    return difflib.SequenceMatcher(None, s1, s2).ratio()\n\ndf['string_ratio'] = df.apply(string_ratio, axis=1)\n\ndf[['jaccard_sim', 'tfidf_cosine_sim', 'string_ratio', 'is_duplicate']].head()"),
        nbf.v4.new_code_cell("df.to_csv('../data/processed/final_features.csv', index=False)\nprint('Saved final features.')")
    ]
    create_notebook(out_dir / "02_semantic_and_string_distances.ipynb", cells_02)

    # 3. Similarity Classification Model
    cells_03 = [
        nbf.v4.new_markdown_cell("# 03. Similarity Classification Model\n\n**Goal:** Feed our engineered, dense numerical features into a robust Tabular Model (Random Forest) to classify duplicates."),
        nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import train_test_split\nimport joblib\nimport sys\nsys.path.append('../../../shared')\nfrom evaluation_utils import evaluate_classification"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("df = pd.read_csv('../data/processed/final_features.csv')\n\n# Define the engineered feature columns\nfeatures = ['jaccard_sim', 'word_overlap', 'len_diff', 'tfidf_cosine_sim', 'string_ratio']\nX = df[features]\ny = df['is_duplicate']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"),
        nbf.v4.new_markdown_cell(decision_note(
            "Final Classification Model",
            "Random Forest Classifier",
            "We have transitioned from a sparse massive text matrix to 5 dense numerical features. Tabular tree-based models like Random Forest excel at finding non-linear relationships in dense tabular data.",
            "Logistic Regression", "Fast and interpretable", "Assumes linear boundaries. Our distance metrics might interact in highly non-linear ways (e.g. high overlap but completely different cosine sim).",
            "Naive Bayes", "Great for text", "Terrible for continuous correlated numerical features (it assumes feature independence).",
            "Random Forests natively handle interacting numerical features, making them perfect for this engineered feature set."
        )),
        nbf.v4.new_markdown_cell("## 2. Train Random Forest"),
        nbf.v4.new_code_cell("rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)\nrf.fit(X_train, y_train)"),
        nbf.v4.new_markdown_cell("## 3. Evaluate"),
        nbf.v4.new_code_cell("y_pred = rf.predict(X_test)\nevaluate_classification(y_test, y_pred, labels=['Not Duplicate', 'Duplicate'])\njoblib.dump(rf, '../models/random_forest.pkl')"),
        nbf.v4.new_markdown_cell("## 4. Feature Importance"),
        nbf.v4.new_code_cell("importances = rf.feature_importances_\nfeature_imp = pd.Series(importances, index=features).sort_values(ascending=False)\n\nplt.figure(figsize=(8, 5))\nsns.barplot(x=feature_imp, y=feature_imp.index)\nplt.title('Feature Importances in Duplicate Detection')\nplt.xlabel('Importance')\nplt.ylabel('Feature')\nplt.show()"),
        nbf.v4.new_markdown_cell("## Key Takeaways\n- [x] In Classical NLP Pairwise tasks, **Feature Engineering** is king.\n- [x] We successfully translated raw text into numerical similarity scores (Jaccard, Cosine, String Match).\n- [x] A Random Forest easily digests these similarities to predict duplicate intents.")
    ]
    create_notebook(out_dir / "03_similarity_classification_model.ipynb", cells_03)

if __name__ == "__main__":
    generate_notebooks()
