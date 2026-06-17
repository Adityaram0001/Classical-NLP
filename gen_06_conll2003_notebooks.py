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
    out_dir = Path("datasets/06_conll2003_ner/notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. EDA and Sequence Structure
    cells_01 = [
        nbf.v4.new_markdown_cell("# 01. EDA and Sequence Structure\n\n**Goal:** Understand the unique data shape required for Sequence Tagging. Unlike standard classification where `X` is a single text string, here `X` is an ordered list of tokens (a sentence), and `y` is an ordered list of tags of the exact same length."),
        nbf.v4.new_code_cell("import pandas as pd\nimport json\nimport matplotlib.pyplot as plt\nimport seaborn as sns"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("def load_jsonl(filepath):\n    data = []\n    with open(filepath, 'r') as f:\n        for line in f:\n            data.append(json.loads(line))\n    return data\n\ntrain_data = load_jsonl('../data/raw/train.jsonl')\ntest_data = load_jsonl('../data/raw/test.jsonl')\nprint(f'Train sentences: {len(train_data)}\\nTest sentences: {len(test_data)}')"),
        nbf.v4.new_markdown_cell("## 2. Inspect a Sequence\nNotice how every token maps perfectly to a POS (Part of Speech) tag and an NER (Named Entity Recognition) tag."),
        nbf.v4.new_code_cell("sample = train_data[0]\nfor token, pos, ner in zip(sample['tokens'], sample['pos_tags'], sample['ner_tags']):\n    print(f'{token:15} {pos:10} {ner}')"),
        nbf.v4.new_markdown_cell("## 3. Tag Distribution"),
        nbf.v4.new_code_cell("all_ner_tags = [tag for sentence in train_data for tag in sentence['ner_tags'] if tag != 'O']\nplt.figure(figsize=(10, 5))\nsns.countplot(y=all_ner_tags, order=pd.Series(all_ner_tags).value_counts().index)\nplt.title('Distribution of Named Entities (Excluding O)')\nplt.show()")
    ]
    create_notebook(out_dir / "01_eda_and_sequence_structure.ipynb", cells_01)

    # 2. Contextual Feature Engineering
    cells_02 = [
        nbf.v4.new_markdown_cell("# 02. Contextual Feature Engineering\n\n**Goal:** Transform lists of tokens into lists of feature dictionaries. In classical NER, we look at the word itself, its suffixes, whether it's capitalized, and vitally: **what words surround it**."),
        nbf.v4.new_code_cell("import json\nimport joblib\n\ndef load_jsonl(filepath):\n    data = []\n    with open(filepath, 'r') as f:\n        for line in f:\n            data.append(json.loads(line))\n    return data\n\ntrain_data = load_jsonl('../data/raw/train.jsonl')\ntest_data = load_jsonl('../data/raw/test.jsonl')"),
        nbf.v4.new_markdown_cell(decision_note(
            "Feature Engineering Method",
            "Context Window Dictionaries",
            "We extract explicitly defined features for every token (is_upper, suffix, previous_word, next_word).",
            "Word Embeddings (Word2Vec)", "Captures semantics", "Requires averaging/summing which loses token-level precision unless used in a BiLSTM.",
            "Raw Bag-of-Words", "Simple", "Destroys sequence order entirely. In NER, order is everything (e.g., 'New York').",
            "For CRFs, extracting localized contextual windows as discrete boolean/string features is the most robust classical approach."
        )),
        nbf.v4.new_markdown_cell("## 1. Feature Extraction Function\nThis is the secret sauce of Classical NER! We look at word `i` and extract features not just about `i`, but about `i-1` and `i+1`."),
        nbf.v4.new_code_cell("def word2features(sent_tokens, sent_pos, i):\n    word = sent_tokens[i]\n    postag = sent_pos[i]\n\n    features = {\n        'bias': 1.0,\n        'word.lower()': word.lower(),\n        'word[-3:]': word[-3:],\n        'word[-2:]': word[-2:],\n        'word.isupper()': word.isupper(),\n        'word.istitle()': word.istitle(),\n        'word.isdigit()': word.isdigit(),\n        'postag': postag,\n        'postag[:2]': postag[:2],\n    }\n    if i > 0:\n        word1 = sent_tokens[i-1]\n        postag1 = sent_pos[i-1]\n        features.update({\n            '-1:word.lower()': word1.lower(),\n            '-1:word.istitle()': word1.istitle(),\n            '-1:word.isupper()': word1.isupper(),\n            '-1:postag': postag1,\n            '-1:postag[:2]': postag1[:2],\n        })\n    else:\n        features['BOS'] = True # Beginning of sentence\n\n    if i < len(sent_tokens)-1:\n        word1 = sent_tokens[i+1]\n        postag1 = sent_pos[i+1]\n        features.update({\n            '+1:word.lower()': word1.lower(),\n            '+1:word.istitle()': word1.istitle(),\n            '+1:word.isupper()': word1.isupper(),\n            '+1:postag': postag1,\n            '+1:postag[:2]': postag1[:2],\n        })\n    else:\n        features['EOS'] = True # End of sentence\n\n    return features"),
        nbf.v4.new_markdown_cell("## 2. Apply to Dataset"),
        nbf.v4.new_code_cell("def extract_features(data):\n    X = []\n    y = []\n    for sentence in data:\n        tokens = sentence['tokens']\n        pos = sentence['pos_tags']\n        ner = sentence['ner_tags']\n        \n        # Convert sentence into list of dicts\n        sent_features = [word2features(tokens, pos, i) for i in range(len(tokens))]\n        X.append(sent_features)\n        y.append(ner)\n    return X, y\n\nX_train, y_train = extract_features(train_data)\nX_test, y_test = extract_features(test_data)\n\nprint(f'Sentence 1, Token 1 Features:\\n{X_train[0][0]}')"),
        nbf.v4.new_code_cell("joblib.dump((X_train, y_train, X_test, y_test), '../data/processed/crf_data.pkl')\nprint('Saved extracted features.')")
    ]
    create_notebook(out_dir / "02_contextual_feature_engineering.ipynb", cells_02)

    # 3. CRF Model
    cells_03 = [
        nbf.v4.new_markdown_cell("# 03. Conditional Random Fields (CRF)\n\n**Goal:** Train a sequence model that learns not only from our contextual features, but also learns the *transition probabilities* between tags (e.g., I-ORG is highly likely to follow B-ORG, but impossible to follow B-LOC)."),
        nbf.v4.new_code_cell("import joblib\nimport sklearn_crfsuite\nfrom sklearn_crfsuite import metrics as crf_metrics\nfrom seqeval.metrics import classification_report, f1_score\nimport matplotlib.pyplot as plt"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("X_train, y_train, X_test, y_test = joblib.load('../data/processed/crf_data.pkl')\nprint(f'Loaded {len(X_train)} train sentences and {len(X_test)} test sentences.')"),
        nbf.v4.new_markdown_cell(decision_note(
            "Model Selection for NER",
            "Conditional Random Fields (CRF)",
            "CRFs explicitly model the transition probabilities between output labels in a sequence. It 'learns the grammar' of tags (e.g. I-PER must follow B-PER).",
            "Hidden Markov Models (HMM)", "Generative sequence model", "Cannot handle rich, overlapping, contextual features easily. Much less expressive than CRFs.",
            "Random Forest", "Great tabular model", "Ignores the sequence of tags entirely! It predicts token $i$ without knowing what it predicted for token $i-1$.",
            "CRFs are the absolute pinnacle of Classical NLP for sequence tagging because they combine rich feature engineering with sequential label awareness."
        )),
        nbf.v4.new_markdown_cell("## 2. Train CRF"),
        nbf.v4.new_code_cell("crf = sklearn_crfsuite.CRF(\n    algorithm='lbfgs',\n    c1=0.1,  # L1 regularization\n    c2=0.1,  # L2 regularization\n    max_iterations=100,\n    all_possible_transitions=True\n)\n\nprint('Training CRF model (this takes ~15 seconds on M4 Macs)...')\ncrf.fit(X_train, y_train)\nprint('Done!')"),
        nbf.v4.new_markdown_cell("## 3. Evaluation\nWe use `seqeval` because standard classification metrics penalize NER systems incorrectly if they get the boundary slightly wrong. `seqeval` evaluates on the *Entity* level, not the token level."),
        nbf.v4.new_code_cell("y_pred = crf.predict(X_test)\nprint(classification_report(y_test, y_pred))\n\nprint(f'Entity-Level F1 Score: {f1_score(y_test, y_pred):.3f}')"),
        nbf.v4.new_markdown_cell("## 4. Inspect Learned Transitions\nLet's see what the CRF learned about the 'grammar' of tags!"),
        nbf.v4.new_code_cell("from collections import Counter\ndef print_transitions(trans_features):\n    for (label_from, label_to), weight in trans_features:\n        print(f'{label_from:9} -> {label_to:9} : {weight:.4f}')\n\nprint('Top 5 Likely Transitions:')\nprint_transitions(Counter(crf.transition_features_).most_common(5))\nprint('\\nTop 5 Unlikely Transitions:')\nprint_transitions(Counter(crf.transition_features_).most_common()[-5:])"),
        nbf.v4.new_markdown_cell("## Key Takeaways\n- [x] In Sequence Tagging, `X` is a sequence of tokens and `y` is a sequence of tags.\n- [x] We engineer features by creating **Context Windows** around every single word.\n- [x] **CRFs** are the classical masters of this task because they learn both the features of the word AND the transition grammar of the tags!")
    ]
    create_notebook(out_dir / "03_conditional_random_fields_crf.ipynb", cells_03)

if __name__ == "__main__":
    generate_notebooks()
