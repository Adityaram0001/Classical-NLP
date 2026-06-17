# SMS Spam Collection Dataset

## Overview
A collection of 5,574 SMS text messages tagged as spam or ham (legitimate).
It's a classic dataset for learning basic text classification.

## Learning Goals
- Text cleaning (regex, lowercase, punctuation removal)
- Tokenization and stopword removal
- Bag-of-Words (CountVectorizer) vs TF-IDF
- Building and evaluating a Naive Bayes classifier

## Setup
Download the dataset from Kaggle or UCI ML repository and place the raw file (`SMSSpamCollection`) inside `data/raw/`.

```bash
# Example Kaggle command if available
# kaggle datasets download -d uciml/sms-spam-collection-dataset -p data/raw
```
