# Project 03: IMDB Movie Reviews

## Problem Statement
Given a long-form movie review containing raw text and HTML artifacts, classify the reviewer's sentiment as either **Positive** or **Negative**.

## Dataset
- **Name:** IMDB Movie Reviews Dataset
- **Task:** Binary Sentiment Analysis
- **Classes:** 2 (Positive, Negative)

## Learning Goals
- Handling HTML stripping and extensive noise.
- N-grams (Bigrams/Trigrams) feature extraction.
- Handling negations (e.g., "not good" vs "good") in sentiment analysis.
- Managing large matrix sparsity.
- Using regularized linear models (Ridge/Logistic) to prevent overfitting on massive N-gram matrices.
