# IMDB Movie Reviews Dataset

## Overview
A dataset of 50,000 highly polar movie reviews for binary sentiment classification.

## Learning Goals
- N-gram Vectorization (Bigrams) to capture negations.
- HTML Artifact Cleaning.
- Handling Massive Sparsity and Curse of Dimensionality.
- L2 Regularization (Ridge Classifier, Logistic Regression) for sparse data.

## Data Acquisition
The dataset is too large to store in Git (the raw CSV is 62MB, and the vectorized models exceed 100MB). 
If you are cloning this repository, the first notebook (`01_eda_and_cleaning.ipynb`) automatically contains the Python code to download the dataset on-the-fly using the Hugging Face `datasets` library and save it locally.

You can also manually download it by running the script at the root of the project:
```bash
pip install datasets pyarrow
python download_imdb.py
```
