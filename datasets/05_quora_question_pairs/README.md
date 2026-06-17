# Project 05: Quora Question Pairs

## Problem Statement
Given two separate questions asked by different users, determine if they have the exact same intent (**Duplicate**) or not. This requires pairwise text classification rather than single-document classification.

## Dataset
- **Name:** Quora Question Pairs (Sampled to 100k)
- **Task:** Pairwise Text Similarity (Binary Classification)
- **Classes:** 2 (Duplicate, Not Duplicate)

## Learning Goals
- Transitioning from matrix representation to **Feature Engineering**.
- Calculating textual distance metrics: Jaccard Similarity, Word Overlap, Length differences.
- Semantic metrics: TF-IDF Cosine Similarity and Character-Level String Ratios.
- Feeding engineered distance metrics into Tabular ML Models (Random Forest) instead of using massive concatenated TF-IDF arrays.
