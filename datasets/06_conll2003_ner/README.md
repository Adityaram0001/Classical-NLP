# Project 06: CoNLL-2003 Named Entity Recognition

## Problem Statement
Given a sequence of words (a sentence), **predict the correct Named Entity tag (Person, Organization, Location, Miscellaneous, or Outside)** for *every single word* in that sequence, taking into account the context of the surrounding words.

## Dataset
- **Name:** CoNLL-2003 NER Dataset
- **Task:** Sequence Tagging (Token Classification)
- **Classes:** 9 (B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O)

## Learning Goals
- Understanding the `X = sequence of tokens`, `y = sequence of tags` data structure.
- Contextual feature engineering: Extracting token-level features, prefixes, suffixes, capitalization, and creating overlapping sliding windows of `Word[-1]` and `Word[+1]`.
- Understanding why standard Classifiers fail at sequence tasks.
- Training and evaluating **Conditional Random Fields (CRFs)** to learn both token features and label transition probabilities (e.g. learning that I-ORG must follow B-ORG).
- Using `seqeval` for Entity-level F1 scoring.
