# Exercise 4.13: BERT Sentence Similarity with Cosine Distance

This repository provides the starter code for Exercise 4.13. Your task is to implement the **sentence vector extraction**, **normalization**, and **cosine similarity calculation** steps to compute the semantic similarity between two sentences using the BERT model.

## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Code** | Open `bert_similarity_starter.py` and download the pre-trained BERT model (`bert-base-uncased`) from [https://huggingface.co/google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) to the local `./BERT` directory. |
| **Run** | Execute: `python bert_similarity_starter.py` |
| **Observe** | The terminal will output the two sentences and their calculated semantic similarity score. |

> **Hint:** The cosine similarity between two normalized vectors is equal to their dot product: $similarity = A \cdot B$.

## Files

| File | Purpose |
|------|---------|
| `bert_similarity_starter.py` | Starter script. Contains the core logic for tokenization, model inference, and similarity calculation. |
| `./BERT/` | Directory containing the pre-trained BERT model and tokenizer files. |