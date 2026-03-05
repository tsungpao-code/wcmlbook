# Exercise 5.6: DNN-Based Power Allocation in Gaussian Interference Channels

This directory provides the starter code for Exercise 5.6. Your goal is to train neural networks for wireless power control and compare their performance against the WMMSE baseline.

## Experiment Setup

Both scripts use a Gaussian interference-channel setting with:
- **Number of users `K`:** 10
- **Noise power:** 1
- **WMMSE labels/baseline:** generated inside the scripts

Script-specific defaults:
- **`exercise_5.6a_starter.py` (supervised):** `num_H=25000`, `num_test=5000`, `training_epochs=200`
- **`exercise_5.6b_starter.py` (unsupervised/QoS):** `num_H=10000`, `num_test=500`, `training_epochs=1000`

## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Code A (Supervised)** | Open `exercise_5.6a_starter.py` and complete the `# YOUR CODE HERE` block in `PowerControl.train`. Implement the standard training step: zero gradients, forward pass, compute supervised loss (prediction vs. WMMSE labels), `backward()`, and optimizer step. |
| **Code B (Unsupervised/QoS)** | Open `exercise_5.6b_starter.py` and complete the `# YOUR CODE HERE` block in `PowerControl.train`. Implement unsupervised training that maximizes rate (with the built-in QoS-aware objective), including zero gradients, forward pass, loss computation, `backward()`, and optimizer step. |
| **Run** | Execute `python exercise_5.6a_starter.py` and `python exercise_5.6b_starter.py`. |
| **Observe** | Compare DNN and WMMSE performance from printed metrics and generated figures/files (e.g., CDF/histogram outputs and saved training logs). |

> **Hint:** Keep network outputs within valid power range and ensure tensor shapes are consistent between model output, labels, and channel-matrix reshaping.

## Files

| File | Purpose |
|------|---------|
| `exercise_5.6a_starter.py` | Supervised DNN power-control starter script (with TODO). |
| `exercise_5.6b_starter.py` | Unsupervised/QoS-aware DNN power-control starter script (with TODO). |