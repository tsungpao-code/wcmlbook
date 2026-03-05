# Exercise 5.9: Graph Connectivity in GNN-based Power Control

This directory provides the starter code for investigating how graph topology affects the performance of Graph Convolutional Networks (GCN) in wireless power control.  
In this exercise, you will explore the trade-off between **graph sparsity** and **model performance**:
- compare a **fully-connected graph (D=0)** against a **pruned graph (D=50)**; and
- analyze convergence rates across **varying distance thresholds** D in {0, 10, 20, 30, 40, 50, 60} meters.

## Experiment Setup

The scripts are pre-configured with:
- **Number of links `train_K`:** 50
- **Field size:** 1000m x 1000m
- **Bandwidth:** 5 MHz
- **Transmit Power:** 40 dBm
- **Training epochs:** 50

## What You Need to Do

| Checklist | Details                                                                                                                                                                                 |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Understand baseline** | Read the main script to see how the fully connected baseline (`D=0`) and the sparse graph (`D=50`) are constructed and compared.                                                        |
| **Design dynamic graphs** | Implement the multi-threshold loop in your script. Use the `proc_data` function to dynamically rebuild the graph topology by pruning interference edges where the distance exceeds `D`. |
| **Run** | Execute:`exercise_5.9_a.py`, `exercise_5.9_b.py`                                                                                                                                        |
| **Observe** | Check the recorded sum-rates and the output plot , then compare how the convergence rate and final throughput vary with different threshold values.                                     |

> **Hint:** Keep the physical dataset consistent end-to-end: while the graph topology edges changes with $D$, the underlying distance matrices, path losses, and channel fading conditions must remain identical for a fair comparison.

## Files

| File                             | Purpose |
|----------------------------------|---------|
| `exercise_5.9_a.py` | Main script for comparing GNN performance under limited CSI vs. complete CSI. |
| `exercise_5.9_b.py` | Main script for training the  network with varying threshold $D$ and plotting convergence. |
| `wireless_networks_generator.py` | Utility to generate device layouts and compute path losses. |
| `helper_functions.py`            | Functions for adding fast fading, shadowing, and calculating SR loss. |
| `FPLinQ.py`                      | Baseline fractional programming implementation for performance comparison. |

For a fair comparison between different graph connectivities, use the same testing channel conditions before applying the distance thresholds.