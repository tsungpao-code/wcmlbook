# Exercise 5.14: Meta-RL for Spectrum Sharing (Reptile + PPO)

This directory provides the code for Exercise 5.14.  
The objective is to learn meta-initialized PPO parameters across multiple V2X tasks, then adapt quickly to a specific target task.

## Requirements

- **Python:** > 3.6
- **PyTorch:** > 1.4.0

## Experiment Setup

The implementation follows a two-stage pipeline:
- **Meta-training stage:** `marltrain_ppo_meta.py` samples different tasks and updates shared initialization with a Reptile-style meta update.
- **Adaptation stage:** `sarltrain_ppo_adapt.py` loads meta parameters and fine-tunes PPO on a specific environment.

Default settings in the scripts include:
- **Vehicles `n_veh`:** 4
- **Meta outer loops:** 200
- **Meta inner loops:** 20
- **Adaptation episodes:** 200 (increase if training from scratch)

## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Meta-train** | Run `python marltrain_ppo_meta.py` to obtain meta actor/critic checkpoints (Reptile-based). |
| **Adapt** | Run `python sarltrain_ppo_adapt.py` to load meta checkpoints and adapt to a single target task. |
| **Compare** | Compare adaptation reward with random baseline and (optionally) with training from scratch by disabling meta checkpoint loading and increasing episodes. |

> **Hint:** For a fair comparison, keep task-generation and random seeds consistent when comparing meta-initialized adaptation vs. scratch training.

## Files

| File | Purpose |
|------|---------|
| `Environment_meta.py` | Multi-task environment generator (task factors are configurable). |
| `Environment_marl.py` | Single-task environment used for adaptation/fine-tuning. |
| `marltrain_ppo_meta.py` | Meta-training script (PPO inner updates + Reptile meta update). |
| `sarltrain_ppo_adapt.py` | Task adaptation script using learned meta initialization. |
