# Exercise 5.10: Hybrid PPO for Spectrum Sharing

This directory provides the starter code for Exercise 5.10.  
In this exercise, you extend spectrum sharing from a fully discrete RL setup to a **hybrid-action PPO** setup:
- choose a **discrete sub-band**; and
- choose a **continuous transmit power** in **(-100, 23) dBm**.

## Experiment Setup

The scripts are pre-configured with:
- **Number of vehicles `n_veh`:** 4
- **Neighbors per vehicle `n_neighbor`:** 1
- **Resource blocks `n_RB`:** 4
- **Training episodes:** 6000
- **Environment:** vehicular V2X spectrum-sharing simulator

## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Understand baseline** | Read `sarltrain_ppo.py` to see the discrete PPO baseline where sub-band and power are encoded as one discrete action. |
| **Design hybrid policy** | Open `sarltrain_2ppo_starter.py` and complete the `# YOUR CODE HERE` blocks in the continuous-power branch (`Actorp.forward`, `Agentp.choose_action`, and `Agentp.update`). Build a policy that outputs continuous power in dBm while keeping sub-band selection discrete. |
| **Run** | Execute training with the provided scripts (for example, `python sarltrain_2ppo_starter.py`). |
| **Observe** | Check training rewards and saved model files, then compare the hybrid PPO behavior against the discrete baseline. |

> **Hint:** Keep the action semantics consistent end-to-end: if the actor outputs power in dBm, the sampled action, stored transition, PPO probability ratio, and environment call should all use the same unit/range.

## Files

| File | Purpose |
|------|---------|
| `Environment_marl.py` | Training environment for vehicular spectrum sharing. |
| `Environment_marl_test.py` | Testing environment with fixed evaluation setup. |
| `marltrain.py` | MARL-DQN reference implementation from the cited paper. |
| `sarltrain_ppo.py` | Discrete-action PPO baseline. |
| `sarltrain_2ppo_starter.py` | Hybrid PPO starter script (includes TODOs for Exercise 5.10(a)). |
| `marltest_2ppo_decay.py` | Evaluation script for the hybrid PPO setting. |

For a fair comparison between discrete and hybrid methods, use the same testing channel conditions (same random seed / CSI generation protocol).
