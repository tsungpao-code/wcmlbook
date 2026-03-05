# Exercise 5.5: WMMSE Power Allocation for D2D Links

This directory contains the starter code for Exercise 5.5. Your task is to complete the core update steps in the WMMSE (Weighted MMSE) iteration and solve the transmit power allocation for 4 D2D links.

## Experiment Setup

The script is pre-configured with the textbook example setup:
- **Number of links `K`:** 4
- **Maximum power `P_max`:** 1
- **Initial power `P_ini`:** all-ones vector
- **Channel gain matrix `H`:** provided in the script
- **Noise power `var_noise`:** 1
- **Iteration count:** 100

## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Code** | Open `exercise_5.5_starter.py` and locate `# YOUR CODE HERE` inside `WMMSE_sum_rate`. Complete the per-iteration updates for `f`, `w`, and `b` (and objective tracking with `VV`, if needed), then project the power to satisfy `0 <= p_k <= P_max`. |
| **Run** | Execute: `python exercise_5.5_starter.py` |
| **Observe** | The script prints the final power allocation `Power_allocation` for the 4 D2D links. |

> **Hint:** Update the receiver filter `f` and weight `w` from the current `b` first, then update the transmit amplitude `b`. Finally compute power with `p_opt = b^2` and enforce the maximum-power constraint.

## Files

| File | Purpose |
|------|---------|
| `exercise_5.5_starter.py` | Starter script for the exercise (includes TODO). |

