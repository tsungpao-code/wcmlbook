# Exercise 4.10: End-to-End Communication over Multipath Fading Channels

This repository provides the starter code for Exercise 4.10. Your task is to use Python and PyTorch to model a learned end-to-end communication system operating over a time-varying multipath fading channel, specifically using the 3GPP Tapped Delay Line (TDL) model and Orthogonal Frequency-Division Multiplexing (OFDM).

## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Code (Task 1)** | Open `TDL_torch_starter.py`. You need to:<br> &nbsp;• Model a time-varying multipath fading channel based on the 3GPP TDL model.<br> &nbsp;• Ensure the generated channel model has **no trainable parameters**. |
| **Code (Task 2)** | Open `OFDM_TDL_torch_starter.py`. You need to:<br> &nbsp;• Extend your TDL channel model to an OFDM scheme according to **Table 4.7**.<br> &nbsp;• Integrate the channel with the provided resource grid, AWGN, and MMSE channel estimation/equalization modules. |
| **Run** | Execute your scripts to verify your implementations:<br> `python TDL_torch_starter.py`<br> `python OFDM_TDL_torch_starter.py`<br>*(Note: Ensure you have installed required libraries such as `torch`, `numpy`, and `importlib_resources` first).* |
| **Observe** | Observe how the time-varying multipath fading channel (TDL) alters the signal, and evaluate how the OFDM architecture coupled with MMSE equalization mitigates these fading effects. |

## Files

| File | Purpose |
|------|---------|
| `TDL_torch_starter.py` | Starter script for Task 1. Implement the 3GPP TDL channel model here. |
| `OFDM_TDL_torch_starter.py` | Starter script for Task 2. Implement the OFDM scheme and integrate your TDL model here. |
| `resource_grid.py` | Utility script defining the OFDM resource grid and mapping. |
| `Channel_estimation.py` | Utility script providing channel estimation and MMSE equalization modules. |
| `channel_utils_torch.py` | Helper functions for channel operations (e.g., CIR to OFDM channel conversion, complex noise generation). |
| `pilot_pattern.py` | Definitions for pilot symbol placement within the resource grid. |
| `models/` | Directory containing JSON configuration files for various TDL models (e.g., `TDL-A.json`). |