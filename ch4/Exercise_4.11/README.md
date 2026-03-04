# Exercise 4.11: E2E Communication System over Rayleigh Channel

This repository provides the starter code for Exercise 4.11. Your task is to implement the **Rayleigh fading channel simulation**, **training**, and **evaluation** steps for an end-to-end communication system, and analyze its Block Error Rate (BLER) performance under different parameters.

## Experiment Setup
The script is pre-configured with the following key parameters:
*   **Number of Messages $M$:** 256
*   **Channel Uses $n$:** 32, 64, 128 (Variable for Part a)
*   **Training SNR:** 1 dB, 7 dB, 20 dB (Variable for Part b)
*   **Test SNR Range:** [0, 20] dB in steps of 3 dB

## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Code** | Open `E2EComm_Rayleigh.py` and look for all `# TODO:` comment blocks. Replace the placeholders with code that implements:<br>  • In `CommunicationSystem.forward`: Rayleigh channel simulation, AWGN noise addition, and Zero-Forcing equalization.<br>  • In the `train_model` function: The training step (forward pass, loss calculation, backward pass, and optimizer update).<br>  • In the `evaluate` function: The evaluation step (forward pass without gradients, get predictions, and calculate the BLER). |
| **Run (Part a)** | After completing the code, run the script to train and evaluate model performance with **fixed training SNR = 7 dB and different n (32, 64, 128)**. |
| **Run (Part b)** | Run the script to train and evaluate model performance with **fixed n = 128 and different training SNR (1, 7, 20 dB)**. |
| **Observe** | The script should generate and save the result plots:<br>  • `bler_vs_snr_n_comparison.png` (Part a result): Shows BLER vs. Test SNR for different $n$.<br>  • `bler_vs_snr_trainSNR_comparison.png` (Part b result): Shows BLER vs. Test SNR for different training SNRs. |

> **Hint:** In the channel simulation, carefully generate the complex Rayleigh fading coefficients and scale the AWGN noise power correctly based on the given SNR and the normalized signal power.

## Files

| File | Purpose |
|------|---------|
| `E2EComm_Rayleigh.py` | Main starter script (with all TODOs). Defines the `CommunicationSystem` class, training, and evaluation functions. |
| `bler_vs_snr_n_comparison.png` | Plot generated after running Part (a). |
| `bler_vs_snr_trainSNR_comparison.png` | Plot generated after running Part (b). |