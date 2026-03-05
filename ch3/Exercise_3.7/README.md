# Exercise 3.7: Gradient-based MCMC for MIMO Detection

This repository provides the code for Exercise 3.7. Your task is to utilize the gradient-based MCMC algorithm. You will evaluate its Bit Error Rate (BER) performance in MIMO detection and compare it with other classical and modern methods to reproduce the results shown in **Figure 3.5**.

## Experiment Setup
The script is pre-configured with the specific MIMO system parameters:
* **Antenna Configuration:** $8 \times 8$ MIMO ($N_t = 8, M_r = 8$)
* **Modulation Scheme:** 16-QAM ($\mu = 4$)
* **Channel Model:** Rayleigh fading channel (with perfect CSI, `csi = 0`)
* **MCMC Hyperparameters:** $16$ samplers (`samplers = 16`), $8$ iterations (`samples = 8`)
* **SNR Range:** 0 dB to 25 dB


## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Code** | Open `main.py` and look for the `SysIn` class configuration. Modify the `detect_type` variable to sequentially run different algorithms (e.g., `'MHGD'`, `'EP'`, `'MMSE'`, `'ML'`). Then, write a separate script to load the generated `.mat` files and plot the BER vs. SNR curves. |
| **Run** | Execute: `python main.py` (Run this multiple times, changing `detect_type` each time) |
| **Observe** | Real-time BER/SER statistics will print in the console. Resulting `.mat` files (e.g., `Results_MHGD_8x8_16QAM.mat`) are saved for each algorithm to construct Figure 3.5. |


> **Hint:**  To compare the BER performance, you can use MATLAB or Python's `scipy.io.loadmat()` to read the saved `.mat` files, and use a semi-logarithmic y-axis (`plt.semilogy()` in matplotlib) to plot the BER curves.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Main script to configure system parameters and loop through the SNR list. |
| `MIMO_detection.py` | (Inside `tools/`) Core simulation engine handling modulation, channel generation, noise, and detector selection. |
| `EP.py`, `MHGD.py` | (Inside `tools/`) Contains the core mathematical implementations of the detection algorithms. |
| `Results_*.mat` | Data files generated after you run the script, containing BER/SER arrays. |