# Exercise 2.7: Data-Driven SISO-OFDM Channel Estimation

This repository provides the reference code for Exercise 2.7. Your task is to implement and evaluate channel estimators using both **DNN-based** and **LMMSE** methods for a SISO-OFDM system, and reproduce the simulation results demonstrating MSE performance (Figure 2.9).


## Experiment Setup
The scripts are configured to simulate the OFDM system with the following parameters:
* **Subcarriers $K$ :** 64
* **Pilot Symbol:** 1st OFDM symbol (64 QPSK-modulated pilot symbols)
* **Data Symbol:** 2nd OFDM symbol (64-QAM modulation)
* **SNR Range:** 5 dB to 40 dB (in 5 dB increments)
* **Channel Estimators:** DNN-based (multi-layer perceptron) and Linear Minimum Mean Square Error (LMMSE)
* **Scenarios:** With Cyclic Prefix (CP) for ideal conditions, and without CP to demonstrate inter-symbol interference.


## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Code** | Open `tools/networks.py`, look for the `# YOUR CODE HERE` blocks inside **`build_ce_dnn`**, and fill in the blanks.  |
| **Train DNN** | Open `main.py` and set `ce_type = 'dnn'` and `test_ce = False`. Execute the script to train the DNN-based channel estimator across the SNR range. |
| **Evaluate DNN** | Change the settings in `main.py` to `ce_type = 'dnn'` and `test_ce = True`. Run the script to calculate the MSE performance. |
| **Evaluate LMMSE** | Change the settings in `main.py` to `ce_type = 'mmse'` and `test_ce = True`. Run the script to calculate the baseline LMMSE MSE performance. |
| **Remove CP** | To reproduce the dashed-line results (CP-free), set `CP_flag = False` in `main.py` and repeat the evaluation steps above. |
| **Run** | Execute: `python main.py` for each configuration phase. |



## Files

| File | Purpose |
|------|---------|
| `main.py` | The main executable script to configure parameters, run training loops, and evaluate MSE performance across different SNRs. |
| `tools/networks.py` | Contains the `build_ce_dnn` function, which defines and trains the DNN-based channel estimator using TensorFlow. |
| `tools/raputil.py` | Contains utility functions including `MMSE_CE` for the LMMSE channel estimator calculation. |
| `tools/other_files` | Other necessary files for running the project. |


