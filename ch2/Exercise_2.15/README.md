## Q7(b)(c): CsiNet NMSE Evaluation and Mixed-Dataset Training

This section explains how Q7(b) and Q7(c) are completed using the six COST2100 datasets generated in Q7(a).  
The main Python script used in this part is:

```text
q7bc_csinet_cost2100.py
```

This script evaluates the CSI reconstruction performance of CsiNet under different COST2100 channel datasets and compares single-dataset training with mixed-dataset training.

---

## What You Need to Do

| Step | Task | Details |
| :---: | :--- | :--- |
| 1 | **Prepare Q7(a) Datasets** | Make sure the six COST2100 `.mat` files generated in Q7(a) are placed in `cost2100/cost2100-master/cost2100-master/matlab/q7_generated_datasets/`. |
| 2 | **Prepare Python Environment** | Install required packages such as `tensorflow`, `numpy`, `scipy`, `pandas`, and `h5py`. The `h5py` package is needed because the MATLAB datasets are saved in v7.3 format. |
| 3 | **Run Q7(b)(c) Script** | Run `python q7bc_csinet_cost2100.py` in the Q7 project folder. |
| 4 | **Q7(b): Single-Dataset Training** | Train CsiNet using only `D1_indoor_uniform.mat`, then test the trained model on all six datasets. |
| 5 | **Q7(c): Mixed-Dataset Training** | Mix D1–D6 together, retrain CsiNet, and test the mixed-dataset model on all six datasets. |
| 6 | **Check Results** | The final NMSE comparison table is saved as `result/q7bc_nmse_results.csv`. |

---

## File Structure

| File / Directory | Purpose |
|---|---|
| `q7bc_csinet_cost2100.py` | Main Python script for Q7(b)(c). It trains CsiNet, evaluates NMSE, and compares single-dataset and mixed-dataset training. |
| `q7_generated_datasets/` | Folder containing the six COST2100 datasets generated in Q7(a). |
| `D1_indoor_uniform.mat` | Baseline dataset used for Q7(b) single-dataset training. |
| `D2_indoor_center.mat` | Dataset with users concentrated near the base station. |
| `D3_indoor_edge.mat` | Dataset with users located near the indoor boundary. |
| `D4_indoor_hotspot.mat` | Dataset with users clustered around hotspot regions. |
| `D5_indoor_ring.mat` | Dataset with users distributed in a ring-shaped region. |
| `D6_indoor_line.mat` | Dataset with users distributed along a line. |
| `result/` | Output folder for training loss files and NMSE comparison results. |
| `saved_model/` | Output folder for saved CsiNet models. |
| `result/q7bc_nmse_results.csv` | Final comparison table of Q7(b) and Q7(c) NMSE results. |

---

## How to Run

Run the following commands in PowerShell:

```powershell
cd C:\mach_ai_mid\Q7
python q7bc_csinet_cost2100.py
```

If `h5py` is not installed, install it first:

```powershell
python -m pip install h5py
```

If other packages are missing, install them with:

```powershell
python -m pip install tensorflow numpy scipy pandas
```

---

## Detailed Task Breakdown

### Part 1: Data Loading

The script loads the six datasets generated in Q7(a):

```text
D1_indoor_uniform.mat
D2_indoor_center.mat
D3_indoor_edge.mat
D4_indoor_hotspot.mat
D5_indoor_ring.mat
D6_indoor_line.mat
```

Each `.mat` file contains COST2100-generated CSI data. Since the MATLAB files were saved in v7.3 format, the script uses `h5py` to read them when `scipy.io.loadmat()` cannot load the file.

The script mainly uses `H_norm` as the CSI input. If `H_norm` is not available, it uses `H_complex` and normalizes it manually. Then, the complex CSI is converted into two channels:

- real part
- imaginary part

The final input shape for CsiNet is:

```text
[num_samples, 32, 32, 2]
```

This means each CSI sample is represented as a 32 × 32 image with two channels.

---

### Part 2: CsiNet Model

The model used in this script is a simplified CsiNet-style autoencoder.

The encoder compresses the CSI input into a lower-dimensional feature vector:

```text
Conv2D → BatchNormalization → LeakyReLU → Flatten → Dense
```

The decoder reconstructs the CSI from the compressed representation:

```text
Dense → Reshape → Residual Blocks → Conv2D
```

The model is trained using MSE loss, and its reconstruction quality is evaluated using NMSE.

---

### Part 3: Q7(b) Single-Dataset Training

In Q7(b), the model is trained only on:

```text
D1_indoor_uniform.mat
```

After training, the model parameters are fixed and tested on all six datasets:

```text
D1_indoor_uniform.mat
D2_indoor_center.mat
D3_indoor_edge.mat
D4_indoor_hotspot.mat
D5_indoor_ring.mat
D6_indoor_line.mat
```

The purpose of Q7(b) is to test whether a CsiNet model trained on one channel distribution can generalize to other unseen user distributions.

If the NMSE becomes worse on D2–D6, it means the model is sensitive to channel distribution mismatch.

---

### Part 4: Q7(c) Mixed-Dataset Training

In Q7(c), the script mixes all six datasets together:

```text
D1 + D2 + D3 + D4 + D5 + D6
```

This mixed dataset is used to train a new CsiNet model. After training, the model is again tested on all six datasets.

The purpose of Q7(c) is to check whether training with multiple channel distributions can improve CsiNet generalization.

Compared with Q7(b), Q7(c) allows the model to see more channel variations during training. Therefore, the mixed-dataset model is expected to produce lower NMSE on average and reduce performance degradation under unseen user distributions.

---

## Evaluation Metric: NMSE

The reconstruction performance is evaluated using NMSE:

```text
NMSE = ||H - H_hat||² / ||H||²
```

where:

- `H` is the original CSI.
- `H_hat` is the reconstructed CSI.

The result is reported in dB:

```text
NMSE_dB = 10 log10(NMSE)
```

A lower NMSE value means better reconstruction performance.  
For NMSE in dB, a more negative value means smaller reconstruction error.

For example:

```text
-5 dB is better than 0 dB.
0 dB is better than 10 dB.
```

---

## Experimental Results

The final results are:

| Testing Dataset | Q7(b) Single-Dataset Training NMSE (dB) | Q7(c) Mixed-Dataset Training NMSE (dB) | Improvement (B − C, dB) |
|---|---:|---:|---:|
| `D1_indoor_uniform` | 1.1089 | -2.7851 | 3.8940 |
| `D2_indoor_center` | 21.1534 | 14.2977 | 6.8558 |
| `D3_indoor_edge` | -0.0029 | -5.3976 | 5.3948 |
| `D4_indoor_hotspot` | 0.0622 | -4.1280 | 4.1902 |
| `D5_indoor_ring` | 0.1405 | -4.3296 | 4.4701 |
| `D6_indoor_line` | 10.5952 | 4.4983 | 6.0968 |

The results are saved in:

```text
result/q7bc_nmse_results.csv
```

---

## Why These Results Are Produced

The Q7(b) model is trained only on `D1_indoor_uniform.mat`. Therefore, it mainly learns the CSI characteristics of uniformly distributed indoor users. When the testing dataset changes to center, edge, hotspot, ring, or line distribution, the channel statistics are different from the training data. This causes domain mismatch and increases the reconstruction NMSE.

For example, `D2_indoor_center` has a high NMSE in Q7(b), which means the model trained only on uniform users does not reconstruct this distribution well. Similarly, `D6_indoor_line` also has a high NMSE because the line distribution is different from the uniform training distribution.

In Q7(c), the model is trained using a mixture of all six datasets. Since the model sees multiple user distributions during training, it learns more general CSI features. As a result, Q7(c) improves NMSE for all six testing datasets.

The improvement column shows that mixed-dataset training improves the reconstruction performance on every dataset. This supports the idea that using diverse training data can improve the generalization ability of CSI feedback methods.

---

## Difference Between Q7(b) and Q7(c)

| Item | Q7(b): Single-Dataset Training | Q7(c): Mixed-Dataset Training |
|---|---|---|
| Training data | Only `D1_indoor_uniform.mat` | Mixed D1–D6 datasets |
| Testing data | D1–D6 | D1–D6 |
| Main purpose | Test cross-dataset generalization | Improve generalization using diverse training data |
| Expected behavior | Good only for similar distributions; worse for mismatched distributions | More stable performance across different distributions |
| Result in this experiment | NMSE is high on some shifted datasets | NMSE improves on all datasets |

In short, Q7(b) tests the weakness of a single-distribution model, while Q7(c) tests whether mixed training can reduce this weakness.

---




## Discussion

The results show that mixed-dataset training improves CsiNet generalization. In Q7(b), the model trained only on uniform users performs poorly when tested on some different distributions, especially `D2_indoor_center` and `D6_indoor_line`. This indicates that the single-dataset model overfits to the training distribution.

In Q7(c), the model trained with all six user distributions achieves better NMSE on every dataset. This means that mixed-dataset training helps the model learn more robust CSI features.

However, `D2_indoor_center` and `D6_indoor_line` still have relatively high NMSE after mixed training. This suggests that these distributions are still more difficult for the model to reconstruct. Possible reasons include stronger distribution shift, different channel power range, or limited training samples.

To further improve performance, the following methods can be considered:

- Use more training samples.
- Train for more epochs.
- Add more user distributions.
- Include outdoor COST2100 scenarios.
- Apply SNR-based data augmentation.
- Use transfer learning or fine-tuning for difficult distributions.
- Tune CsiNet hyperparameters such as `encoded_dim`, number of residual blocks, and learning rate.

---

## Conclusion

Q7(b)(c) demonstrates that CsiNet reconstruction performance depends strongly on the training data distribution. A model trained on only one dataset may not generalize well to other channel distributions. By contrast, mixed-dataset training improves the NMSE on all six datasets and reduces domain mismatch.

Therefore, for practical CSI feedback systems, it is better to train CsiNet with diverse channel datasets instead of relying on only one fixed channel distribution.
