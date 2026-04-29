# Extra Credit: DA-LiteTCN CsiNet

This folder contains the proposed architecture for the extra credit problem.  
The goal is to replace CsiNet-LSTM with a more UE-friendly and Doppler-robust CSI feedback architecture.

The proposed model is called:

```text
DA-LiteTCN CsiNet
```

which stands for:

```text
Doppler-Aware Lightweight Temporal Convolutional CsiNet
```

---

## Motivation

CsiNet-LSTM uses temporal correlation between adjacent CSI matrices to improve reconstruction accuracy in time-varying channels. This is useful because wireless channels are not independent over time. In practical mobile systems, adjacent CSI frames often contain similar spatial and frequency-domain structures.

However, LSTM-based temporal modeling can increase computational complexity and latency, especially if temporal processing is placed on the UE side. Since UE devices usually have limited power, memory, and computation resources, a practical CSI feedback architecture should avoid heavy recurrent operations at the UE.

Another challenge is Doppler spread. When the UE moves faster, the channel changes more rapidly over time. In this case, simply relying on long-term temporal memory may not always help. The model should be able to adapt its temporal reconstruction strategy according to the channel variation level.

To address these issues, I propose **DA-LiteTCN CsiNet**. The main idea is to keep the UE side lightweight and move the heavier temporal modeling to the BS side.

---

## Main Idea

The proposed architecture separates the CSI feedback task into two sides:

| Side | Main Responsibility |
|---|---|
| UE side | Lightweight CSI compression and delta latent feedback |
| BS side | Temporal modeling, Doppler-aware reconstruction, and CSI decoding |

Instead of placing an LSTM at the UE side, the UE only performs lightweight CNN-based encoding. The BS then uses a temporal convolutional network (TCN) to exploit temporal correlation across several CSI frames.

This design is more practical because the BS usually has more computational resources than the UE.

---

## Architecture Overview

| Module | Location | Function |
|---|---|---|
| Lightweight Spatial Encoder | UE | Compress each CSI matrix into a latent vector. |
| Delta Feedback Module | UE | Send latent difference instead of full latent vector when the channel changes slowly. |
| Doppler Indicator | UE / Latent domain | Estimate a simple channel variation indicator from adjacent latent vectors. |
| Latent Recovery | BS | Recover the current latent vector from delta feedback. |
| Doppler-Aware TCN | BS | Extract temporal correlation and adapt to Doppler variation. |
| Residual CSI Decoder | BS | Reconstruct the CSI matrix from temporal features. |

---

## Why This Design?

This architecture is designed for three objectives.

### 1. Temporal Correlation Utilization

Adjacent CSI matrices are usually correlated in time. Instead of reconstructing each CSI frame independently, the BS-side TCN uses several previous latent CSI vectors to model temporal dependency.

The input sequence is:

```text
[X_{t-T+1}, X_{t-T+2}, ..., X_t]
```

The model reconstructs the current CSI:

```text
X_t
```

In the current prototype, the sequence length is set to:

```text
SEQ_LEN = 4
```

This means the model uses four consecutive CSI frames as input.

---

### 2. Lower UE-side Computation

The UE does not run LSTM or any heavy temporal model. It only performs:

```text
CSI matrix → Lightweight CNN encoder → latent vector
```

The encoder uses depthwise separable convolution, which is lighter than a standard CNN encoder. This reduces UE-side computational overhead.

In the prototype code, this module is implemented as:

```text
UE_Lightweight_Encoder
```

The output latent vector dimension is:

```text
ENCODED_DIM = 128
```

Therefore, each CSI frame is compressed into a 128-dimensional latent vector.

---

### 3. Robustness Against Doppler Spread

When Doppler spread is high, the CSI changes quickly. In this case, older CSI frames may become less reliable. To address this, the proposed model computes a simple Doppler indicator from adjacent latent vectors:

```text
d_t = ||z_t - z_{t-1}||_2
```

If the latent difference is large, it indicates stronger channel variation. The BS can use this information to adjust the temporal reconstruction process.

In the prototype, the Doppler indicator is implemented as a `Lambda` layer:

```text
doppler_indicator_from_latent_difference
```

---

## UE-side Processing

The UE performs lightweight CSI encoding:

```text
CSI matrix → Depthwise Conv2D → Pointwise Conv2D → Flatten → Dense → latent vector z_t
```

Then it computes the latent difference:

```text
Δz_t = z_t - z_{t-1}
```

If the channel changes slowly, the UE can send `Δz_t` instead of the full latent vector. This can reduce feedback overhead.

If the channel changes quickly, the UE can send the full latent vector to avoid error accumulation.

The UE also computes a simple Doppler indicator:

```text
d_t = ||z_t - z_{t-1}||_2
```

This design avoids running a heavy temporal model on the UE.

---

## BS-side Processing

The BS reconstructs the latent vector:

```text
z_hat_t = z_hat_{t-1} + Δz_t
```

Then the BS uses a Doppler-aware TCN:

```text
[z_hat_{t-T+1}, ..., z_hat_t] + d_t → temporal feature
```

Finally, a residual decoder reconstructs the CSI matrix:

```text
temporal feature → Dense → Reshape → Residual Blocks → reconstructed CSI
```

The BS-side module is heavier than the UE-side module, but this is acceptable because the BS has stronger computation capability.

---

## Code File

The prototype implementation is provided in:

```text
extra_da_litetcn_csinet.py
```

This file builds the proposed DA-LiteTCN CsiNet model and verifies the input/output tensor shapes using dummy data.

---

## File Structure

| File / Directory | Purpose |
|---|---|
| `extra_da_litetcn_csinet.py` | Prototype implementation of the proposed DA-LiteTCN CsiNet architecture. |
| `README.md` | Explanation of the proposed architecture, motivation, execution result, and expected validation method. |

---

## Program Structure

The script includes the following main parts:

| Function / Module | Purpose |
|---|---|
| `build_lightweight_ue_encoder()` | Builds the UE-side lightweight encoder. |
| `build_bs_temporal_decoder()` | Builds the BS-side Doppler-aware TCN decoder. |
| `residual_decoder_block()` | Builds the residual block used for CSI reconstruction. |
| `build_da_litetcn_csinet()` | Combines the UE encoder and BS decoder into the complete proposed architecture. |
| Dummy test in `__main__` | Creates random CSI sequence input and checks whether the output shape is correct. |

---

## Implementation Details

### UE-side Lightweight Encoder

The UE-side encoder uses:

```text
DepthwiseConv2D → BatchNormalization → LeakyReLU → Pointwise Conv2D → Flatten → Dense
```

This reduces computation compared with a heavier CNN or LSTM-based temporal encoder.

The output shape of the UE encoder is:

```text
(None, 128)
```

When applied to a sequence using `TimeDistributed`, the output becomes:

```text
(None, 4, 128)
```

This means each CSI frame is compressed into a 128-dimensional latent vector, and the model processes 4 CSI frames at a time.

---

### Doppler Indicator

The Doppler indicator is computed from the difference between the last two latent vectors:

```text
d_t = ||z_t - z_{t-1}||_2
```

The output shape is:

```text
(None, 1)
```

This is used as additional information for the BS-side decoder.

---

### BS-side Doppler-Aware TCN Decoder

The BS-side decoder uses dilated causal temporal convolution:

```text
Conv1D dilation=1 → Conv1D dilation=2 → Conv1D dilation=4
```

This allows the model to capture short-term and longer-term temporal dependency without using LSTM.

After temporal feature extraction, the Doppler indicator is concatenated with the temporal feature:

```text
temporal feature + Doppler indicator
```

Then the decoder reconstructs the CSI using:

```text
Dense → Reshape → Residual Blocks → Conv2D
```

---

## How to Run

Open PowerShell and move to the folder:

```powershell
cd C:\mach_ai_mid\extra_point
```

Run:

```powershell
python extra_da_litetcn_csinet.py
```

---

## Prototype Execution Result

The prototype script successfully builds the proposed model and prints the Keras model summary.

The actual execution result is:

```text
Model: "DA_LiteTCN_CsiNet"
Total params: 892,866
Trainable params: 892,750
Non-trainable params: 116
```

The dummy test also produces:

```text
Dummy input shape: (2, 4, 32, 32, 2)
Dummy output shape: (2, 32, 32, 2)
```

This means the model receives a batch of CSI sequences as input. Each sequence contains 4 CSI frames, and each CSI frame has shape:

```text
32 × 32 × 2
```

where the two channels represent the real and imaginary parts of CSI.

The model output is:

```text
32 × 32 × 2
```

which matches the expected reconstructed CSI shape.

---

## Interpretation of the Execution Result

The execution result verifies that the proposed model is implementable.

The model summary confirms the following:

| Module | Output Shape | Interpretation |
|---|---|---|
| `csi_sequence_input` | `(None, 4, 32, 32, 2)` | The model accepts 4 consecutive CSI frames. |
| `time_distributed_ue_encoder` | `(None, 4, 128)` | Each CSI frame is compressed into a 128-dimensional latent vector. |
| `doppler_indicator_from_latent_difference` | `(None, 1)` | A Doppler/channel variation indicator is computed from adjacent latent vectors. |
| `BS_Doppler_Aware_TCN_Decoder` | `(None, 32, 32, 2)` | The BS reconstructs the CSI matrix using temporal features and Doppler-aware information. |

Therefore, the prototype confirms that the architecture can process temporal CSI sequences and reconstruct the current CSI matrix.

---

## Training Strategy

For a complete experiment, the model should be trained using CSI sequences:

```text
Input:  [X_{t-T+1}, X_{t-T+2}, ..., X_t]
Target: X_t
```

The loss function can combine reconstruction loss and temporal consistency loss:

```text
Loss = reconstruction loss + λ × temporal consistency loss
```

The reconstruction loss can be MSE or NMSE:

```text
NMSE = ||H - H_hat||² / ||H||²
```

The temporal consistency loss can be:

```text
||ΔX_hat_t - ΔX_t||²
```

This encourages the reconstructed CSI to preserve realistic temporal variation.

To improve Doppler robustness, the training data should include:

- low-Doppler sequences
- medium-Doppler sequences
- high-Doppler sequences

Online adaptation is optional and should be performed only at the BS side to avoid increasing UE complexity.

---

## Ablation Studies

### Ablation 1: Doppler Indicator

Compare the full model with a version without Doppler conditioning.

Purpose:

```text
Verify whether Doppler-aware conditioning improves high-mobility robustness.
```

Expected result:

```text
The model with Doppler conditioning should achieve lower NMSE under high-Doppler conditions.
```

---

### Ablation 2: TCN vs LSTM

Compare the proposed TCN-based temporal model with CsiNet-LSTM.

Purpose:

```text
Verify whether TCN can reduce latency while preserving temporal reconstruction performance.
```

Expected result:

```text
TCN should provide comparable reconstruction performance with lower inference latency because temporal convolution is more parallelizable than LSTM.
```

---

### Ablation 3: Delta Feedback

Compare delta latent feedback with full latent feedback.

Purpose:

```text
Verify whether delta feedback reduces feedback overhead while maintaining NMSE.
```

Expected result:

```text
Delta feedback should reduce feedback overhead under low-Doppler conditions while maintaining similar reconstruction quality.
```

---

## Expected Evaluation Metrics

A complete evaluation should include the following metrics:

| Metric | Purpose |
|---|---|
| NMSE | Measure CSI reconstruction error. |
| Inference time | Measure reconstruction latency. |
| UE-side parameter count | Estimate UE computational overhead. |
| Feedback dimension / bits | Evaluate feedback overhead. |
| Robustness under Doppler spread | Compare performance under low, medium, and high mobility. |

---

## Expected Results

The proposed architecture is expected to:

- reduce UE-side computational overhead;
- maintain temporal correlation modeling;
- improve robustness under Doppler spread;
- reduce average NMSE compared with non-temporal CsiNet;
- achieve similar or better performance than CsiNet-LSTM with lower UE-side complexity.

---

## Difference from CsiNet-LSTM

| Item | CsiNet-LSTM | Proposed DA-LiteTCN CsiNet |
|---|---|---|
| Temporal module | LSTM | Temporal Convolution Network |
| Temporal processing | Sequential recurrent modeling | Parallelizable convolution-based modeling |
| UE-side burden | Higher if temporal modeling is placed near UE | Lower because UE only uses lightweight encoder |
| Doppler awareness | Not explicitly conditioned | Uses latent-difference Doppler indicator |
| Feedback design | Usually full compressed CSI | Supports delta latent feedback |
| Expected advantage | Good temporal modeling | Lower UE overhead and better Doppler robustness |

---

## Conclusion

DA-LiteTCN CsiNet is proposed as a UE-friendly and Doppler-robust replacement for CsiNet-LSTM.

The key innovation is to move heavy temporal modeling from the UE side to the BS side. The UE only performs lightweight CNN-based CSI compression and optional delta latent feedback, while the BS performs Doppler-aware temporal reconstruction using TCN.

The prototype code successfully builds the model and verifies that the input/output dimensions are correct:

```text
Input:  (2, 4, 32, 32, 2)
Output: (2, 32, 32, 2)
```

This confirms that the proposed architecture can process temporal CSI sequences and reconstruct CSI matrices. Future work can train this model on real CSI sequence datasets and compare it with CsiNet-LSTM using NMSE, latency, feedback overhead, and Doppler robustness.
