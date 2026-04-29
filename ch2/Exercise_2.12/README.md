# Extra Credit: DA-LiteTCN CsiNet

This folder contains the proposed architecture for the extra credit problem.  
The goal is to replace CsiNet-LSTM with a more UE-friendly and Doppler-robust CSI feedback architecture.

## Motivation

CsiNet-LSTM uses temporal correlation between adjacent CSI matrices to improve reconstruction accuracy in time-varying channels. However, LSTM-based temporal modeling can increase computational complexity and latency, especially if temporal processing is placed on the UE side.

To address this issue, I propose **DA-LiteTCN CsiNet**, which stands for:

```text
Doppler-Aware Lightweight Temporal Convolutional CsiNet
```

The main idea is to keep the UE side lightweight and move the heavier temporal modeling to the BS side.

---

## Architecture Overview

| Module | Location | Function |
|---|---|---|
| Lightweight Spatial Encoder | UE | Compress each CSI matrix into a latent vector. |
| Delta Feedback Module | UE | Send latent difference instead of full latent vector when the channel changes slowly. |
| Doppler Indicator | UE | Estimate a simple channel variation indicator from adjacent latent vectors. |
| Latent Recovery | BS | Recover the current latent vector from delta feedback. |
| Doppler-Aware TCN | BS | Extract temporal correlation and adapt to Doppler variation. |
| Residual CSI Decoder | BS | Reconstruct the CSI matrix from temporal features. |

---

## Why This Design?

This architecture is designed for three objectives:

1. **Temporal correlation utilization**  
   The BS-side TCN uses several previous latent CSI vectors to model temporal dependency.

2. **Lower UE-side computation**  
   The UE only performs lightweight CNN encoding and delta feedback. It does not run LSTM.

3. **Robustness against Doppler spread**  
   The Doppler indicator helps the BS adjust temporal memory under different mobility levels.

---

## UE-side Processing

The UE performs:

```text
CSI matrix → Lightweight CNN encoder → latent vector z_t
```

Then it computes:

```text
Δz_t = z_t - z_{t-1}
```

If the channel changes slowly, the UE sends `Δz_t`.  
If the channel changes quickly, the UE can send the full latent vector.

The UE also computes a simple Doppler indicator:

```text
d_t = ||z_t - z_{t-1}||_2
```

This avoids running a heavy temporal model on the UE.

---

## BS-side Processing

The BS reconstructs the latent vector:

```text
z_hat_t = z_hat_{t-1} + Δz_t
```

Then, the BS uses a Doppler-aware TCN:

```text
[z_hat_{t-T+1}, ..., z_hat_t] + d_t → temporal feature
```

Finally, a residual decoder reconstructs the CSI matrix:

```text
temporal feature → Dense → Reshape → Residual Blocks → reconstructed CSI
```

---

## Training Strategy

The model is trained using CSI sequences:

```text
Input:  [X_{t-T+1}, X_{t-T+2}, ..., X_t]
Target: X_t
```

The loss function combines reconstruction loss and temporal consistency loss:

```text
Loss = reconstruction loss + λ × temporal consistency loss
```

To improve Doppler robustness, the training data should include low-Doppler, medium-Doppler, and high-Doppler sequences.

Online adaptation is optional and should be performed only at the BS side to avoid increasing UE complexity.

---

## Ablation Studies

### Ablation 1: Doppler Indicator

Compare the full model with a version without Doppler conditioning.

Purpose:

```text
Verify whether Doppler-aware conditioning improves high-mobility robustness.
```

### Ablation 2: TCN vs LSTM

Compare the proposed TCN-based temporal model with CsiNet-LSTM.

Purpose:

```text
Verify whether TCN can reduce latency while preserving temporal reconstruction performance.
```

### Ablation 3: Delta Feedback

Compare delta latent feedback with full latent feedback.

Purpose:

```text
Verify whether delta feedback reduces feedback overhead while maintaining NMSE.
```

---

## Expected Results

The proposed architecture is expected to:

- reduce UE-side computational overhead;
- maintain temporal correlation modeling;
- improve robustness under Doppler spread;
- reduce average NMSE compared with non-temporal CsiNet;
- achieve similar or better performance than CsiNet-LSTM with lower UE-side complexity.
