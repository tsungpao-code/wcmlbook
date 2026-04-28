# Micro-Doppler Shift — Rotary-Wing UAV Sub-6 GHz

Implementation of the channel model, estimation algorithm, and interactive
animated system model from:

> **Hou, H.-A., Wang, L.-C., & Lin, H.-P.** (2021).
> *Micro-Doppler Shift and Its Estimation in Rotary-Wing UAV Sub-6 GHz Communications.*
> IEEE Wireless Communications Letters, 10(10), 2185–2189.
> https://doi.org/10.1109/LWC.2021.3095898
---

## Repository Structure

```
Micro-Doppler-Shift_Rotary-Wing-UAV_Sub6G/
├── system-model.html                  # Interactive animated system model
├── microdoppler_channel.py            # Core physics simulation + analytical plots
├── microdoppler_estimation.py         # L-FMCW estimation pipeline + RMSE bound
├── q2_serviceability_resilience.py    # Q2: Serviceability violation probability and resilience index
├── README.md                          # Project documentation
└── [generated outputs]
    ├── microdoppler_results.png
    ├── microdoppler_estimation_results.png
    └── q2_serviceability_results.txt
```
---

## Quick Start

### 1. Interactive Animation (no dependencies)

```bash
# Simply open in any modern browser
xdg-open system-model.html        # Linux
open system-model.html            # macOS
```

The animation shows:
- Quadcopter UAV with rotating propellers (top-down view)
- Direct path (blue dashed) and reflection path (orange, active only in zone)
- Propeller detail inset with reflection zone (±θ_rz sectors)
- Real-time f_D(t) waveform with cursor
- Live parameter readouts
- Interactive sliders for f_m, f_c, D_p, d_ant, N_b, animation speed

### 2. Python — Channel Simulation

```bash
pip install numpy scipy matplotlib
python microdoppler_channel.py
# → microdoppler_results.png  (~5 s)
```

Produces a 4-panel dark-theme figure:
- Panel 1: Blade angle θ_e(t) with reflection zone markers  [eq. 1]
- Panel 2: Micro-Doppler f_D,r(t)                           [eq. 5]
- Panel 3: |f_D|_max vs carrier frequency f_c
- Panel 4: |f_D|_max vs motor RPM
```markdown
### 3. Python — Estimation Algorithm

```bash
python microdoppler_estimation.py
# → microdoppler_estimation_results.png  (~15 s)
```
### 4. Python — Q2 Serviceability and Resilience Metrics

```bash
python q2_serviceability_resilience.py
# → q2_serviceability_results.txt
```
This script computes:

- RMS micro-Doppler estimation error based on Eq. (20)
- Serviceability violation probability P_fail
- Single-UAV serviceable probability
- k-out-of-n system resilience index R_sys

Default Q2 parameters:

| Parameter | Value |
|---|---:|
| SNR | 10 dB |
| Sampling rate f_s | 60 kSPS |
| Return loss RL | 5 dB |
| Threshold T | 50 Hz |
| Number of UAVs n | 10 |
| Required serviceable UAVs k | 7 |

Expected output:

```text
RMS estimation error sigma = 10739.9282 Hz
P_fail = P(|epsilon| > T) = 0.996285
P_success = 0.003715
R_sys = 1.159551e-15
```

---

## Key Parameters (Table I & II, Hou et al. 2021)

| Symbol | Parameter                  | Default    | Unit |
|--------|----------------------------|------------|------|
| D_p    | Propeller diameter         | 254        | mm   |
| f_m    | Motor revolutions          | 4620       | RPM  |
| N_b    | Blades per propeller       | 2          | —    |
| f_c    | Carrier frequency          | 2.5 / 38   | GHz  |
| d_UE   | UE-to-antenna distance     | 1000       | m    |
| d_ant  | Hub-to-antenna distance    | 200        | mm   |
| RL     | Return loss of propellers  | 5, 15      | dB   |
| f_s    | Sampling frequency         | 30, 60, 480| kSPS |
| SNR    | Signal-to-noise ratio      | 40         | dB   |
| N_avg  | Averaging half-window      | 4          | —    |

---

## Key Equations

**Blade angle**  [eq. 1]:
```
θ_e(t) = 2π f_m t + θ_p(0) + 2mπ/N_b − π/2
```

**Reflection zone** [eq. 4]:
```
θ_rz = arcsin(D_p / (4 d_ant))
     = arcsin(0.254 / (4 × 0.200))  ≈  0.32 rad  ≈  18.4°
```

**Instantaneous micro-Doppler** [eq. 5]:
```
f_D,r(t) = −4π f_m d_ant f_c sin(2θ_e(t)) / c
```

**Maximum |f_D|** (at θ_e = ±π/4):
```
|f_D|_max = 4π f_m d_ant f_c / c
           = 4π × 77 × 0.2 × 2.5×10⁹ / 3×10⁸  ≈  963 Hz
```

**Blade cycle period**:
```
T_blade = 1 / (f_m × N_b)  =  1 / (77 × 2)  ≈  6.494 ms
```

**Reflection duration per window**:
```
Δt_refl = 2 θ_rz T_blade / π  ≈  1.323 ms
```

**Doppler estimation** [eq. 15]:
```
f̂_D,r[n] = f_s · Arg(â_r[n] / â_r[n−1]) / (2π)
```

**Smoothing** [eq. 16]:
```
f̄_D,r[n] = (1/(2N_avg+1)) Σ_{m=−N_avg}^{N_avg} f̂_D,r[n+m]
```

**RMS error bound** [eq. 20]:
```
10 log E(|ε_{f_D}|²) < 20 log(f_s / π) + RL − SNR
```
At (f_s=60kSPS, SNR=40dB, RL=5dB): bound → **18.83 Hz** (1.95% of |f_D|_max).

```markdown
---

## Q2: Serviceability and Resilience Metrics

For Q2, the UAV-aided network is considered serviceable if the micro-Doppler estimation error stays below the serviceability threshold:

```text
T = 50 Hz
```

This threshold represents the requirement for maintaining NOMA power-domain multiplexing stability in a 6G UAV-aided URLLC network.

---

### RMS Error from Eq. (20)

From Eq. (20), the RMS error bound of the micro-Doppler frequency estimation is:

```text
10 log10 E(|epsilon_fD|^2) < 20 log10(f_s / pi) + RL − SNR
```

Converting it to the linear RMS error gives:

```text
sigma = sqrt(E(|epsilon_fD|^2))
      = (f_s / pi) × 10^((RL − SNR) / 20)
```

For the Q2 setting:

```text
f_s = 60000 Hz
SNR = 10 dB
RL = 5 dB
```

the RMS estimation error is:

```text
sigma ≈ 10739.93 Hz
```

---

### Serviceability Violation Probability

The estimation error is modeled as a zero-mean Gaussian random variable:

```text
epsilon_fD ~ N(0, sigma^2)
```

Although the problem statement writes epsilon_fD > T, both positive and negative estimation errors can violate the serviceability requirement. Therefore, I use the two-sided violation probability:

```text
P_fail = P(|epsilon_fD| > T)
       = 2Q(T / sigma)
```

With:

```text
T = 50 Hz
sigma = 10739.93 Hz
```

we obtain:

```text
P_fail ≈ 0.996285
P_success = 1 − P_fail ≈ 0.003715
```

This means that under the low-SNR condition, a single UAV link has a very high probability of failing the serviceability requirement.

---

### System Resilience Index

For a UAV swarm with:

```text
n = 10
k = 7
```

the network is considered resilient if at least 7 UAVs maintain serviceable links.

Let:

```text
X ~ Binomial(n = 10, P_success)
```

Then the system resilience index is:

```text
R_sys = P(X >= 7)
      = sum_{i=7}^{10} C(10,i) P_success^i (1 − P_success)^(10−i)
```

The calculated result is:

```text
R_sys ≈ 1.16 × 10^−15
```

Therefore, the system resilience index is almost zero under this low-SNR condition. This indicates that the network should trigger the P3 reconfiguration strategy and switch from the high-rate mode M0 to the resilient mode M1.

---

## Take-Home Exam Modification

This repository was modified for the take-home mid-term exam. The original implementation focuses on the micro-Doppler channel model, the L-FMCW estimation algorithm, and visualization.

I added a new script:

```text
q2_serviceability_resilience.py
```

This script extends the physical-layer micro-Doppler estimation result to a system-level serviceability and resilience analysis.

Specifically, it:

1. uses the RMS error bound from Eq. (20);
2. computes the RMS estimation error sigma under the Q2 parameters;
3. models the estimation error as a Gaussian random variable;
4. calculates the serviceability violation probability P_fail;
5. calculates the k-out-of-n system resilience index R_sys;
6. interprets whether the UAV network should switch from M0 to M1.

This modification connects the paper's micro-Doppler estimation error to 6G UAV-aided network serviceability and resilience.

---


## Dependencies

| Package    | Version tested | Purpose                       |
|------------|---------------|-------------------------------|
| numpy      | ≥ 1.24        | Array maths                   |
| scipy      | ≥ 1.10        | Signal processing              |
| matplotlib | ≥ 3.7         | Dark-theme plotting (Agg)     |

No GPU required. HTML animation requires no Python at all.

---

## Citation

```bibtex
@article{hou2021microdoppler,
  author  = {Hou, Hsin-An and Wang, Li-Chun and Lin, Hsin-Piao},
  title   = {Micro-{D}oppler Shift and Its Estimation in Rotary-Wing
             {UAV} Sub-6 {GHz} Communications},
  journal = {IEEE Wireless Communications Letters},
  volume  = {10},
  number  = {10},
  pages   = {2185--2189},
  year    = {2021},
  doi     = {10.1109/LWC.2021.3095898}
}
```
