#  Exercise 2.7 — Data-Driven SISO-OFDM Channel Estimation

This project implements a **Deep Neural Network (DNN)-based channel estimator** for a SISO-OFDM system.  
The goal is to learn the mapping between received pilot signals and the channel response, and evaluate its performance under different Signal-to-Noise Ratio (SNR) conditions.

---

##  System Configuration

The OFDM system is configured as follows:

- **Subcarriers (K):** 64  
- **Pilot Symbol:** 1st OFDM symbol (64 QPSK pilots)  
- **Data Symbol:** 2nd OFDM symbol (64-QAM modulation)  
- **SNR Range:** 5 dB ~ 40 dB (step = 5 dB)  
- **Channel Type:** Rayleigh fading (generated in code)  
- **Channel Estimation Methods:**
  - DNN-based estimator (implemented)
  - LMMSE estimator (baseline)

---

##  Methodology (原理說明)

### 1. Channel Estimation Problem

在 OFDM 系統中，接收訊號可表示為：

Y = XH + N

其中：

- `Y`：接收訊號  
- `X`：已知 pilot  
- `H`：通道響應（未知）  
- `N`：雜訊  

傳統方法（LS / LMMSE）需要：
- 通道統計資訊
- 線性假設

---

### 2. DNN-Based Estimation

本專案改用 **data-driven 方法**：
