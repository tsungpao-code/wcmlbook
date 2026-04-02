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
(Yp, Xp) → DNN → Ĥ

也就是讓神經網路直接學習：

「接收訊號 → 通道」

---

### 3. Data Representation

由於訊號是 complex number，我們轉為 real-valued：

- Input:
[Re(Yp), Im(Yp), Re(Xp), Im(Xp)] → 4K

- Output:
- [Re(H), Im(H)] → 2K
  
---

### 4. Network Architecture

- Fully Connected Neural Network (MLP)
- 2 hidden layers (ReLU)
- 1 output layer (linear)

Loss function:

MSE = ||Ĥ - H||²


Optimizer:

- Adam
optimizer = tf.train.AdamOptimizer(learning_rate=lr_)
並搭配 learning rate decay：
lr = initial_lr × decay_rate^(step / decay_steps)

---
### 5. 加入 CP / 無 CP 比較機制

使用兩種 OFDM 模型：
- 無 CP → ofdm_simulate_cp_free()
- 有 CP → ofdm_simulate()

可以分析 CP 對 channel estimation 的影響

---
### 6. 建立四條線比較
新增：
main_compare_4lines.py
用來同時比較：
| 方法    | CP |
| ----- | -- |
| DNN   | 有  |
| LMMSE | 有  |
| DNN   | 無  |
| LMMSE | 無  |



---
### 7. 繪製比較圖

輸出：
compare_results_4lines.png
圖中包含四條曲線：
-DNN with CP
-LMMSE with CP
-DNN without CP
-LMMSE without CP



---
## 🛠 Implementation Details

### ✔ Completed `build_ce_dnn()`

- Defined input/output placeholders  
- Built DNN architecture  
- Implemented forward pass  
- Defined MSE loss  
- Added optimizer  

---
## How to Run
### Step 1：Train DNN
修改 main.py：
ce_type = 'dnn'
test_ce = False
CP_flag = True

執行：
cd C:\Data-Driven
python main.py
會訓練模型並儲存：
dnn_ce/CE_DNN_*.npz
### Step 2：Test DNN
修改：
ce_type = 'dnn'
test_ce = True
執行：
python main.py

會輸出：

MSE_T
MSE_F

並產生 .mat 檔
### Step 3：Run LMMSE Baseline
ce_type = 'mmse'
test_ce = True
與 DNN 做比較
### Step 4（Optional）：No CP
CP_flag = False
可觀察無 CP 情況下效能
### Training Results
在 SNR = 40 dB 時：

Test MSE ≈ 0.008
模型成功收斂
無 NaN 或 divergence
### Observations
DNN 能成功學習 channel estimation mapping
高 SNR 時效果顯著優於低 SNR
訓練過程穩定
與 LMMSE 可進行性能比較
### Conclusion
本實驗成功實現 DNN-based channel estimation，並驗證：

深度學習可取代傳統 channel estimation 方法
在高 SNR 下具有良好性能
可應用於未來 data-driven wireless systems
