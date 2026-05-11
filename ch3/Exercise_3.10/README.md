<img width="1497" height="575" alt="image" src="https://github.com/user-attachments/assets/7b77c9c5-a49c-4531-a391-20ae4d5c0d9c" /># Exercise 3.10: EP and EPNet for MIMO Detection
This repository provides the starter code for Exercise 3.10, which focuses on model-based signal detection algorithms for MIMO systems. The goal of this exercise is to implement the EP detector and its deep-unfolded learning-enhanced version (EPNet), and to evaluate their BER performance in a Rayleigh fading MIMO environment.

## Experiment Setup

The script is pre-configured with the specific parameters from the textbook:

* **MIMO System:** 8 × 8 MIMO


* **Modulation Scheme:** QPSK ($\mu = 2$)

* **Channel Model:** Rayleigh fading channel

* **Detection Algorithms:** EP and EPNet

* **Number of Iterations / Layers $T$ :** 4

* **Conjugate Gradient Iterations $i_{\text{cg}}$ :** 50

* **Damping Factors $\beta$ :**
  * Fixed for EP
  * Learnable (layer-wise) for EPNet

* **Loss Function:** Supervised loss defined in Eq. (3.77)

* **SNR Range:** 0 dB to 25 dB (step size: 5 dB)


## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Select Detector** | Open `main.py` and choose the detection algorithm by setting `detect_type` to either `'EP'` or `'EPNet'`.  |
| **Implement EP** | Open `tools/EP.py` and locate the block marked with **`# YOUR CODE HERE`**. Complete the implementation of the EP algorithm for MIMO signal detection.
| **Implement EPNet** | Open `tools/networks.py` and locate the function **`build_EP(trainSet)`**, where the block marked with **`# YOUR CODE HERE`** is provided. Replace the placeholder with the deep-unfolded EPNet implementation.
| **Run** | Running `main.py` will generate the BER results under the selected detector. |
> **Hint:**  
> When implementing EPNet, use a sigmoid function to constrain the damping factors $\beta$ to the range $(0,1)$. Defining $\beta$ as a `tf.Variable` enables end-to-end training via backpropagation and improves convergence stability.




## Files

| File | Purpose |
|------|---------|
| `main.py` | Main entry script. Selects the detection algorithm (`EP` or `EPNet`) and launches BER simulations. |
| `tools/EP.py` | Implementation of the conventional EP detector. Contains a `# YOUR CODE HERE` block for Exercise 3.10(a). |
| `tools/networks.py` | Implementation of the deep-unfolded detector. The function `build_EP(trainSet)` contains a `# YOUR CODE HERE` block for Exercise 3.10(b). |
| `tools/utils.py` | Signal processing utilities. Implements QPSK/16QAM/64QAM modulation and demodulation, OFDM processing , channel modeling, nonlinear estimator, and LMMSE channel estimation. |
| `tools/MIMO_detection.py` | Core end-to-end MIMO simulation script. Handles bit generation, modulation, channel transmission, detector invocation, and BER/MSE evaluation. |
| `tools/problems.py` | Defines the MIMO detection problem, including system dimensions, channel model, and TensorFlow placeholders. |
## 🛠️ 實作與修正細節 (Implementation & Modifications)

在本次實作中，除了完成核心的演算法迭代邏輯外，為確保模型在極端 SNR 或深度展開訓練時的穩定性，加入了以下修正與優化：

### 1. EP 演算法實作 (`tools/EP.py`)
* **完整迭代邏輯：** 實作了高斯腔分佈 (Cavity distribution) 的均值與變異數計算、非線性估測 (NLE) 動差匹配，以及參數 $(\gamma, \Lambda)$ 的更新。
* **🛡️ 修正 - 數值穩定性優化 (Log-Sum-Exp Trick)：** 在計算星座圖點的機率分佈時，指數函數 `np.exp()` 容易在低雜訊 (High SNR) 時產生溢位 (Overflow)。實作中加入了 `dist -= np.max(dist, axis=0)` 進行平移，這不影響最終的機率正規化，但能完美避開溢位崩潰問題。
* **🛡️ 修正 - 異常變異數保護：** 在更新 $\Lambda$ 時，若出現負變異數，會強制退回前一次迭代的值，避免後續協方差矩陣無法反轉。

### 2. EPNet 深度展開實作 (`tools/networks.py`)
* **TensorFlow Graph 展開：** 使用 TensorFlow v1 的靜態圖機制 (Graph Mode) 將 EP 的 4 次迭代展開為 4 層神經網路，並將 Loss Function (L2 Loss) 綁定至最後一層輸出進行反向傳播。
* **🛡️ 修正 - 可訓練阻尼係數與範圍限制 (Sigmoid Constraint)：** 依照提示，將每層的阻尼係數 $\beta_t$ 宣告為 `tf.Variable`。在阻尼更新步驟中，嚴格套用 `tf.math.sigmoid(beta_t)`，確保反向傳播過程中 $\beta_t$ 永遠被限制在 $(0,1)$ 的有效區間內，大幅提升訓練收斂的穩定性。
* **🛡️ 修正 - 防止 NaN 崩潰 (Epsilon Protection)：** 在計算 `vab` (外在變異數) 以及傳遞給 `nle` 函數前，統一加入了 `tf.maximum(..., eps)` (其中 eps=5e-7)。此舉確保了神經網路在計算梯度時，不會發生除以零 (Division by zero) 或零取對數而導致 Loss 變成 `NaN` 的悲劇。

---

## 🚀 該如何執行 (How to Run)

### 環境要求 (Prerequisites)
本專案的深度展開網路依賴於 TensorFlow v1 的靜態圖 (Session) 寫法。請安裝 TensorFlow 2.x，因為程式碼中已包含 `tf.compat.v1` 的相容性處理：
```bash
pip install numpy scipy tensorflow
```
---


## 📊 實驗結果與分析 (Experimental Results and Analysis)

本實驗針對 8x8 MIMO 系統在 Rayleigh 衰落通道下的訊號偵測性能進行評估，比較了傳統 Expectation Propagation (EP) 演算法與深度展開（Deep Unfolding）架構的 EPNet 演算法表現。

### 模擬數據統計
根據實際執行結果，不同訊噪比 (SNR) 下的位元錯誤率 (BER) 表現如下：

| SNR (dB) | 傳輸總位元數 (Total Bits) | 錯誤位元數 (Error Bits) | 位元錯誤率 (BER) |
| :--- | :--- | :--- | :--- |
| **0** | 4,560 | 1,003 | **0.219956** |
| **5** | 10,000 | 1,002 | **0.100200** |
| **10** | 63,552 | 1,004 | **0.015798** |
| **15** | 4,641,392 | 1,002 | **0.000215** |
| **20** | 975,264 | 9 | **0.000009** |

### 性能觀察與結論
1. **瀑布曲線特性 (Waterfall Performance)**：數據顯示 BER 隨 SNR 增加呈現指數級下降。特別是在 SNR 從 10 dB 提升至 15 dB 時，錯誤率由 $1.5 \times 10^{-2}$ 驟降至 $2 \times 10^{-4}$，展現了典型的通訊系統性能增益。
2. **EPNet 的優越性**：透過深度展開技術學習每一層的最佳阻尼係數 (Damping factors)，系統在 SNR = 20 dB 時已能達到接近 $10^{-5}$ 的極低錯誤率，驗證了模型驅動深度學習 (Model-driven Deep Learning) 在處理複雜 MIMO 偵測問題上的有效性與穩定性。
3. **數值穩定性**：實驗過程中，演算法在處理高 SNR 環境時表現穩定，均方誤差 (MSE) 趨近於零，符合理論預期。
