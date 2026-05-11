# Exercise 3.9: OAMP and OAMP-Net for MIMO Detection

This repository provides the skeleton code for Exercise 3.9. Your task is to implement the **OAMP** (Orthogonal Approximate Message Passing) algorithm and its deep learning-unrolled version, **OAMP-Net**, to solve the MIMO signal detection problem. You will compare the performance of the traditional iterative algorithm with the learning-based approaches.

## Experiment Setup
The scripts are pre-configured with the following system parameters:
* **Antenna Configuration:** $8 \times 8$ MIMO ($N_t = 8, N_r = 8$)
* **Modulation Scheme:** QPSK ($\mu = 2$)
* **Channel Model:** Rayleigh fading channel (Real-valued decomposition used)
* **Network/Iteration Depth:** 10 iterations (layers)
* **SNR Range:** 0 dB to 25 dB with 5 dB increment
* **Training Setup:** (For Task b & c) Supervised learning with MSE loss, Adam optimizer, SNR = 20 dB for training.

## What You Need To Do

| Checklist | Details |
| :---- | :--- |
| **(a) OAMP** | Open `OAMP_ex_a.py`. Fill in the **5 key steps** of the OAMP algorithm: (1) LMMSE matrix $W_{LMMSE}$, (2) Trace normalization for $W_t$, (3) Linear residual $r_t$, (4) Posterior variance $\tau^2$, and (5) Signal variance $v^2$ update. |
| **(b) OAMPNet-b** | Open `OAMP_ex_b.py`. Implement the OAMP-Net within a PyTorch `nn.Module`. Use learnable parameters $(\gamma_t, \theta_t)$ as defined in (3.73) and (3.74). Train the model and plot the BER curve. |
| **(c) OAMPNet-c** | Open `OAMP_ex_c.py`. Extend the model to make $(\gamma_t, \theta_t, \phi_t, \xi_t)$ all learnable parameters. The parameters $(\phi_t, \xi_t)$ are used to optimize the nonlinear MMSE denoiser. |
| **(d) Comparison** | Run all three scripts and compare the resulting BER vs. SNR curves. Observe how learning-based parameters affect the convergence and final performance. |

## How to Run
| Task | Execution Command |
| :--- | :--- |
| **Run OAMP** | `python OAMP_ex_a.py` |
| **Train/Test OAMP-Net-b** | `python OAMP_ex_b.py` |
| **Train/Test OAMP-Net-c** | `python OAMP_ex_c.py` |

> **Note:** The PyTorch scripts will automatically detect if a GPU (CUDA) is available.

## Hint
* **Mathematical Reference:** Please strictly follow the formulations in **Section 3.2.3** for standard OAMP and **Section 3.3.2** for the OAMP-Net architecture.
* **Denoiser:** In Task (c), ensure that the learnable $\phi_t$ and $\xi_t$ are properly integrated into the MMSE denoising function as described in equation (3.76).
* **Matrix Inversion:** In the LMMSE step, remember to add a small regularization term (noise variance) to the diagonal to ensure numerical stability during matrix inversion.

## Files
| File | Purpose |
| :--- | :--- |
| `OAMP_ex_a.py` | NumPy-based skeleton code for the traditional OAMP algorithm (Task a). |
| `OAMP_ex_b.py` | PyTorch-based skeleton code for OAMP-Net with learnable $(\gamma, \theta)$ (Task b). |
| `OAMP_ex_c.py` | PyTorch-based skeleton code for OAMP-Net with learnable $(\gamma, \theta, \phi, \xi)$ (Task c). |

# Exercise 3.9: OAMP and OAMP-Net for MIMO Detection

本專案提供練習題 3.9 的骨架程式碼。任務是實作 **OAMP**（正交近似信息傳遞）演算法及其深度學習展開版本 **OAMP-Net**，以解決 MIMO 信號檢測問題。您將比較傳統迭代演算法與基於學習的方法之性能。

## 實驗設定 (Experiment Setup)
腳本預先配置了以下系統參數：
* **天線配置：** $8 \times 8$ MIMO ($N_t = 8, N_r = 8$)
* **調變方案：** QPSK ($\mu = 2$)
* **通道模型：** 瑞利衰落通道（使用實數值分解）
* **網絡/迭代深度：** 10 次迭代（層）
* **SNR 範圍：** 0 dB 到 25 dB，增量為 5 dB
* **訓練設定：** （針對任務 b 和 c）使用 MSE 損失函數的監督式學習、Adam 優化器，訓練 SNR = 20 dB。

---

## 實作進度更新 (Implementation Updates)

### 已完成：任務 (a) 傳統 OAMP 演算法 (`OAMP_ex_a.py`)
目前已成功在 `oamp_detector` 函式中，使用 NumPy 矩陣運算實作了 OAMP 演算法的 **5 個關鍵迭代步驟**：

1.  **LMMSE 矩陣計算 (LMMSE Matrix Calculation)：**
    實作了線性最小均方誤差估測器，並加入噪聲方差作為正則化項（Regularization term），以確保矩陣求逆時的數值穩定性。
2.  **跡規範化 (Trace Normalization)：**
    對線性估測矩陣 $W_{LMMSE}$ 進行縮放，使其滿足跡條件 $tr(W_t H) = K$，以維持誤差項的正交性。
3.  **線性殘差計算 (Linear Residual Update)：**
    根據當前的估計值計算線性觀測量 $r_t$。
4.  **後驗方差估計 (Posterior Variance Estimation)：**
    結合先驗方差與加性高斯白噪聲（AWGN），計算有效噪聲的方差 $\tau^2$。
5.  **信號方差更新 (Signal Variance Update)：**
    根據非線性 MMSE 降噪器後的殘差值，更新下一輪迭代所需的信號方差 $v^2$。

---

## 如何執行 (How to Run)

本實驗包含三個主要腳本，請依照下列步驟執行：

### 1. 環境準備
請確保您的 Python 環境已安裝 `numpy`, `matplotlib` 以及 `torch` 。

### 2. 執行任務指令
請在終端機（Terminal）中輸入以下指令來執行不同任務：

| 任務 | 執行指令 | 說明 |
| :--- | :--- | :--- |
| **執行 OAMP** | `python OAMP_ex_a.py` | 執行傳統迭代演算法（任務 a）。 |
| **訓練/測試 OAMP-Net-b** | `python OAMP_ex_b.py` | 針對可學習參數 $(\gamma, \theta)$ 進行 OAMP-Net 訓練與測試（任務 b）。 |
| **訓練/測試 OAMP-Net-c** | `python OAMP_ex_c.py` | 針對可學習參數 $(\gamma, \theta, \phi, \xi)$ 進行 OAMP-Net 訓練與測試（任務 c）。 |

### 3. 結果比較
執行完畢後，可以觀察並比較三者的 **BER vs. SNR** 效能曲線，分析學習型參數如何影響收斂速度與最終檢測性能。
傳統 OAMP (Task a):

表現： 根據實測結果 a_result.png，BER 曲線隨 SNR 增加而下降，但在高 SNR 區域（15 dB 以上）出現飽和（Plateau）甚至微幅上升的現象。

分析： 傳統迭代演算法高度依賴於精確的數學假設與參數調整。在沒有可學習參數的情況下，迭代中的正交性偏差可能導致收斂效果在高 SNR 環境下受限。
OAMP-Net-b (Task b):表現： 根據 b_result.png，引入可學習參數 $(\gamma_t, \theta_t)$ 後，BER 曲線呈現穩定的指數型下降，在 25 dB 時效能顯著優於傳統 OAMP，達到約 $10^{-4}$ 等級。分析： 透過監督式學習，網路能夠自動調整步長與參數，有效補償了傳統 OAMP 在實際運算中的估計偏差，提升了檢測的穩定性。
OAMP-Net-c (Task c):表現： 根據 c_result.png，在全參數 $(\gamma_t, \theta_t, \phi_t, \xi_t)$ 學習的設定下，獲得了本實驗中的最佳效能。分析： 額外將降噪器參數 $(\phi_t, \xi_t)$ 納入優化，使得模型驅動的非線性 MMSE 降噪步驟能更靈活地處理非高斯噪聲特徵，最大程度地逼近最佳檢測性能。
根據實驗數據對比，可以得出以下核心結論：

深度展開 (Deep Unrolling) 的優越性： 將傳統通訊演算法展開為深度神經網路（OAMP-Net），能有效克服傳統迭代法在複雜通道環境下的效能瓶頸。

參數學習的深度影響： 參數學習的自由度越高（如 Task c 同時優化降噪器參數），系統對通道不確定性的適應能力越強，最終獲得的誤碼率（BER）越低。

模型驅動與數據驅動的結合： OAMP-Net 保留了傳統演算法的結構（Model-driven）並結合了深度學習的優化能力（Data-driven），在有限的 10 次迭代內即可達到優異的性能

