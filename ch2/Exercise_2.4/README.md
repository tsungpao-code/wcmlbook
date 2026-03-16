# Exercise 2.4：Channel GAN Implementation

本作業實作 Exercise 2.4，目標是利用 **Conditional Generative Adversarial Network (CGAN)** 來模擬 Rayleigh fading 通道的接收訊號分布。此方法的核心概念是在不知道完整通道資訊（CSI）的情況下，透過 GAN 學習通道輸出的統計特性。

本實驗使用 QuaDRiGa 生成的通道資料 `rayleigh_channel_dataset.mat` 作為訓練資料來源，並在 Python 程式 `Exercise_2_4_starter.py` 中完成原本 TODO 的資料生成函式 `generate_real_samples_with_labels_Rayleigh()`，用來產生 GAN 訓練所需的真實樣本。

在該函式中，首先從通道資料集中隨機選取通道係數 \(h\)，接著隨機生成 **16-QAM 傳送符號 \(x\)**。然後根據無線通道模型

\[
y = hx + n
\]

計算接收訊號，其中 \(n\) 為加性高斯白雜訊（AWGN）。由於神經網路無法直接處理複數數據，因此將接收訊號 \(y\) 拆分為 **實部與虛部** 作為輸入特徵。此外，為了讓 CGAN 能夠在特定條件下生成樣本，本實驗建立了 **conditioning vector**，其內容包含傳送符號與通道係數的實部與虛部。透過這些資料，GAN 可以學習 Rayleigh fading 通道下接收訊號的統計分布。

---

# 實驗設定

本實驗使用的 GAN 設定如下：

- **Dataset**：`rayleigh_channel_dataset.mat`（由 QuaDRiGa 生成）
- **Modulation**：16-QAM
- **Channel Model**：\(y = hx + n\)
- **Noise Model**：Gaussian noise（AWGN）
- **GAN Architecture**：Conditional GAN（Generator + Discriminator）
- **Noise Vector Dimension (Z)**：16
- **Training Iterations**：750000 iterations

---

# 程式實作內容

在 `generate_real_samples_with_labels_Rayleigh()` 函式中完成以下四個步驟：

1. **隨機選取通道係數**  
   從通道資料集中隨機抽取通道係數 \(h\)。

2. **生成隨機 QAM 符號**  
   從 16-QAM 星座點中隨機選取傳送符號 \(x\)。

3. **模擬接收訊號**  
   使用通道模型 \(y = hx + n\) 計算接收訊號，並加入高斯雜訊。

4. **建立 conditioning vector**  
   將傳送符號與通道係數的實部與虛部組合為條件向量，作為 CGAN 的輸入條件。

這些資料將作為 GAN 訓練中的 **real samples**。

---

# 如何執行

## Step 1：生成通道資料

先在 MATLAB 中執行 QuaDRiGa 腳本：

```matlab
QuaDRiGa_channel_generator
執行後會生成通道資料檔：
rayleigh_channel_dataset.mat

## Step 2：執行 GAN 訓練

確認 `rayleigh_channel_dataset.mat` 與 `Exercise_2_4_starter.py` 位於同一個資料夾，然後執行：

```bash
python Exercise_2_4_starter.py

## Step 3：觀察結果
程式在訓練過程中會產生接收訊號在複數平面上的分布圖：
結果圖會儲存在：
ChannelGAN_Rayleigh_images/
模型 checkpoint 會儲存在：
Models/
若 GAN 成功學習 Rayleigh fading 通道分布，則生成的點會在複數平面上呈現集中於原點附近的隨機散佈形狀。
