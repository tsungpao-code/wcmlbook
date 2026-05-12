# Exercise 3.1: Learning-based Signal Detection for OFDM Systems

This repository provides the starter code for Exercise 3.1. Your task is to use Deep Learning (FC-DNN) to implicitly estimate the channel and recover transmitted bits in an OFDM system, based on the paper by Ye et al. [15]. You will modify network dimensions, modulation schemes, and pilot configurations to evaluate the system's Bit Error Rate (BER).


## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Task (b)** | Open `main.py`. Implement a loop to iterate `config.SNR` from 5 to 25 dB. <br> Modify `config.Pilots` (e.g., 8, 16, 64) to evaluate the impact of **different pilot numbers**, and simulate a **no-pilot** scenario. Collect BER results to reproduce Figure 3.3. |
| **Task (c)** | Open `Train.py` and `Test.py` and change `mu = 6` for **64-QAM**. <br> Open `main.py` and change `config.pred_range = np.arange(48, 96)`. <br> Open `Train.py` and change the network output size to `n_output = 48`. |
| **Task (d)** | Revert to QPSK (`mu = 2`). Implement a **single large DNN**: <br> • Open `main.py` and set `config.pred_range = np.arange(0, 128)`. <br> • Open `Train.py` and set `n_output = 128`. <br> • Increase `n_hidden_1`, `n_hidden_2`, etc., in `Train.py` to give the network enough capacity. |
| **Run** | Execute: `python main.py` for each specific configuration. |
| **Observe** | The console will output the testing BER. You should manually record these values to plot BER vs. SNR curves and compare the performance differences in your report. |


> **Hint:** The codes have been tested on Ubuntu 16.04 + tensorflow 1.1 + Python 2.7.



## Files

| File | Purpose |
|------|---------|
| `main.py` | Main script where you configure hyperparameters (`sysconfig`) and run the pipeline. |
| `DNN_Detection/Train.py` | Contains the FC-DNN architecture definition and the training loop. |
| `DNN_Detection/Test.py` | Evaluates the trained models and calculates the final BER. |
| `DNN_Detection/utils.py` | Mathematical formulations for OFDM (IDFT, CP addition, channel convolution). |
| `H_dataset/` | Pre-generated dataset containing Rayleigh fading channel responses. |
> **Hint:** The H_dataset can be downloaded from the following link: https://github.com/haoyye/OFDM_DNN.

[15] H. Ye, G. Y. Li, and B.-H. Juang, “Power of deep learning for channel estimation and signal detection in OFDM systems,” *IEEE Wireless Communications Letters*,
vol. 7, no. 1, pp. 114–117, Feb. 2018.

---

## 任務(b)實驗結果與說明

本次 Exercise 3.1 的主要目標是使用 learning-based signal detection 方法，透過 FC-DNN 在 OFDM 系統中進行通道隱式估測與 transmitted bits 偵測。實驗以 QPSK modulation 為主要設定，其中 `mu = 2`，OFDM subcarriers 數量為 `K = 64`，並比較不同 SNR 與不同 pilot 數量對 BER（Bit Error Rate）的影響。

在 Task (b) 中，SNR 設定為 5、10、15、20、25 dB，pilot 數量則設定為 8、16、64。透過訓練對應的 DNN model，並在 testing 階段 restore 對應 checkpoint，最後由 terminal 輸出 mean error 與 BER 結果。本次程式執行後不會自動產生圖片，而是輸出數值結果，因此需要手動記錄 terminal 中最後輸出的 BER，並整理成表格或自行繪製 BER vs. SNR 曲線。

Task (b) 的 BER 結果如下表所示：

| SNR (dB) | Pilot = 8 BER | Pilot = 16 BER | Pilot = 64 BER |
|---:|---:|---:|---:|
| 5 | 0.280119360 | 0.124803126 | 0.045217386 |
| 10 | 0.244356500 | 0.047893107 | 0.012438572 |
| 15 | 0.230941890 | 0.017064989 | 0.004126915 |
| 20 | 0.226443770 | 0.007111251 | 0.001527438 |
| 25 | 0.225586240 | 0.004008114 | 0.000816294 |

由表格結果可以觀察到，當 SNR 增加時，BER 整體呈現下降趨勢。這是因為 SNR 越高，代表接收訊號中雜訊成分相對較低，因此 FC-DNN detector 可以更準確地從 received OFDM signal 中恢復 transmitted bits。

另外，pilot 數量的增加也明顯改善 BER 表現。以 SNR = 25 dB 為例，Pilot = 8 時 BER 為 0.225586240，Pilot = 16 時 BER 降為 0.004008114，而 Pilot = 64 時 BER 進一步下降至 0.000816294。這表示較多的 pilot 可以提供更充分的通道資訊，使 DNN 更容易學習通道響應與 transmitted bits 之間的關係，因此能有效降低偵測錯誤率。

整體而言，本實驗結果顯示，在 OFDM learning-based signal detection 中，提高 SNR 與增加 pilot 數量皆能提升系統偵測效能，其中 pilot 數量對 BER 的改善尤其明顯。

在程式執行流程方面，首先需要在 `main.py` 中設定實驗參數，例如 `Pilots = 8`、`SNR = 20`、`snr_list = [20]`，並將 `IS_Training = True` 進行模型訓練。訓練完成後，模型會儲存在 `Models/SNR_xx/` 資料夾中，例如 `DetectionModel_SNR_20_Pilot_8_epoch_195`。接著將 `IS_Training` 改為 `False`，並把 `model_name` 指向對應的 checkpoint，例如 `model_name = r'C:\ml_learning_hw3\Exercise_3.1\Models\SNR_20\DetectionModel_SNR_20_Pilot_8_epoch_195'`，再重新執行程式進行測試。測試完成後，terminal 會輸出類似 `OFDM Detection QAM output number is 16 SNR = 20 Num Pilot 8 prediction and the mean error on test set are: 0.22807147 0.22644377` 的結果，其中最後一個數值就是 BER。

在實作過程中，原始程式是使用 TensorFlow 1.x 撰寫，因此若在 TensorFlow 2.x 環境下執行，需要將 `import tensorflow as tf` 改為 `import tensorflow.compat.v1 as tf`，並加上 `tf.disable_v2_behavior()`，才能正常使用 `tf.placeholder()`、`tf.Session()` 與 `tf.train.Saver()` 等 TensorFlow 1.x 語法。

另外，讀取 `H_dataset` 時應使用 `os.path.join()` 來組合資料路徑，例如 `H_file = os.path.join(H_folder, str(test_idx) + '.txt')`，避免產生 `H_dataset1.txt` 這類錯誤路徑。若出現 checkpoint 錯誤，例如 `ValueError: The passed save_path is not a valid checkpoint`，則需要確認 `model_name` 是否正確指向已存在的 checkpoint，而且路徑不需要加上 `.index`、`.meta` 或 `.data-00000-of-00001`。

此外，當 pilot 數量改變時，pilot file 也需要重新產生，因為 pilot bit 長度應為 `P * mu`，而不是 `K * mu`。若舊的 `Pilot_8`、`Pilot_16` 或 `Pilot_64` 檔案仍存在，可能造成 `ValueError: shape mismatch`，因此需要刪除舊 pilot 檔案或在程式中加入長度檢查，確保 pilotValue 的長度與 pilotCarriers 數量一致。

總結來說，本次實驗成功完成不同 SNR 與不同 pilot 數量下的 BER 比較，結果證明 learning-based OFDM detector 的效能會受到 SNR 與 pilot 設定影響；當 SNR 較高或 pilot 數量較多時，模型能取得更好的偵測結果，BER 也會明顯下降。
## Task (c)：64-QAM 訊號偵測

在 Task (c) 中，本實驗將 modulation scheme 由 QPSK 改為 64-QAM，並在其他模擬設定大致不變的情況下，觀察調變方式改變後對 BER 的影響。由於 64-QAM 每個 symbol 可以承載 6 bits，因此需要將 `Train.py` 與 `Test.py` 中的 `mu` 由 `2` 改為 `6`。此外，因為每個 DNN 需要預測的 bit 數增加，所以也需要將 `main.py` 中的 `pred_range` 改為 `np.arange(48,96)`，並將 `Train.py` 與 `Test.py` 中的 DNN output size 改為 `n_output = 48`。

本小題使用 64-QAM modulation，並以 Pilot = 16 作為實驗設定，分別測試 SNR = 5、10、15、20、25 dB 下的 BER 表現。實驗結果如下表所示：

| SNR (dB) | 64-QAM，Pilot = 16 BER |
|---:|---:|
| 5 | 0.35137498 |
| 10 | 0.32710410 |
| 15 | 0.31131250 |
| 20 | 0.30381250 |
| 25 | 0.30039500 |

由表格結果可以觀察到，當 SNR 增加時，BER 整體呈現下降趨勢，表示在較高 SNR 的情況下，接收訊號受到 noise 的影響較小，因此 DNN detector 可以更準確地恢復 transmitted bits。不過，相較於 Task (b) 中 QPSK 的結果，64-QAM 的 BER 明顯較高。這是因為 64-QAM 的 constellation points 較密集，相鄰 symbols 之間的 Euclidean distance 較小，因此更容易受到 noise 與 channel distortion 影響，造成 bit detection error 增加。

整體而言，64-QAM 雖然可以在每個 symbol 中傳送更多 bits，因此具有較高的 spectral efficiency，但也因為 constellation 較密集，所以需要更好的通道條件與更強的 detection model 才能達到較低的 BER。與 QPSK 相比，在相同 pilot 數量與 SNR 條件下，64-QAM 的 BER 表現較差，顯示高階調變雖然能提高傳輸速率，但也會增加偵測難度。

---

## Task (d)：Single Large DNN

在 Task (d) 中，本實驗將 modulation scheme 改回 QPSK，因此 `mu = 2`。由於 OFDM 系統共有 64 個 subcarriers，而 QPSK 每個 symbol 可承載 2 bits，因此每個 OFDM data symbol 共有 128 個 transmitted bits。為了讓單一 DNN 一次預測完整的 bit vector，本實驗將 `main.py` 中的 `pred_range` 改為 `np.arange(0,128)`，並將 `Train.py` 與 `Test.py` 中的 DNN output size 改為 `n_output = 128`。

原本的架構是使用多個 smaller DNN，每個 DNN 只負責預測一小段 transmitted bits，例如 output size = 16。Task (d) 則改為使用 single large DNN，讓同一個 DNN 一次輸出完整的 128 bits。這樣的做法可以讓模型直接預測整個 transmitted bit vector，但由於 output dimension 從 16 增加到 128，模型需要學習的 mapping 更複雜，因此訓練難度也會提高。

本實驗使用 QPSK、Pilot = 16、SNR = 20 dB 作為比較設定，並將 single large DNN 與原本的 smaller DNN 架構進行比較。比較結果如下表所示：

| 方法 | Modulation | Pilot | SNR (dB) | Output Size | BER |
|---|---|---:|---:|---:|---:|
| 原本的 smaller DNN | QPSK | 16 | 20 | 16 | 0.007111251 |
| Single large DNN | QPSK | 16 | 20 | 128 | 0.145289060 |

由比較結果可以觀察到，single large DNN 的 BER 高於原本的 smaller DNN。原本 smaller DNN 在 Pilot = 16、SNR = 20 dB 時的 BER 為 0.007111251，而 single large DNN 的 BER 為 0.145289060。這表示雖然 single large DNN 可以一次預測全部 128 個 transmitted bits，但由於輸出維度較大，模型需要同時學習更多 bit positions 的對應關係，因此訓練難度明顯增加。

造成 single large DNN BER 較高的原因，主要是因為 output size 從 16 增加到 128 後，模型參數需要處理更複雜的輸入與輸出關係。如果 hidden layer 容量不足，或 training epochs 不夠多，模型可能無法充分收斂，因此 BER 會比原本多個 smaller DNN 的架構更高。相反地，原本的 smaller DNN 每次只需要預測部分 bits，學習任務較簡單，因此能得到較低的 BER。

整體而言，single large DNN 的優點是架構較直觀，且可以一次輸出完整的 transmitted bit vector；缺點是模型輸出維度較大、訓練較困難，且需要更大的 network capacity 與更多訓練時間才能達到較好的 BER 表現。因此，若以 BER 效能作為主要考量，原本使用多個 smaller DNN 分段預測 bits 的方法表現較佳；若以模型架構簡化與一次輸出完整 bit vector 為目標，single large DNN 則是一種可行但需要進一步調整模型容量的方式。
