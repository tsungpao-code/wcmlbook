# Exercise 1.12: PPO Implementation for MountainCar-v0

## 作者資訊
- **姓名**：[曾宗保]
- **學號**：[313513035]

## 實作內容
本程式補全了 `PPO.update()` 方法，實作了近端策略優化 (PPO) 演算法。
主要更新包括：
1. 計算重要性比率 (Ratio)。
2. 實作截斷代理損失 (Clipped Surrogate Objective)。
3. 使用 Adam 優化器更新 Actor 與 Critic 網路權重。

## 如何執行
確保安裝 `torch` 與 `gym` 後，執行：
`python PPO_MountainCar-v0.py`
