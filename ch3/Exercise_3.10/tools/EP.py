#!/usr/bin/python

import sys
import numpy as np
import numpy.linalg as la
import math
from .utils import QAM_Modulation, NLE, de2bi, _QPSK_Constellation, _16QAM_Constellation, _64QAM_Constellation

beta = np.zeros(20)
para = {}
# loading the trained model parameters
try:
    for k,v in np.load("EP_4×4_16QAM_15dB_I_1.npz").items():
        para[k] = v
except IOError:
    print("no such file")
    pass
# get parameters for CG-OAMP-NET
for t in range(20):
    if para.get("beta_"+str(t)+":0",-1) != -1:
        beta[t] = para["beta_"+str(t)+":0"]
beta = 1. / (1. + np.exp(-beta))


def EP(x,A,y,noise_var,T=10,mu=2,soft=False,pp_llr=None):  # ub as output, stable

    M = A.shape[0]
    N = A.shape[1]
    
    # 預先計算以節省運算時間
    AT = A.T
    ATA = AT @ A
    MSE = np.zeros(T)
    
    # 初始化先驗 LLR (pp_llr)
    if pp_llr is None:
        pp_llr = np.zeros((mu//2, N))
    else:
        pp_llr = np.concatenate((pp_llr[:, :mu//2], pp_llr[:, mu//2:]), axis=0)
        
    # 產生位元映射表
    bin_array = np.sign(de2bi(np.arange(2**(mu // 2)), mu // 2) - 0.5).astype(int)
    
    # 定義星座圖正規化因子
    if mu == 2:
        constellation_norm = np.array([-1, +1]) / np.sqrt(2)
    elif mu == 4:
        constellation_norm = np.array([-3, -1, +3, +1]) / np.sqrt(10)
    else:
        constellation_norm = np.array([-7, -5, -1, -3, +7, +5, +1, +3]) / np.sqrt(42)
        
    # 計算星座圖軟估測的平均值與變異數
    dist = 0.5 * bin_array @ pp_llr
    dist -= np.max(dist, axis=0)  # 減去最大值以維持數值穩定性 (避免 exp 溢位)
    
    probs = np.exp(dist).T # (N, 2**(mu//2))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    
    s_est = np.sum(probs * constellation_norm, axis=1, keepdims=True) # (N, 1)
    e_est = np.sum(probs * (s_est * np.ones((N, 2 ** (mu // 2))) - constellation_norm) ** 2, axis=1, keepdims=True)
    e_est = np.maximum(e_est, 1e-8)
    
    # 計算 EP 的初始參數對 (gamma, Lambda)
    Lambda = (1 / e_est).reshape(N) # (N,)
    gamma = s_est * Lambda.reshape(N, 1) # (N, 1)
    
    # 開始 EP 迭代
    for t in range(T):
        # 1. Compute the mean and covariance matrix
        Sigma = la.inv(ATA + noise_var * np.diag(Lambda))
        Mu = Sigma @ (AT @ y + noise_var * gamma)
        MSE[t] = np.mean((x - Mu)**2)
        
        # 2. Compute the extrinsic mean and covariance matrix
        diag = noise_var * np.diag(Sigma).reshape(N, 1)
        vab = diag / (1 - diag * Lambda.reshape(N, 1))
        vab = np.maximum(vab, 5e-7)
        uab = vab * (Mu / diag - gamma)
        
        # 3. Compute the posterior mean and covariance matrix
        if soft:
            ub, vb, ext_probs = NLE(vab, uab, orth=False, mu=mu, EP=True, norm=np.sqrt(1), soft=True)
            ext_probs = np.maximum(np.exp(-(uab * np.ones((N, 2 ** (mu//2))) - constellation_norm) ** 2 / (2 * vab)), 1e-100)
            post_probs = probs * ext_probs
            post_probs = post_probs / np.sum(post_probs, axis=1, keepdims=True)
            ub = np.sum(post_probs * constellation_norm, axis=1, keepdims=True)
            vb = np.sum(post_probs * (ub * np.ones((N, 2 ** (mu//2))) - constellation_norm) ** 2, axis=1, keepdims=True)
        else:
            ub, vb = NLE(vab, uab, orth=False, mu=mu, EP=True, norm=np.sqrt(1))
            vb = np.maximum(vb, 5e-13)
            
        # 4. Update gamma and Lambda
        gamma_last = gamma
        Lambda_last = Lambda
        
        gamma = (ub * vab - uab * vb) / (vb * vab)
        Lambda = ((vab - vb) / (vb * vab)).reshape(N)
        
        # 避免負變異數的異常情況
        idx = Lambda < 0
        Lambda[idx] = Lambda_last[idx]
        gamma[idx] = gamma_last[idx]
        
        # 5. Damping (引入全域載入的 beta 陣列)
        # 注意：這裡呼叫你在程式最上方解析出來的 beta 陣列
        current_beta = beta[t] if isinstance(beta, np.ndarray) else 0.2
        
        gamma = current_beta * gamma + (1 - current_beta) * gamma_last
        Lambda = current_beta * Lambda + (1 - current_beta) * Lambda_last

    if soft:
        return ub, MSE, ext_probs

    return ub, MSE
