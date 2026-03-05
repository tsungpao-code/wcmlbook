import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid
import wireless_networks_generator as wg
import helper_functions
from FPLinQ import FP_optimize
import time


# ================= 1. 函数定义 (修复版) =================

def simple_greedy(X, AAA, label):
    n, K = X.shape[0], X.shape[1]
    thd = int(np.sum(label) / n)
    Y = np.zeros((n, K))
    for ii in range(n):
        alpha = AAA[ii, :]
        H_diag = alpha * np.square(np.diag(X[ii, :, :]))
        xx = np.argsort(H_diag)[::-1]
        for jj in range(thd):
            Y[ii, xx[jj]] = 1
    return Y


def normalize_data(train_data, test_data):
    tr_layouts, tr_K, _ = train_data.shape
    te_layouts, te_K, _ = test_data.shape
    mask_tr = np.eye(tr_K)
    train_copy = np.copy(train_data)
    diag_H = np.multiply(mask_tr, train_copy)
    diag_mean = np.sum(diag_H) / (tr_layouts * tr_K)
    diag_var = np.sqrt(np.sum(np.square(diag_H)) / (tr_layouts * tr_K))
    tmp_diag = (diag_H - diag_mean) / diag_var
    off_diag = train_copy - diag_H
    off_diag_mean = np.sum(off_diag) / (tr_layouts * tr_K * (tr_K - 1))
    off_diag_var = np.sqrt(np.sum(np.square(off_diag)) / (tr_layouts * tr_K * (tr_K - 1)))
    tmp_off = (off_diag - off_diag_mean) / off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off, mask_tr)
    norm_train = np.multiply(tmp_diag, mask_tr) + tmp_off_diag
    mask_te = np.eye(te_K)
    test_copy = np.copy(test_data)
    diag_H_te = np.multiply(mask_te, test_copy)
    tmp_diag_te = (diag_H_te - diag_mean) / diag_var
    off_diag_te = test_copy - diag_H_te
    tmp_off_te = (off_diag_te - off_diag_mean) / off_diag_var
    tmp_off_diag_te = tmp_off_te - np.multiply(tmp_off_te, mask_te)
    norm_test = np.multiply(tmp_diag_te, mask_te) + tmp_off_diag_te
    return norm_train, norm_test


def build_graph(loss, dist, norm_dist, norm_loss, K, threshold):
    x1 = np.expand_dims(np.diag(norm_dist), axis=1)
    x2 = np.expand_dims(np.diag(norm_loss), axis=1)
    x3 = np.zeros((K, 1))
    x = torch.tensor(np.concatenate((x1, x2, x3), axis=1), dtype=torch.float)

    dist2 = np.copy(dist)
    mask = np.eye(K)
    dist2 = dist2 + 1000 * mask
    if threshold > 0:
        dist2[dist2 > threshold] = 0

    res = np.nonzero(dist2)
    attr_ind = np.array(res)  # 修复 AttributeError: 'tuple' object has no attribute 'shape'

    edge_attr = torch.tensor(np.expand_dims(norm_dist[res], axis=-1), dtype=torch.float)
    adj = np.zeros(attr_ind.shape)
    adj[0, :] = attr_ind[1, :]
    adj[1, :] = attr_ind[0, :]
    edge_index = torch.tensor(adj, dtype=torch.long)
    y = torch.tensor(np.expand_dims(loss, axis=0), dtype=torch.float)
    return Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=y)


def proc_data(HH, dists, norm_dists, norm_HH, K, threshold):
    return [build_graph(HH[i, :, :], dists[i, :, :], norm_dists[i, :, :], norm_HH[i, :, :], K, threshold) for i in
            range(HH.shape[0])]


# ================= 2. 网络结构 =================

class IGConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(IGConv, self).__init__(aggr='max', **kwargs)
        self.mlp1, self.mlp2 = mlp1, mlp2

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.mlp1(torch.cat([x_j, edge_attr], dim=1))

    def update(self, aggr_out, x):
        comb = self.mlp2(torch.cat([x, aggr_out], dim=1))
        return torch.cat([x[:, :2], comb], dim=1)


def MLP(channels):
    return Seq(*[Seq(Lin(channels[i - 1], channels[i]), ReLU()) for i in range(1, len(channels))])


class IGCNet(torch.nn.Module):
    def __init__(self):
        super(IGCNet, self).__init__()
        self.mlp1 = MLP([4, 32, 32])
        self.mlp2 = Seq(MLP([35, 16]), Seq(Lin(16, 1), Sigmoid()))
        self.conv = IGConv(self.mlp1, self.mlp2)

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        for _ in range(3):
            x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return x


# ================= 3. 完整参数类与初始化 =================

class init_parameters():
    def __init__(self, n_links=50, f_len=1000):
        self.n_links = n_links
        self.field_length = f_len
        self.shortest_directLink_length = 2
        self.longest_directLink_length = 65
        self.shortest_crossLink_length = 1
        self.bandwidth = 5e6
        self.carrier_f = 2.4e9
        self.tx_height = 1.5
        self.rx_height = 1.5
        self.antenna_gain_decibel = 2.5
        self.tx_power_milli_decibel = 40
        self.tx_power = np.power(10, (self.tx_power_milli_decibel - 30) / 10)
        self.noise_density_milli_decibel = -169
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel - 30) / 10)) * self.bandwidth
        self.output_noise_power = self.input_noise_power
        self.SNR_gap_dB = 6
        self.SNR_gap = np.power(10, self.SNR_gap_dB / 10)
        self.setting_str = "{}_links_{}X{}_{}_{}_length".format(
            self.n_links, self.field_length, self.field_length,
            self.shortest_directLink_length, self.longest_directLink_length
        )
        self.cell_length = 5
        self.n_grids = np.round(self.field_length / self.cell_length).astype(int)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_K, train_layouts = 50, 1000
train_config = init_parameters(train_K)
var = train_config.output_noise_power / train_config.tx_power
bw_factor = train_config.bandwidth / 1e6

# 固定生成一次底层数据集
print("正在生成训练数据集 (K=50)...")
layouts, train_dists = wg.generate_layouts(train_config, train_layouts)
train_path_losses = wg.compute_path_losses(train_config, train_dists)
train_channel_losses = helper_functions.add_fast_fading(helper_functions.add_shadowing(train_path_losses))
train_losses_mat = np.multiply(np.eye(train_K), train_channel_losses) + (
            train_path_losses - np.multiply(np.eye(train_K), train_path_losses))

norm_tr_dist, _ = normalize_data(1 / train_dists, 1 / train_dists)
norm_tr_loss, _ = normalize_data(np.sqrt(train_channel_losses), np.sqrt(train_channel_losses))

# ================= 4. 多阈值训练对比 =================

thresholds = [0, 10, 20, 30, 40, 50, 60]
epochs = 50
results_history = {}

for D in thresholds:
    print(f"\n--- 实验启动: Threshold D = {D}m ---")
    # 动态构图生成当前阈值的 DataLoader
    current_data_list = proc_data(train_losses_mat, train_dists, norm_tr_dist, norm_tr_loss, train_K, D)
    loader = DataLoader(current_data_list, batch_size=64, shuffle=True)

    # 独立初始化模型
    model = IGCNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history_sr = []
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)

            # 计算 Sum Rate Loss
            power = torch.reshape(out[:, 2], (-1, train_K, 1))
            abs_H_2 = data.y.permute(0, 2, 1)
            rx_p = torch.mul(abs_H_2, power)
            mask = torch.eye(train_K).to(device)
            # SINR = S / (I + N)
            interference_plus_noise = torch.sum(rx_p * (1 - mask), 1) + var
            sinr = torch.sum(rx_p * mask, 1) / interference_plus_noise
            loss = torch.neg(torch.mean(torch.sum(torch.log2(1 + sinr), 1)))

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        # 换算为 M/bps 并记录
        avg_sr = -(total_loss / train_layouts) * bw_factor
        history_sr.append(avg_sr)

        if epoch % 10 == 0:
            print(f"   D={D}m | Epoch {epoch}/{epochs} | Sum Rate: {avg_sr:.2f} M/bps")

    results_history[D] = history_sr

# ================= 5. 绘图与结果保存 =================

plt.figure(figsize=(10, 6))
# 使用颜色映射区分不同阈值
colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))

for i, D in enumerate(thresholds):
    lbl = f'D = {D}m (Full)' if D == 0 else f'D = {D}m'
    plt.plot(range(1, epochs + 1), results_history[D],
             label=lbl, color=colors[i], linewidth=1.5)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Sum Rates(M/bps)', fontsize=12)

plt.legend(loc='lower right', ncol=2, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# 保存为高质量 EPS 文件
plt.savefig('convergence_comparison.eps', format='eps', dpi=300)
plt.show()

print("\n实验完成。收敛对比图已生成。")