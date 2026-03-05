import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import function_wmmse_powercontrol as wf
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
import wireless_networks_generator as wg
from FPLinQ import FP_optimize, FP
import helper_functions
import time

# 假设这些模型定义在其他文件中，确保它们已被正确导入
from GCN_model import GCNNet
from GAT_model import GATNet


# ================= 函数定义修改 =================

def simple_greedy(X, AAA, label):
    """
    修改点：不再使用全局 test_K，改用输入 X 的实际维度
    """
    n = X.shape[0]
    K = X.shape[1]  # 自动获取链路数量
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
    """
    修改点：通过输入数据的形状自动识别 K，不再依赖全局变量
    """
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

    # 归一化测试集
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
    x = np.concatenate((x1, x2, x3), axis=1)
    x = torch.tensor(x, dtype=torch.float)

    dist2 = np.copy(dist)
    mask = np.eye(K)
    diag_dist = np.multiply(mask, dist2)
    dist2 = dist2 + 1000 * diag_dist

    if threshold > 0:
        dist2[dist2 > threshold] = 0

    attr_ind = np.nonzero(dist2)
    edge_attr = norm_dist[attr_ind]
    edge_attr = np.expand_dims(edge_attr, axis=-1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    attr_ind = np.array(attr_ind)
    adj = np.zeros(attr_ind.shape)
    adj[0, :] = attr_ind[1, :]
    adj[1, :] = attr_ind[0, :]
    edge_index = torch.tensor(adj, dtype=torch.long)

    y = torch.tensor(np.expand_dims(loss, axis=0), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=y)
    return data


def proc_data(HH, dists, norm_dists, norm_HH, K, threshold):
    n = HH.shape[0]
    data_list = []
    for i in range(n):
        data = build_graph(HH[i, :, :], dists[i, :, :], norm_dists[i, :, :], norm_HH[i, :, :], K, threshold)
        data_list.append(data)
    return data_list


# ================= 网络结构定义 =================
class IGConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(IGConv, self).__init__(aggr='max', **kwargs)
        self.mlp1 = mlp1
        self.mlp2 = mlp2

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        return self.mlp1(tmp)

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        comb = self.mlp2(tmp)
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
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
        x2 = self.conv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        out = self.conv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        return out


# ================= 训练/测试逻辑 =================
def sr_loss(data, out, K):
    power = out[:, 2]
    power = torch.reshape(power, (-1, K, 1))
    abs_H_2 = data.y.permute(0, 2, 1)
    rx_power = torch.mul(abs_H_2, power)
    mask = torch.eye(K).to(device)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)
    interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + var
    rate = torch.log2(1 + torch.div(valid_rx_power, interference))
    return torch.neg(torch.mean(torch.sum(rate, 1)))


def train(model, loader, optimizer, K, layouts_count):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = sr_loss(data, out, K)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / layouts_count


def test(model, loader, K, layouts_count, config, direct_l, cross_l):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = sr_loss(data, out, K)
            total_loss += loss.item() * data.num_graphs

    power = out[:, 2].reshape(-1, K).detach().cpu().numpy()
    rates = helper_functions.compute_rates(config, power, direct_l, cross_l)
    return total_loss / layouts_count, np.mean(np.sum(rates, axis=1))


# ================= 实验主流程 =================

train_K = 50
test_K = 50
train_layouts = 2000
test_layouts = 500


class init_parameters():
    def __init__(self, n_links=50, f_len=1000):
        # 基础无线网络设置
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

        # --- 补回报错缺失的属性 ---
        self.setting_str = "{}_links_{}X{}_{}_{}_length".format(
            self.n_links, self.field_length, self.field_length,
            self.shortest_directLink_length, self.longest_directLink_length
        )

        # 2D 栅格设置（如果有函数用到）
        self.cell_length = 5
        self.n_grids = np.round(self.field_length / self.cell_length).astype(int)

train_config = init_parameters(train_K)
var = train_config.output_noise_power / train_config.tx_power

# 数据生成
layouts, train_dists = wg.generate_layouts(train_config, train_layouts)
train_path_losses = wg.compute_path_losses(train_config, train_dists)
train_channel_losses = helper_functions.add_fast_fading(helper_functions.add_shadowing(train_path_losses))
train_losses = np.multiply(np.eye(train_K), train_channel_losses) + (
            train_path_losses - np.multiply(np.eye(train_K), train_path_losses))

test_config = init_parameters(test_K)
layouts, test_dists = wg.generate_layouts(test_config, test_layouts)
test_path_losses = wg.compute_path_losses(test_config, test_dists)
test_channel_losses = helper_functions.add_fast_fading(helper_functions.add_shadowing(test_path_losses))

norm_tr_dist, norm_te_dist = normalize_data(1 / train_dists, 1 / test_dists)
norm_tr_loss, norm_te_loss = normalize_data(np.sqrt(train_channel_losses), np.sqrt(test_channel_losses))

# 数据集 Loader
train_loader_full = DataLoader(proc_data(train_losses, train_dists, norm_tr_dist, norm_tr_loss, train_K, 0),
                               batch_size=64, shuffle=True)
train_loader_d = DataLoader(proc_data(train_losses, train_dists, norm_tr_dist, norm_tr_loss, train_K, 50),
                            batch_size=64, shuffle=True)
test_loader_full = DataLoader(proc_data(test_channel_losses, test_dists, norm_te_dist, norm_te_loss, test_K, 0),
                              batch_size=test_layouts)
test_loader_d = DataLoader(proc_data(test_channel_losses, test_dists, norm_te_dist, norm_te_loss, test_K, 50),
                           batch_size=test_layouts)

# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_full = IGCNet().to(device)
model_d = IGCNet().to(device)
opt_full = torch.optim.Adam(model_full.parameters(), lr=0.001)
opt_d = torch.optim.Adam(model_d.parameters(), lr=0.001)

# 训练记录
h_full, h_d = [], []
for epoch in range(1, 51):
    l_f = train(model_full, train_loader_full, opt_full, train_K, train_layouts)
    l_d = train(model_d, train_loader_d, opt_d, train_K, train_layouts)
    h_full.append(l_f)
    h_d.append(l_d)
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss Full: {l_f:.4f} | Loss D: {l_d:.4f}")

# 绘图
h_full_arr = np.array(h_full)
h_d_arr = np.array(h_d)


inv_h_full = -h_full_arr
inv_h_d = -h_d_arr

plt.plot(range(1, len(h_full) + 1), inv_h_full, label='Sum rates(Full, Th=0)', color='b', linestyle='-')
plt.plot(range(1, len(h_d) + 1), inv_h_d, label='Sum rates(D, Th=50m)', color='r', linestyle='--')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Sum Rates(M/bps)', fontsize=12)

plt.legend();
plt.grid();
plt.savefig('convergence.eps');
plt.show()

# ================= 泛化测试修复 =================
density = 1000 ** 2 / 50
for K_gen in [20, 50, 100]:
    config_gen = init_parameters(K_gen, int(np.sqrt(density * K_gen)))
    _, dists_gen = wg.generate_layouts(config_gen, 50)
    ch_loss_gen = helper_functions.add_fast_fading(
        helper_functions.add_shadowing(wg.compute_path_losses(config_gen, dists_gen)))

    dir_gen = helper_functions.get_directLink_channel_losses(ch_loss_gen)
    cross_gen = helper_functions.get_crossLink_channel_losses(ch_loss_gen)

    # 基准计算
    Y_fp = FP_optimize(config_gen, ch_loss_gen, np.ones([50, K_gen]))
    bl_Y = simple_greedy(ch_loss_gen, np.ones([50, K_gen]), Y_fp)
    # 这里不会再报错，因为 simple_greedy 现在会自动识别 K_gen
    rates_bl = helper_functions.compute_rates(config_gen, bl_Y, dir_gen, cross_gen)
    print(f"K={K_gen} | Baseline SR: {np.mean(np.sum(rates_bl, axis=1)):.4f}")

    # GNN 测试
    _, n_dist_gen = normalize_data(1 / train_dists, 1 / dists_gen)
    _, n_loss_gen = normalize_data(np.sqrt(train_channel_losses), np.sqrt(ch_loss_gen))

    ldr_f = DataLoader(proc_data(ch_loss_gen, dists_gen, n_dist_gen, n_loss_gen, K_gen, 0), batch_size=50)
    _, sr_f = test(model_full, ldr_f, K_gen, 50, config_gen, dir_gen, cross_gen)
    print(f"K={K_gen} | IGCNet Full SR: {sr_f:.4f}")