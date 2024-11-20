import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
def load_dow_price_data(data_addr, adj_addr):
    data = pd.read_csv(data_addr).values
    adj = pd.read_csv(adj_addr, header=None).values
    # data = normalize(data, axis=0)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, adj

# 计算归一化邻接矩阵
def normalized_adj(adj):
    adj = adj + np.eye(adj.shape[0])  # 加上单位矩阵
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_adj = np.dot(np.dot(adj, d_mat_inv_sqrt), d_mat_inv_sqrt)
    return normalized_adj

# 数据预处理
def preprocess_data(data, labels, time_len, train_rate, seq_len, pre_len):
    X, Y, pre_Y = [], [], []
    for i in range(time_len - seq_len - pre_len):
        X.append(data[i:i + seq_len, :])
        Y.append(labels[i + seq_len:i + seq_len + pre_len])
        pre_Y.append(labels[(i + seq_len - 1):(i + seq_len + pre_len - 1)])

    # 划分训练集和测试集
    train_size = int(train_rate * len(X))
    X_train = np.array(X[:train_size])
    Y_train = np.array(Y[:train_size])
    X_test = np.array(X[train_size:])
    Y_test = np.array(Y[train_size:])
    # pre_Y_test = labels[train_size + seq_len:train_size + seq_len + len(X_test)]
    pre_Y_test = np.array(pre_Y[train_size:])

    return X_train, Y_train, X_test, Y_test, pre_Y_test


# GcGRUCell 模型定义
class GcGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, adj, device):
        super(GcGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.adj = torch.tensor(adj, dtype=torch.float32).to(device)  # 确保邻接矩阵也在正确的设备上

        # 定义 GRU 的权重和偏置
        self.Wz = nn.Linear(input_dim, hidden_dim)
        self.Wr = nn.Linear(input_dim, hidden_dim)
        self.Wh = nn.Linear(input_dim, hidden_dim)
        self.Uz = nn.Linear(hidden_dim, hidden_dim)
        self.Ur = nn.Linear(hidden_dim, hidden_dim)
        self.Uh = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h):
        z = torch.sigmoid(self.Wz(x) + self.Uz(h))
        r = torch.sigmoid(self.Wr(x) + self.Ur(h))
        h_tilde = torch.tanh(self.Wh(x) + self.Uh(r * h))
        h_new = (1 - z) * h + z * h_tilde
        return h_new

# 模型
class GcGRUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, adj, device):
        super(GcGRUNetwork, self).__init__()
        self.device = device
        self.cell = GcGRUCell(input_dim, hidden_dim, adj, device)  # 将设备传递给 cell
        self.dense = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.cell.hidden_dim).to(self.device)  # 确保隐藏状态在正确的设备上
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        output = self.dense(h)
        return output


# 训练与评估
def train_and_evaluate(data_addr, adj_addr):
    # 加载数据
    data, adj = load_dow_price_data(data_addr, adj_addr)

    # 计算归一化邻接矩阵
    adj = normalized_adj(adj)

    labels = data[:, 5]  # 预测第几列
    time_len = data.shape[0]
    train_rate = 0.8
    seq_len = 12
    pre_len = 1

    # 数据预处理
    X_train, y_train, X_test, y_test, pre_y_test = preprocess_data(data, labels, time_len, train_rate, seq_len, pre_len)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)  # 确保数据在正确的设备上
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    pre_y_test = torch.tensor(pre_y_test, dtype=torch.float32).to(device)

    # 初始化模型
    input_dim = X_train.shape[2]
    hidden_dim = 64
    model = GcGRUNetwork(input_dim, hidden_dim, adj, device).to(device)  # 确保模型在正确的设备上
    # model = GcLSTNetwork(input_dim, hidden_dim, adj, device).to(device)

    # 损失函数和优化器
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 训练
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        output = model(X_train)
        loss = criterion(output.squeeze(), y_train.squeeze())
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 测试
    model.eval()
    with torch.no_grad():
        result = model(X_test).squeeze()

    # 评估指标
    result = result.reshape(-1, 1)
    # print(y_test.shape)
    # print(result.shape)
    rmse = sqrt(mean_squared_error(y_test.cpu(), result.cpu()))  # 确保结果在 CPU 上进行计算
    print(f'RMSE: {rmse:.4f}')
    r2 = r2_score(y_test.cpu(), result.cpu())
    print(f'R2 Score: {r2:.4f}')
    mae = mean_absolute_error(y_test.cpu(), result.cpu())
    print(f'MAE: {mae:.4f}')
    # print(len(get_trend(pre_y_test.cpu(), result.cpu())))
    accuracy = accuracy_score(get_trend(pre_y_test.cpu(), y_test.cpu()), get_trend(pre_y_test.cpu(), result.cpu()))
    print(f'Accuracy: {accuracy:.4f}')

def get_trend(pre, cur):
    return (cur - pre > 0).to(torch.int).view(-1)  # 使用 .view(-1) 来确保输出是 1D 张量  # 使用 PyTorch 的 .to(torch.int) 来转换类型

# 运行
data_addr = './data/data/dow/dow_1day_price.csv'
adj_addr = './data/adj/dow/1day/dow_1day_090_01_corr.csv'
train_and_evaluate(data_addr, adj_addr)
