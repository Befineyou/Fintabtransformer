import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from new_model import FTTransformer  # 假设FT-Transformer库已安装


# FT-Transformer配置
class FTTransformerConfig:
    def __init__(self):
        self.categories = []  # 若无类别特征，设为空列表
        self.num_continuous = 96  # 连续特征数 x1到x96
        self.dim = 32  # 嵌入维度
        self.dim_out = 1  # 输出维度为1，回归任务
        self.depth = 6  # transformer深度
        self.heads = 8  # multi-head attention的头数
        self.attn_dropout = 0.1  # attention dropout
        self.ff_dropout = 0.1  # feedforward dropout
        self.mlp_hidden_mults = (4, 2)  # MLP层
        self.mlp_act = nn.ReLU()  # 激活函数


# 自定义Dataset类，用于加载训练集和测试集
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.features = data.loc[:, "x1":"x96"].values.astype(np.float32)
        self.labels = data["CS"].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


# 定义FT-Transformer模型
class FTTransformerModel(nn.Module):
    def __init__(self, config):
        super(FTTransformerModel, self).__init__()
        self.model = FTTransformer(
            num_continuous=config.num_continuous,
            dim=config.dim,
            dim_out=config.dim_out,
            depth=config.depth,
            heads=config.heads,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.ff_dropout,
        )

    def forward(self, x_numer):  # 修改为x_numer
        return self.model(x_numer=x_numer)  # 传递x_numer


# 滚动学习率调整
def adjust_learning_rate(optimizer, epoch):
    if epoch < 15:
        lr = 0.0001
    else:
        lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 训练函数
def train_model(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        features, labels = batch
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features).squeeze()  # 删除 `x_numer=`

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 实时输出训练信息，避免因未加 `.item()` 出现的错误
        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{total_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)


# 评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions, true_values = [], []
    with torch.no_grad():
        for batch in test_loader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(labels.cpu().numpy())

    # 计算RMSE, R², DM指标
    rmse = np.sqrt(np.mean((np.array(predictions) - np.array(true_values)) ** 2))
    r2 = 1 - (np.sum((np.array(true_values) - np.array(predictions)) ** 2) /
              np.sum((np.array(true_values) - np.mean(true_values)) ** 2))
    dm_stat = np.mean((np.array(predictions) - np.array(true_values)) ** 2)

    return rmse, r2, dm_stat, predictions, true_values


# 绘图函数
def plot_results(true_values, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(true_values)), true_values, color='blue', label='True Values')
    plt.scatter(range(len(predictions)), predictions, color='red', label='Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.title('True Values vs Predicted Values')
    plt.legend()
    plt.savefig('predictions_vs_true.png', bbox_inches='tight', dpi=300)
    plt.show()


# 主函数
def main(train_file, test_file, total_epochs=30, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = FTTransformerConfig()

    # 准备数据
    train_dataset = CSVDataset(train_file)
    test_dataset = CSVDataset(test_file)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型与优化器
    model = FTTransformerModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    # 训练模型
    for epoch in range(total_epochs):
        # 动态调整学习率
        adjust_learning_rate(optimizer, epoch)

        train_loss = train_model(model, train_loader, optimizer, criterion, device, epoch, total_epochs)
        print(f"Epoch {epoch + 1}/{total_epochs} - Training Loss: {train_loss:.4f}")

        # 每个 epoch 后评估一次模型
        rmse, r2, dm_stat, predictions, true_values = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{total_epochs} - Evaluation - RMSE: {rmse:.4f}, R²: {r2:.4f}, DM: {dm_stat:.4f}")

    # 保存和可视化
    plot_results(true_values, predictions)
    results_df = pd.DataFrame({'True Values': true_values, 'Predicted Values': predictions})
    results_df.to_csv('predictions_vs_true.csv', index=False)


if __name__ == "__main__":
    train_file = "train_data.csv"  # 训练集文件路径
    test_file = "test_data.csv"  # 测试集文件路径
    main(train_file, test_file)
