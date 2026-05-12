"""
昆虫分类神经网络 — 选项2
支持数据集1(干净)和数据集2(含噪)的训练与评估
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import os
from matplotlib.colors import ListedColormap

# ============================================================
# 数据加载
# ============================================================
def load_data(filepath):
    """加载数据文件，返回 (features, labels)"""
    data = np.loadtxt(filepath)
    X = data[:, :2].astype(np.float32)
    y = data[:, 2].astype(np.int64)
    return X, y


def load_dataset(data_dir, dataset_id):
    """
    加载指定数据集。
    dataset_id=1 → insects-training/test.txt
    dataset_id=2 → insects-2-training/test.txt
    """
    if dataset_id == 1:
        train_file = os.path.join(data_dir, "insects-training.txt")
        test_file = os.path.join(data_dir, "insects-testing.txt")
    else:
        train_file = os.path.join(data_dir, "insects-2-training.txt")
        test_file = os.path.join(data_dir, "insects-2-testing.txt")

    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
    return X_train, y_train, X_test, y_test


# ============================================================
# 神经网络模型
# ============================================================
class InsectClassifier(nn.Module):
    def __init__(self, hidden_layers, activation='relu', dropout=0.0):
        """
        hidden_layers: list of int, 每个元素为一个隐藏层的神经元数
        activation: 'relu', 'tanh', 'sigmoid', 'leaky_relu'
        dropout: dropout 概率
        """
        super().__init__()

        act_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.1),
        }[activation]

        layers = []
        in_features = 2
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = h
        layers.append(nn.Linear(in_features, 3))  # 3类输出

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# 训练
# ============================================================
def train_model(model, train_loader, epochs, lr, weight_decay, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

    return train_losses


# ============================================================
# 评估
# ============================================================
@torch.no_grad()
def evaluate(model, X, y, device):
    """返回 accuracy, predictions"""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    outputs = model(X_tensor)
    preds = outputs.argmax(dim=1).cpu().numpy()
    acc = (preds == y).mean()
    return acc, preds


# ============================================================
# 可视化
# ============================================================
def plot_decision_boundary(model, X, y, scaler, title, save_path, device):
    """绘制二维决策边界"""
    model.eval()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_scaled = scaler.transform(grid)

    with torch.no_grad():
        grid_tensor = torch.tensor(grid_scaled, dtype=torch.float32).to(device)
        Z = model(grid_tensor).argmax(dim=1).cpu().numpy()
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ['#FF0000', '#00AA00', '#0000FF']
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    for c in range(3):
        mask = y == c
        plt.scatter(X[mask, 0], X[mask, 1], c=cmap_bold[c], label=f'Class {c}',
                    edgecolors='k', s=30)
    plt.xlabel('Body Length')
    plt.ylabel('Wing Length')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_loss_curve(losses, title, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_data_distribution(X, y, title, save_path):
    """绘制数据分布散点图"""
    plt.figure(figsize=(8, 6))
    colors = ['#FF0000', '#00AA00', '#0000FF']
    for c in range(3):
        mask = y == c
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[c], label=f'Class {c}',
                    edgecolors='k', s=30)
    plt.xlabel('Body Length')
    plt.ylabel('Wing Length')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Insect Classification with Neural Network')
    parser.add_argument('--dataset', type=int, default=1, choices=[1, 2],
                        help='Dataset ID (1=clean, 2=noisy)')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[32, 16],
                        help='Hidden layer sizes (e.g. --hidden-layers 64 32)')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'tanh', 'sigmoid', 'leaky_relu'])
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory for figures and results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = os.path.join(os.path.dirname(__file__), 'insects')

    # 加载数据
    X_train, y_train, X_test, y_test = load_dataset(data_dir, args.dataset)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建 DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 构建模型
    model = InsectClassifier(
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        dropout=args.dropout
    ).to(device)

    print(f"\n{'='*60}")
    print(f"Dataset {args.dataset} | Architecture: {[2] + args.hidden_layers + [3]}")
    print(f"Activation: {args.activation} | Dropout: {args.dropout}")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Weight Decay: {args.weight_decay}")
    print(f"{'='*60}")

    # 训练
    train_losses = train_model(model, train_loader, args.epochs, args.lr,
                               args.weight_decay, device)

    # 评估
    train_acc, _ = evaluate(model, X_train_scaled, y_train, device)

    # 测试集分段评估
    X_test_first60 = X_test_scaled[:60]
    y_test_first60 = y_test[:60]
    X_test_last150 = X_test_scaled[60:]
    y_test_last150 = y_test[60:]

    acc_first60, preds_first60 = evaluate(model, X_test_first60, y_test_first60, device)
    acc_last150, preds_last150 = evaluate(model, X_test_last150, y_test_last150, device)
    acc_all, preds_all = evaluate(model, X_test_scaled, y_test, device)

    # 打印结果
    print(f"\n{'='*60}")
    print(f"RESULTS — Dataset {args.dataset}")
    print(f"{'='*60}")
    print(f"Train Accuracy:        {train_acc:.4f}")
    print(f"Test Accuracy (all):   {acc_all:.4f}")
    print(f"Test Accuracy (前60):  {acc_first60:.4f}")
    print(f"Test Accuracy (后150): {acc_last150:.4f}")
    print(f"{'='*60}")

    # 混淆矩阵 (全部测试集)
    cm = confusion_matrix(y_test, preds_all)
    print("\nConfusion Matrix (all test):")
    print(cm)
    print("\nClassification Report (all test):")
    print(classification_report(y_test, preds_all, target_names=['Class 0', 'Class 1', 'Class 2']))

    # 保存可视化
    prefix = f"dataset{args.dataset}"

    plot_loss_curve(train_losses,
                    f'Training Loss — Dataset {args.dataset}',
                    os.path.join(args.output_dir, f'{prefix}_loss.png'))

    # 决策边界 (测试集)
    plot_decision_boundary(model, X_test, y_test, scaler,
                           f'Decision Boundary — Dataset {args.dataset} (Test)',
                           os.path.join(args.output_dir, f'{prefix}_decision_test.png'),
                           device)

    # 决策边界 (训练集)
    plot_decision_boundary(model, X_train, y_train, scaler,
                           f'Decision Boundary — Dataset {args.dataset} (Train)',
                           os.path.join(args.output_dir, f'{prefix}_decision_train.png'),
                           device)

    # 测试数据分布
    plot_data_distribution(X_test, y_test,
                           f'Test Data Distribution — Dataset {args.dataset}',
                           os.path.join(args.output_dir, f'{prefix}_test_data.png'))

    # 保存数值结果
    result_file = os.path.join(args.output_dir, f'{prefix}_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Architecture: {[2] + args.hidden_layers + [3]}\n")
        f.write(f"Activation: {args.activation}\n")
        f.write(f"Dropout: {args.dropout}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"\nTrain Accuracy: {train_acc:.4f}\n")
        f.write(f"Test Accuracy (all): {acc_all:.4f}\n")
        f.write(f"Test Accuracy (前60): {acc_first60:.4f}\n")
        f.write(f"Test Accuracy (后150): {acc_last150:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")

    print(f"\nOutput saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
