import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# === 1. 读取所有 xlsx 文件 ===
def load_data(folder):
    images, labels = [], []
    for f in os.listdir(folder):
        if f.endswith('.xlsx'):
            label = float(os.path.splitext(f)[0])      # 文件名即标签（改成 float）
            arr = pd.read_excel(os.path.join(folder, f), header=None).to_numpy().astype(np.float32)
            if arr.shape != (7, 7):
                raise ValueError(f'{f} 形状不是 7×7')
            images.append(arr)
            labels.append(label)
    data = {'images': np.stack(images), 'labels': np.array(labels)}
    np.save('data.npy', data)
    print(f'共加载 {len(labels)} 个样本，已保存为 data.npy')

# === 2. 数据集定义 ===
class PixelDataset(Dataset):
    def __init__(self, path):
        d = np.load(path, allow_pickle=True).item()
        self.x = d['images'].astype(np.float32)
        self.y = d['labels'].astype(np.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return torch.tensor(self.x[i]).unsqueeze(0), torch.tensor(self.y[i])

# === 3. 模型定义 ===
class DigitRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*3*3, 64), nn.ReLU(),
            nn.Linear(64, 1)  # 🔹 输出一个连续数
        )

    def forward(self, x):
        # 自动检测输入数值范围并放大
        with torch.no_grad():
            abs_mean = x.abs().mean()
            if abs_mean < 1e-5:
                scale = 1e10
                x = x * scale
                # print(f"⚠️ 输入数值过小 (均值={abs_mean:.2e})，已自动放大 {scale} 倍。")
        return self.net(x)

# === 4. 训练函数 ===

def printt():
    ds = PixelDataset("data.npy")
    dl = DataLoader(ds, batch_size=10, shuffle=True)
    for x, y in dl:
        print("x:", x.shape, x)
        print("y:", y.shape, y)
        break

def evaluate(model, dataloader, device, tol=0.00005):
    model.eval()

    total_abs_err = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            out = model(x).squeeze(1)
            abs_err = torch.abs(out - y)

            total_abs_err += abs_err.sum().item()
            correct += (abs_err < tol).sum().item()
            total += y.numel()

    mae = total_abs_err / total
    acc = correct / total

    return mae, acc

def train(data_path, epochs=100, lr=1e-4,
          save_path='digit_regressor.pth',
          log_path='train_log.csv',
          tol=0.5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    ds = PixelDataset(data_path)
    train_loader = DataLoader(ds, batch_size=8, shuffle=True, pin_memory=True)
    eval_loader  = DataLoader(ds, batch_size=16, shuffle=False, pin_memory=True)

    model = DigitRegressor().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = []

    for ep in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad()
            out = model(x).squeeze(1)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ===== 全集评估（CUDA）=====
        eval_mae, eval_acc = evaluate(model, eval_loader, device, tol)

        history.append([ep + 1, avg_loss, eval_mae, eval_acc])

        print(
            f"Epoch {ep+1}/{epochs}  "
            f"train_loss={avg_loss:.6f}  "
            f"eval_mae={eval_mae:.6f}  "
            f"eval_acc(|err|<{tol})={eval_acc:.4f}"
        )

    df = pd.DataFrame(
        history,
        columns=['epoch', 'train_loss', 'eval_mae', 'eval_accuracy']
    )
    df.to_csv(log_path, index=False)

    torch.save(model.state_dict(), save_path)
    print(f"训练完成，权重已保存为 {save_path}")
    print(f"训练日志已保存为 {log_path}")

# === 主程序 ===
if __name__ == '__main__':
    folder = './data'
    load_data(folder)
    train('data.npy', epochs=600)
    # printt()