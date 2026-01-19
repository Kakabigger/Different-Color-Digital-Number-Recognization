import os
import torch
import pandas as pd
import numpy as np
from torch import nn

import os
import numpy as np
import pandas as pd
import torch

folder = "./data"   # 当前目录
results = []        # 用于保存 Ground Truth / Predict

# === 1. 模型结构（和训练时保持一致） ===
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

# === 2. 加载模型 ===
model = DigitRegressor()
model.load_state_dict(torch.load("digit_regressor.pth", map_location="cpu"))
model.eval()

for f in os.listdir(folder):
    if f.endswith(".xlsx"):
        label = float(os.path.splitext(f)[0])  # 文件名作为真实标签（数值型更规范）
        data = pd.read_excel(os.path.join(folder, f), header=None)\
                 .to_numpy().astype(np.float32)

        if data.shape != (7, 7):
            print(f"⚠️ {f} 形状错误：{data.shape}")
            continue

        x = torch.tensor(data).unsqueeze(0).unsqueeze(0)  # (1,1,7,7)
        with torch.no_grad():
            pred = model(x).item()

        results.append({
            "Ground Truth": label,
            "Predict": pred
        })

        print(f"文件: {f} | Ground Truth: {label} | Predict: {pred:.3f}")

# 保存为 CSV
df = pd.DataFrame(results)
df.to_csv("prediction_results.csv", index=False)

