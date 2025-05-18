import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np

import config

# import numpy as np
# from transformer import TransformerRegressor


# 複数線形層 (MLP) を使用するモデル
class MLPRegressor(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers=5, dropout_rate=0.1
    ):
        """
        input_dim : 入力特徴量数（Rstimの各時刻の次元数）
        hidden_dim: 隠れ層の次元数
        output_dim: 出力次元数（Rrespの各時刻のチャネル数）
        num_layers: 線形層の総数 (入力層、隠れ層、出力層を含む >= 2)
        dropout_rate: ドロップアウト率
        """
        super().__init__()
        assert num_layers >= 2, "最低でも入力層と出力層の2層が必要です"

        layers = []
        # 入力層
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # 隠れ層 (num_layers - 2 層)
        # num_layersが2の場合、このループは実行されない
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # 出力層
        layers.append(nn.Linear(hidden_dim, output_dim))
        # 回帰タスクなので出力層の後に活性化関数は通常不要

        # 定義した層をSequentialにまとめる
        self.layers = nn.Sequential(*layers)

    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        # 各タイムステップに対して独立にMLPを適用
        # nn.Sequentialはテンソルの最後の次元以外を保持するため、
        # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, output_dim) となる
        output = self.layers(src)
        return output


# RstimとRrespのシーケンス全体をひとまとめのサンプルとして扱うデータセット
class FMRISequenceDataset(Dataset):
    def __init__(self, Rstim, Rresp):
        """
        Rstim: numpy array, shape (T, input_dim)
        Rresp: numpy array, shape (T, output_dim)
        ※ 時間軸 T は両者で一致している前提
        """
        assert (
            Rstim.shape[0] == Rresp.shape[0]
        ), "RstimとRrespの時間軸は一致している必要があります。"
        self.Rstim = Rstim
        self.Rresp = Rresp

    def __len__(self):
        # ここでは1サンプル（シーケンス全体）としていますが、
        # 必要に応じてウィンドウ分割などで複数サンプルにすることも可能です。
        return 1

    def __getitem__(self, idx):
        return torch.tensor(self.Rstim, dtype=torch.float32), torch.tensor(
            self.Rresp, dtype=torch.float32
        )


def compute_correlation(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    cov_xy = np.mean((x - mean_x) * (y - mean_y))

    std_x = np.sqrt(np.mean((x - mean_x) ** 2))
    std_y = np.sqrt(np.mean((y - mean_y) ** 2))

    correlation = cov_xy / (std_x * std_y)
    return correlation


def eval(model, rstim, resp):
    model.eval()
    with torch.no_grad():
        rstim = (
            torch.tensor(rstim, dtype=torch.float32).unsqueeze(0).to(config.EM_DEVICE)
        )
        pred = model(rstim)
        pred = pred[0].cpu().numpy()
    corr = np.array(
        [compute_correlation(resp[:, i], pred[:, i]) for i in range(resp.shape[-1])]
    )
    print(f"mean(fdr(corr)) : {corr.mean()}")


def train_em(dataset, input_dim, output_dim, device, rstim, resp):
    # シーケンス全体をひとまとめにして扱うのでbatch_sizeは1
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # --- モデルのハイパーパラメータ ---
    hidden_dim = 256  # 隠れ層の次元数 (例: Transformerのd_modelに合わせる)
    num_total_layers = 3  # 線形層の総数 (入力層1 + 隠れ層3 + 出力層1 = 5)
    dropout_rate = 0.3  # ドロップアウト率
    learning_rate = 1e-4  # 学習率 (MLPでは少し下げる方が安定する場合がある)
    num_epochs = 5000  # エポック数

    print("Model Type: MLPRegressor")
    print("Hidden Dim:", hidden_dim)
    print("Total Linear Layers:", num_total_layers)
    print("Dropout Rate:", dropout_rate)
    print("Learning Rate:", learning_rate)
    print("Epochs:", num_epochs)
    # --- ここまで ---

    # モデルのインスタンス化 (MLPRegressorを使用)
    model = MLPRegressor(
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=num_total_layers,
        dropout_rate=dropout_rate,
    )
    model.to(device)
    criterion = nn.MSELoss()  # 回帰問題なので平均二乗誤差
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5000
    # model.train()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for stim_seq, resp_seq in dataloader:
            stim_seq = stim_seq.to(device)
            resp_seq = resp_seq.to(device)

            optimizer.zero_grad()
            pred = model(stim_seq)  # 予測 (batch_size, T, output_dim)
            loss = criterion(pred, resp_seq)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # データローダーは1ステップしかないので、epoch_lossがそのままloss
        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        if epoch % 100 == 0:
            eval(model, rstim, resp)

    return model
