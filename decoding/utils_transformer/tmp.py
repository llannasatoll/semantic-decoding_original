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


# 位置エンコーディング：各時刻に固有の情報を加える
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# Transformerによるシーケンス回帰モデル
class TransformerRegressor(nn.Module):
    def __init__(
        self, input_dim, d_model, nhead, num_layers, output_dim, dropout=0.1, pe_dim=500
    ):
        """
        input_dim  : 入力特徴量数（Rstimの各時刻の次元数）
        d_model    : モデル内部の隠れ層次元数
        nhead      : アテンションヘッドの数
        num_layers : Transformerエンコーダーの層数
        output_dim : 出力次元数（Rrespの各時刻のチャネル数）
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=pe_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        x = self.input_proj(src)  # -> (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)  # 位置エンコーディングの付与
        # x = x.transpose(0, 1)  # Transformerは (seq_len, batch_size, d_model) を要求
        x = self.transformer_encoder(x)  # -> (seq_len, batch_size, d_model)
        # x = x.transpose(0, 1)  # -> (batch_size, seq_len, d_model)
        output = self.output_layer(
            x
        )  # 各時刻ごとに予測 (batch_size, seq_len, output_dim)
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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # モデルのハイパーパラメータ
    d_model = 256
    nhead = 4
    num_layers = 2
    dropout = 0.3
    print("d_model", d_model)
    print("nhead", nhead)
    print("num_layers", num_layers)
    print("dropout", dropout)

    # dist.init_process_group(backend="nccl")
    # local_rank = 0
    # sampler = DistributedSampler(dataset, rank=local_rank)
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     sampler=sampler,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True,
    # )
    model = TransformerRegressor(
        input_dim,
        d_model,
        nhead,
        num_layers,
        output_dim,
        dropout,
        pe_dim=dataset.Rstim.shape[0],
    )
    # model = DDP(model, device_ids=[local_rank])
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)
    criterion = nn.MSELoss()  # 回帰問題なので平均二乗誤差
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5000
    # model.train()
    for epoch in range(num_epochs):
        model.train()
        for stim_seq, resp_seq in dataloader:
            stim_seq = stim_seq.to(device)
            resp_seq = resp_seq.to(device)

            optimizer.zero_grad()
            # stim_seq: (batch_size, T, input_dim)
            # resp_seq: (batch_size, T, output_dim)
            pred = model(stim_seq)  # 予測 (batch_size, T, output_dim)
            loss = criterion(pred, resp_seq)
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        if epoch % 100 == 0:
            eval(model, rstim, resp)

    return model
