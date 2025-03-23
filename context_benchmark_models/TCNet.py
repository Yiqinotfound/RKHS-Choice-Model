import torch
import torch.nn as nn
import numpy as np
import sys, os
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils.benchmark_utils import (
    xavier_initialize_weights,
    warmup_lr_scheduler,
    NewGELU,
)

from utils.creterias_and_loss_utils import safe_log


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim: int, pf_dim: int, device: torch.device):
        super().__init__()

        self.device = device
        self.fc_1 = nn.Linear(hidden_dim, pf_dim, device=self.device)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim, device=self.device)
        self.act = NewGELU()

    def forward(self, x: torch.Tensor):

        # x = [batch size, seq len, hid dim]
        x = self.fc_2(self.act(self.fc_1(x)))
        # x = [batch size, seq len, hid dim]

        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, device: torch.device):
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.device = device

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.fc_q = nn.Linear(hidden_dim, hidden_dim, device=self.device)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim, device=self.device)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim, device=self.device)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim, device=self.device)

        self.dropout = nn.Dropout(0.1)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(self.device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):

        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q: torch.Tensor = self.fc_q(query)
        K: torch.Tensor = self.fc_k(key)
        V: torch.Tensor = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hidden_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_dim: int, num_heads: int, pf_dim: int, device: torch.device
    ):
        super().__init__()
        self.device = device
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim, device=device)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim, device=device)
        self.self_attention = MultiHeadAttentionLayer(
            hidden_dim=hidden_dim, num_heads=num_heads, device=device
        )
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hidden_dim=hidden_dim, pf_dim=pf_dim, device=device
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):

        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class DecoderLayer(nn.Module):
    def __init__(
        self, hidden_dim: int, num_heads: int, pf_dim: int, device: torch.device
    ):
        super().__init__()
        self.device = device
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim, device=self.device)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim, device=self.device)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim, device=self.device)
        self.self_attention = MultiHeadAttentionLayer(
            hidden_dim=hidden_dim, num_heads=num_heads, device=device
        )
        self.encoder_attention = MultiHeadAttentionLayer(
            hidden_dim=hidden_dim, num_heads=num_heads, device=device
        )

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hidden_dim=hidden_dim, pf_dim=pf_dim, device=self.device
        )
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        trg: torch.Tensor,
        enc_src: torch.Tensor,
        trg_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ):

        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention


class TransformerChoiceNet(nn.Module):
    def __init__(self, model_args: dict, device: torch.device):

        super().__init__()
        torch.manual_seed(model_args["SEED"])
        self.model_args = model_args
        self.device = device

        self.input_dim = model_args["input_dim"]
        self.hidden_dim = model_args["hidden_dim"]
        self.num_heads = model_args["num_heads"]
        print("input_dim:",self.input_dim)
        self.ln_0_src = nn.LayerNorm(self.input_dim, device=self.device)
        self.ln_0_trg = nn.LayerNorm(self.input_dim, device=self.device)
        self.src_emb = nn.Linear(self.input_dim, self.hidden_dim, device=self.device)
        self.trg_emb = nn.Linear(self.input_dim, self.hidden_dim, device=self.device)

        self.encoder = EncoderLayer(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            pf_dim=self.hidden_dim,
            device=self.device,
        )
        self.decoder = DecoderLayer(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            pf_dim=self.hidden_dim,
            device=self.device,
        )
        self.prod_fea = model_args["prod_fea"]
        self.emb_need = model_args["emb_need"]
        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim, device=self.device)
        self.dropout = nn.Dropout(0.1)
        self.act = NewGELU()
        self.fc2 = nn.Linear(self.hidden_dim, 1, device=device)

        self.multi_purchase = model_args["multi_purchase"]

    def make_src_mask(self, src: torch.Tensor):

        # src = [batch size, src len]
        if self.prod_fea == 0:
            # Assuming padding tokens are represented with 0
            attention_mask = (
                (src.sum(dim=-1) == 0).unsqueeze(1).unsqueeze(1)
            )  # Shape: [B, 1, 1, T]
        else:
            attention_mask = (
                (src[:, :, : self.prod_fea].sum(dim=-1) == 0).unsqueeze(1).unsqueeze(1)
            )
        # src_mask = [batch size, 1, 1, src len]

        return attention_mask

    def forward(self, src: torch.Tensor):
        trg = src
        if self.prod_fea == 0:
            mask = trg.sum(dim=-1) != 0
        else:
            mask = trg[:, :, : self.prod_fea].sum(dim=-1) != 0
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_src_mask(trg)
        # src_mask = [batch size, 1, 1, src len]
        if self.emb_need:
            embs_src = self.src_emb(self.ln_0_src(src))
            embs_trg = self.trg_emb(self.ln_0_trg(trg))
        else:
            embs_src = src
            embs_trg = trg

        enc_src = self.encoder(embs_src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, _ = self.decoder(embs_trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, hid dim]

        logits = self.fc2(output).squeeze(-1)
        # logits_ = self.act(logits_)
        # logits = self.fc2(output+self.dropout(logits_)).squeeze(-1)
        # logits = [batch size, trg len]

        if self.multi_purchase:
            probs = torch.exp(logits) / (1 + torch.exp(logits)) * mask.float()
            probs = probs.squeeze(-1)
            return probs
        else:
            prob = torch.exp(logits) * mask.float()
            prob = prob / prob.sum(dim=-1).unsqueeze(-1)  # [batch size, trg len]
            return prob

    def compute_prob_batch(
        self,
        X_full: torch.Tensor,
        batch_size: int = 1024,
        device: torch.device = None,
    ):
        with torch.no_grad():
            all_probs = []
            for i in range(0, len(X_full), batch_size):
                X_batch = X_full[i : i + batch_size].to(device)
                probs_batch = self.forward(src=X_batch.to(torch.float32))
                all_probs.append(probs_batch.cpu())

        return torch.cat(all_probs, dim=0).to(device)

    def compute_acc(self, P: torch.Tensor, Y: torch.Tensor):
        P_max = torch.argmax(P, dim=1)
        Y_max = torch.argmax(Y, dim=1)
        return torch.sum(P_max == Y_max).item() / Y.size(0)

    def cross_entropy(self, P: torch.Tensor, Y: torch.Tensor):
        return -torch.sum(Y * safe_log(P)) / Y.size(0)

    def fit(
        self,
        dataset_train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dataset_val: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        dataset_test: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        device: torch.device = "cpu",
    ):

        self.train()

        # record time
        t1 = time.time()
        self.apply(xavier_initialize_weights)

        self.X_train, self.y_train, self.cardinality_train = dataset_train
        self.X_test, self.y_test, self.cardinality_test = dataset_test
        self.X_val, self.y_val, self.cardinality_val = dataset_val
        self.max_items = self.X_train.shape[1]
        

        # smoothed labels
        self.y_train_smooth = (
            1 - self.model_args["smoothing"]
        ) * self.y_train + self.model_args["smoothing"] / self.max_items
        self.y_val_smooth = (
            1 - self.model_args["smoothing"]
        ) * self.y_val + self.model_args["smoothing"] / self.max_items
        self.y_test_smooth = (
            1 - self.model_args["smoothing"]
        ) * self.y_test + self.model_args["smoothing"] / self.max_items

        # train dataset and train loader
        self.train_dataset = TensorDataset(
            self.X_train, self.y_train, self.cardinality_train
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.model_args["batch_size"], shuffle=True
        )

        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.model_args["learning_rate"],
            weight_decay=self.model_args["weight_decay"],
        )


        scheduler = StepLR(
            optimizer=optimizer,
            step_size=self.model_args["step_size"],
            gamma=self.model_args["weight_decay"],
        )

        self.best_val_loss = float("inf")

        with tqdm(
            total=self.model_args["max_epochs"], desc="Training", unit="epoch"
        ) as pbar:
            for epoch in range(self.model_args["max_epochs"]):
                total_train_loss = 0.0
                for X_batch, y_batch, cardinality_batch in self.train_loader:

                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    optimizer.zero_grad()
                    prob_train_batch = self.forward(src=X_batch)
                    nll_train_batch = self.cross_entropy(P=prob_train_batch, Y=y_batch)
                    nll_train_batch.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1)
                    optimizer.step()
                    total_train_loss += nll_train_batch.item()

                # evaluate
                self.evaluate(device)
                if self.nll_val < self.best_val_loss:
                    self.best_val_loss = self.nll_val
                    self.best_parameters = {
                        k: v.clone() for k, v in self.state_dict().items()
                    }

                scheduler.step()
                pbar.set_postfix(
                    {
                        "Train Loss": self.nll_train,
                        "Val Loss": self.nll_val,
                        "Test Loss": self.nll_test,
                    }
                )
                pbar.update(1)
                self.train()
        self.train_time = time.time() - t1
        self.load_state_dict(self.best_parameters)

    def evaluate(self, device):
        with torch.no_grad():

            P_train = self.compute_prob_batch(X_full=self.X_train, device=device)
            self.nll_train = self.cross_entropy(
                P=P_train, Y=self.y_train_smooth.to(device)
            ).item()
            self.acc_train = self.compute_acc(P=P_train, Y=self.y_train.to(device))

            P_val = self.compute_prob_batch(X_full=self.X_val, device=device)
            self.nll_val = self.cross_entropy(
                P=P_val, Y=self.y_val_smooth.to(device)
            ).item()
            self.acc_val = self.compute_acc(P=P_val, Y=self.y_val.to(device))

            P_test = self.compute_prob_batch(X_full=self.X_test, device=device)
            self.nll_test = self.cross_entropy(
                P=P_test, Y=self.y_test_smooth.to(device)
            ).item()
            self.acc_test = self.compute_acc(P=P_test, Y=self.y_test.to(device))


# model_args = {
#     "input_dim": 128,
#     "num_heads": 8,
#     "hidden_dim": 512,
#     "prod_fea": 0,
#     "emb_need": True,
#     "SEED": 42,
#     "multi_purchase": False,
# }
# model = TransformerChoiceNet(model_args=model_args, device="cpu")

# # 假设输入
# src = torch.randn(32, 20, 128)  # [batch=32, src_len=100, feat=128]

# src[:, 10:, :] = 0
# # trg = torch.randn(32, 20, 128)   # [batch=32, trg_len=20, feat=128]

# # 前向计算
# prob = model(src)  # [32,20] 每个目标元素的概率

# print(prob[1])
# print(prob[1].shape)
# print(prob[1].sum())
