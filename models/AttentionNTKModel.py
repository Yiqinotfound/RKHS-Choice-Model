import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.creterias_and_loss_utils import (
    safe_log,
    compute_mask_from_card,
)
from utils.model_utils import report_memory
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class AttentionNTKChoiceModel(nn.Module):
    def __init__(self, d, d0, d2, model_config: dict):
        super().__init__()
        self.H = model_config["H"]
        self.d0 = d0
        self.activation = model_config["activation"]
        self.sigma = model_config["sigma"]
        self.half = model_config["half"]

        self.model_config = model_config

        self.dtype = torch.float16 if self.half else torch.float32

        self.Theta1 = nn.Linear(d, self.H * d0, bias=False, dtype=self.dtype)
        self.Theta2 = nn.Linear(d, self.H * d0, bias=False, dtype=self.dtype)
        self.Theta3 = nn.Linear(d, self.H * d2, bias=False, dtype=self.dtype)

        self.w = nn.Parameter(torch.randn(self.H, d2, dtype=self.dtype))

        self._init_weights(sigma=self.sigma)

    def _init_weights(self, sigma: float = 1):
        def init_linear(layer):
            weight_tensor = torch.empty_like(layer.weight, dtype=self.dtype)
            nn.init.normal_(weight_tensor, mean=0, std=sigma)
            layer.weight = nn.Parameter(weight_tensor)

        init_linear(self.Theta1)
        init_linear(self.Theta2)
        init_linear(self.Theta3)
        nn.init.normal_(self.w.data, mean=0, std=sigma)
        self.w = nn.Parameter(self.w.data.to(dtype=self.dtype))

    def forward(self, X: torch.Tensor, mask: torch.Tensor):
        batch_size, max_items, d = X.shape

        Q = self.Theta1(X).view(batch_size, max_items, self.H, self.d0)  # (B, J, H, d0)
        K = self.Theta2(X).view(batch_size, max_items, self.H, self.d0)  # (B, J, H, d0)
        V = self.Theta3(X).view(batch_size, max_items, self.H, -1)  # (B, J, H, d2)

        # 注意力得分计算（向量化）
        attn_scores = torch.einsum("blhd,bmhd->bhlm", Q, K)  # (B, H, J, J)
        attn_scores = attn_scores / (self.d0**0.5)
        

        attn_mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, J)
        attn_mask = attn_mask & attn_mask.transpose(2, 3)  # (B, 1, J, J)
        if self.half:
            attn_scores = attn_scores.masked_fill(~attn_mask, -1e4)
        else:
            attn_scores = attn_scores.masked_fill(~attn_mask, -1e4)

        # print(attn_scores[0])
        # 计算注意力权重
        # print(attn_scores[0])
        if self.activation == "softmax":
            attn_weights = F.softmax(attn_scores, dim=-1)
        else:
            print("not supprted")
        # print(attn_weights.shape)

        # 加权求和（向量化）
        outputs = torch.einsum("bhlm,bmhd->blhd", attn_weights, V)  # (B, J, H, d2)

        # 最终投影（保持不变）
        # weights = self.w[None, None, ...]  # (1, 1, H, d2)
        v = torch.einsum("blhd,hd->bl", outputs, self.w) / (self.H**0.5)

        # 如果使用半精度， 就将-1e7 换成 -1e4
        if self.half:
            v = v.masked_fill(~mask, -1e4)
        else:
            v = v.masked_fill(~mask, -1e4)

        # 6. 概率计算
        probs = torch.softmax(v, dim=-1)
        return probs

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

        # record the start time
        t1 = time.time()

        # train and test data, max_items
        self.X_train, self.y_train, self.cardinality_train = dataset_train
        self.X_test, self.y_test, self.cardinality_test = dataset_test
        self.max_items = self.X_train.shape[1]

        # smoothed train and test labels
        self.y_train_smooth = (
            1 - self.model_config["smoothing"]
        ) * self.y_train + self.model_config["smoothing"] / self.max_items

        self.y_test_smooth = (
            1 - self.model_config["smoothing"]
        ) * self.y_test + self.model_config["smoothing"] / self.max_items

        # train dataset and train loader
        self.train_dataset = TensorDataset(
            self.X_train, self.y_train, self.cardinality_train
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.model_config["batch_size"], shuffle=True
        )

        # if validation dataset exists, then fit with validation
        if dataset_val is not None:
            self.X_val, self.y_val, self.cardinality_val = dataset_val
            self.y_val_smooth = (
                1 - self.model_config["smoothing"]
            ) * self.y_val + self.model_config["smoothing"] / self.max_items

            # if mixed precision is True, then fit with mixed precision
            if self.model_config["mixed_precision"]:
                self.fit_mixed_precision_with_val(device=device)
            else:
                self.fit_full_precision_with_val(device=device)
        else:
            print("current model doesn't support no validation training")
            pass
        t2 = time.time()
        self.train_time = t2 - t1

    def fit_full_precision_with_val(self, device: torch.device):

        self.train()

        self.patience_counter = 0
        self.patience = self.model_config["patience"]

        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.model_config["learning_rate"],
            weight_decay=0,
        )
        self.best_val_loss = float("inf")
        self.best_parameters = None

        print(f"Start Training...")

        with tqdm(
            total=self.model_config["max_epochs"], desc="Training", unit="epoch"
        ) as pbar:
            for epoch in range(self.model_config["max_epochs"]):
                for X_batch, y_batch, cardinality_batch in self.train_loader:
                    mask_batch = compute_mask_from_card(
                        cardinality=cardinality_batch, d=X_batch.shape[1]
                    )
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    optimizer.zero_grad()
                    prob_train_batch = self.forward(X_batch, mask_batch)
                    nll_train_batch = self.cross_entropy(prob_train_batch, y_batch)
                    nll_train_batch.backward()
                    optimizer.step()

                    if self.model_config["add_noise"]:
                        current_lr = optimizer.param_groups[0]["lr"]
                        scale = current_lr * self.model_config["tau"]
                        with torch.no_grad():
                            for param in self.parameters():
                                noise = torch.randn_like(param.data)
                                param.data.add_(noise, alpha=scale)

                self.eval()
                self.evaluate(device)
                if self.nll_val < self.best_val_loss:
                    self.best_val_loss = self.nll_val
                    self.best_parameters = {
                        k: v.clone() for k, v in self.state_dict().items()
                    }
                else:
                    self.patience_counter += 1

                if self.patience_counter > self.patience:
                    pbar.close()
                    break

                pbar.set_postfix(
                    {
                        "Train Loss": self.nll_train,
                        "Val Loss": self.nll_val,
                        "Test Loss": self.nll_test,
                        "Patience": self.patience_counter,
                    }
                )
                pbar.update(1)
                self.train()

        self.load_state_dict(self.best_parameters)

    def fit_mixed_precision_with_val(
        self,
        device: torch.device,
    ):
        print("Fittiing with mixed precision (have validation set)")

        self.best_val_loss = 1e4
        self.patience = self.model_config["patience"]
        self.patience_counter = 0
        self.best_parameters = None
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.model_config["learning_rate"],
            weight_decay=0,
        )
        scaler = torch.amp.GradScaler()

        with tqdm(
            total=self.model_config["max_epochs"], desc="Training", unit="epoch"
        ) as pbar:
            for eppch in range(self.model_config["max_epochs"]):
                with torch.no_grad():
                    mask_val = compute_mask_from_card(
                        cardinality=self.cardinality_val, d=self.max_items
                    )
                    P_val = self.compute_prob_batch(
                        self.X_val, mask_val, batch_size=256, device=device
                    )
                    self.nll_val = self.cross_entropy(
                        P_val, self.y_val_smooth.to(device)
                    ).item()
                    self.acc_val = self.compute_acc(P_val, self.y_val.to(device))
                print("nll_val:", self.nll_val, "acc_val:", self.acc_val)
                for X_batch, y_batch, cardinality_batch in self.train_loader:
                    mask_batch = compute_mask_from_card(
                        cardinality=cardinality_batch, d=self.max_items
                    )
                    X_batch = X_batch.to(device, dtype=torch.float32)
                    y_batch = y_batch.to(device, dtype=torch.float32)
                    mask_batch = mask_batch.to(device)

                    with torch.amp.autocast(device_type="cuda"):
                        optimizer.zero_grad()
                        # print(self.w.dtype)
                        prob_train_batch = self.forward(X=X_batch, mask=mask_batch)
                        nll_train_batch = self.cross_entropy(prob_train_batch, y_batch)

                    scaler.scale(nll_train_batch).backward()

                    scaler.step(optimizer)
                    scaler.update()

                    if self.model_config["add_noise"]:
                        current_lr = optimizer.param_groups[0]["lr"]
                        scale = current_lr * self.model_config["tau"]
                        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                            for param in self.parameters():
                                noise = torch.randn_like(param.data)
                                param.data.add_(noise, alpha=scale)


                if self.nll_val < self.best_val_loss:
                    self.best_val_loss = self.nll_val
                    self.best_parameters = {
                        k: v.clone() for k, v in self.state_dict().items()
                    }
                else:
                    self.patience_counter += 1

                if self.patience_counter > self.patience:
                    pbar.close()
                    break

                pbar.set_postfix(
                    {
                        "Val Loss": self.nll_val,
                        "Val Acc": self.acc_val,
                        "Patience": self.patience_counter,
                    }
                )
                pbar.update(1)
                self.train()

        optimizer.state.clear()
        del scaler
        del optimizer
        report_memory(device)

        self.load_state_dict(self.best_parameters)

    def compute_prob_batch(
        self,
        X_full: torch.Tensor,
        mask_full: torch.Tensor,
        batch_size: int = 1024,
        device: torch.device = None,
    ):
        with torch.no_grad():
            all_probs = []
            for i in range(0, len(X_full), batch_size):
                X_batch = X_full[i : i + batch_size].to(device)
                mask_batch = mask_full[i : i + batch_size].to(device)
                probs_batch = self.forward(X_batch.to(torch.float32), mask_batch)
                all_probs.append(probs_batch.cpu())

        return torch.cat(all_probs, dim=0).to(device)

    def compute_acc(self, P: torch.Tensor, Y: torch.Tensor):
        P_max = torch.argmax(P, dim=1)
        Y_max = torch.argmax(Y, dim=1)
        return torch.sum(P_max == Y_max).item() / Y.size(0)

    def evaluate(self, device):
        with torch.no_grad():
            max_items = self.X_train.shape[1]

            mask_train = compute_mask_from_card(
                cardinality=self.cardinality_train, d=max_items
            )
            P_train = self.compute_prob_batch(self.X_train, mask_train, device=device)
            self.nll_train = self.cross_entropy(
                P_train, self.y_train_smooth.to(device)
            ).item()
            self.acc_train = self.compute_acc(P_train, self.y_train.to(device))

            mask_val = compute_mask_from_card(
                cardinality=self.cardinality_val, d=max_items
            )
            P_val = self.compute_prob_batch(self.X_val, mask_val, device=device)
            self.nll_val = self.cross_entropy(
                P_val, self.y_val_smooth.to(device)
            ).item()
            self.acc_val = self.compute_acc(P_val, self.y_val.to(device))

            mask_test = compute_mask_from_card(
                cardinality=self.cardinality_test, d=max_items
            )
            P_test = self.compute_prob_batch(self.X_test, mask_test, device=device)
            self.nll_test = self.cross_entropy(
                P_test, self.y_test_smooth.to(device)
            ).item()
            self.acc_test = self.compute_acc(P_test, self.y_test.to(device))
