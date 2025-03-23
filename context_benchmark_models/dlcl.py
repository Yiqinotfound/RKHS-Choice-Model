import time

import numpy as np
import torch
from torch import nn, jit
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from utils.creterias_and_loss_utils import safe_log


# From https://github.com/pytorch/pytorch/issues/31829
@jit.script
def logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float("inf")

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float("inf"))


class Embedding(nn.Module):
    """
    Add zero-ed out dimension to Embedding for the padding index.
    From https://github.com/arjunsesh/cdm-icml with minor tweaks.
    """

    def __init__(self, num, dim, pad_idx=None):
        super().__init__()
        self.num = num
        self.dim = dim
        self.pad_idx = pad_idx

        self.weight = nn.Parameter(torch.randn([self.num, self.dim]))

        with torch.no_grad():
            self.weight[self.pad_idx].fill_(0)

    def forward(self, x):
        with torch.no_grad():
            self.weight[self.pad_idx].fill_(0)

        return self.weight[x]


class DLCL(nn.Module):

    name = "dlcl"

    def __init__(self, num_features, model_args: dict):
        super().__init__()

        self.num_features = num_features
        l1_reg = model_args["l1_reg"]
        self.model_args = model_args

        # Context effect slopes
        self.A = nn.Parameter(
            torch.zeros(self.num_features, self.num_features), requires_grad=True
        )

        # Context effect intercepts
        self.B = nn.Parameter(
            torch.zeros(self.num_features, self.num_features), requires_grad=True
        )

        self.mixture_weights = nn.Parameter(
            torch.ones(self.num_features), requires_grad=True
        )

        self.l1_reg = l1_reg

    def forward(self, choice_set_features, choice_set_lengths):
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()
        self.device = next(self.parameters()).device

        # Compute mean of each feature over each choice set
        mean_choice_set_features = (
            choice_set_features.sum(1) / choice_set_lengths[:, None]
        )

        # Use learned linear context model to compute utility matrices for each sample
        utility_matrices = self.B + self.A * (
            torch.ones(self.num_features, 1, device=self.device)
            @ mean_choice_set_features[:, None, :]
        )

        # Compute utility of each item under each feature MNL
        utilities = choice_set_features @ utility_matrices
        utilities[
            torch.arange(max_choice_set_len, device=self.device)[None, :]
            >= choice_set_lengths[:, None]
        ] = -np.inf

        # Compute MNL log-probs for each feature
        log_probs = nn.functional.log_softmax(utilities, 1)

        # Combine the MNLs into single probability using weights
        # This is what I want to do, but logsumexp produces nan gradients when there are -infs
        # https://github.com/pytorch/pytorch/issues/31829
        # return torch.logsumexp(log_probs + torch.log(self.weights / self.weights.sum()), 2)

        # So, I'm instead using the fix in the issue linked above
        return torch.exp(
            logsumexp(log_probs + torch.log_softmax(self.mixture_weights, 0), 2)
        )

    def loss(self, y_pred, y):
        """
        The error in inferred log-probabilities given observations
        :param y_pred: log(choice probabilities)
        :param y: observed choices
        :return: the loss
        """

        return nn.functional.nll_loss(y_pred, y) + self.l1_reg * self.A.norm(1)

    def compute_prob_batch(
        self,
        X_full: torch.Tensor,
        card_full: torch.Tensor,
        batch_size: int = 1024,
        device: torch.device = None,
    ):
        with torch.no_grad():
            all_probs = []
            for i in range(0, len(X_full), batch_size):
                X_batch = X_full[i : i + batch_size].to(device)
                card_batch = card_full[i : i + batch_size].to(device)
                probs_batch = self.forward(
                    choice_set_features=X_batch, choice_set_lengths=card_batch
                )
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

        self.best_val_loss = float("inf")

        with tqdm(
            total=self.model_args["max_epochs"], desc="Training", unit="epoch"
        ) as pbar:
            for epoch in range(self.model_args["max_epochs"]):
                total_train_loss = 0.0
                for X_batch, y_batch, cardinality_batch in self.train_loader:

                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    cardinality_batch = cardinality_batch.to(device)
                    optimizer.zero_grad()
                    prob_train_batch = self.forward(
                        choice_set_features=X_batch,
                        choice_set_lengths=cardinality_batch,
                    )
                    nll_train_batch = self.cross_entropy(P=prob_train_batch, Y=y_batch)
                    nll_train_batch.backward()
                    optimizer.step()
                    total_train_loss += nll_train_batch.item()

                # evaluate
                self.evaluate(device)
                if self.nll_val < self.best_val_loss:
                    self.best_val_loss = self.nll_val
                    self.best_parameters = {
                        k: v.clone() for k, v in self.state_dict().items()
                    }

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

            P_train = self.compute_prob_batch(
                X_full=self.X_train, card_full=self.cardinality_train, device=device
            )
            self.nll_train = self.cross_entropy(
                P=P_train, Y=self.y_train_smooth.to(device)
            ).item()
            self.acc_train = self.compute_acc(P=P_train, Y=self.y_train.to(device))

            P_val = self.compute_prob_batch(
                X_full=self.X_val, card_full=self.cardinality_val, device=device
            )
            self.nll_val = self.cross_entropy(
                P=P_val, Y=self.y_val_smooth.to(device)
            ).item()
            self.acc_val = self.compute_acc(P=P_val, Y=self.y_val.to(device))

            P_test = self.compute_prob_batch(
                X_full=self.X_test, card_full=self.cardinality_test, device=device
            )
            self.nll_test = self.cross_entropy(
                P=P_test, Y=self.y_test_smooth.to(device)
            ).item()
            self.acc_test = self.compute_acc(P=P_test, Y=self.y_test.to(device))


def train_model(
    model,
    train_data,
    val_data,
    lr,
    weight_decay,
    compute_val_stats=True,
    timeout_min=60,
    only_context_effect=None,
):
    torch.set_num_threads(1)

    batch_size = 128
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay
    )

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    start_time = time.time()

    for epoch in tqdm(range(500)):
        train_loss = 0
        train_count = 0
        train_correct = 0
        total_loss = 0

        if time.time() - start_time > timeout_min * 60:
            break

        for batch in train_data_loader:
            choices = batch[-1]
            model.train()

            choice_pred = model(*batch[:-1])
            loss = model.loss(choice_pred, choices)

            total_loss += nn.functional.nll_loss(
                choice_pred, choices, reduction="sum"
            ).item()

            optimizer.zero_grad()
            loss.backward()

            # If we only want to train one context effect, zero every other gradient
            if only_context_effect is not None:
                tmp = model.A.grad.data[only_context_effect].item()
                model.A.grad.data.zero_()
                model.A.grad.data[only_context_effect] = tmp
            optimizer.step()

            model.eval()
            vals, idxs = choice_pred.max(1)
            train_correct += (idxs == choices).long().sum().item() / choice_pred.size(0)
            train_loss += loss.item()
            train_count += 1

        train_accs.append(train_correct / train_count)
        train_losses.append(total_loss)

        if compute_val_stats:
            total_val_loss = 0
            val_loss = 0
            val_count = 0
            val_correct = 0
            model.eval()
            for batch in val_data_loader:
                choices = batch[-1]
                choice_pred = model(*batch[:-1])
                loss = model.loss(choice_pred, choices)
                vals, idxs = choice_pred.max(1)
                val_correct += (idxs == choices).long().sum().item() / choice_pred.size(
                    0
                )
                val_loss += loss.item()

                total_val_loss += nn.functional.nll_loss(
                    choice_pred, choices, reduction="sum"
                ).item()
                val_count += 1

            val_losses.append(val_loss)
            val_accs.append(val_correct / val_count)

    return model, train_losses, train_accs, val_losses, val_accs


def train_dlcl(
    train_data,
    val_data,
    num_features,
    lr=1e-4,
    weight_decay=1e-4,
    compute_val_stats=False,
    l1_reg=0,
):
    model = DLCL(num_features, l1_reg=l1_reg)
    return train_model(
        model,
        train_data,
        val_data,
        lr,
        weight_decay,
        compute_val_stats=compute_val_stats,
    )
