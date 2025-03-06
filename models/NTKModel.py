import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils.creterias_and_loss_utils import cross_entropy, compute_P, rmse


class NTKChoiceModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, model_args: dict):
        super(NTKChoiceModel, self).__init__()

        # initialize model parameters
        self.model_args = model_args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = model_args["H"]
        self.activation = self.select_activation(model_args["activation"])
        self.sigma = model_args["sigma"]
        self.learning_rate = model_args["learning_rate"]
        self.optimizer = model_args["optimizer"]
        self.beta = model_args["beta"]
        self.max_epochs = model_args["max_epochs"]
        self.batch_size = model_args["batch_size"]

        # initialize model layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=self.sigma)
            nn.init.normal_(layer.bias, mean=0.0, std=self.sigma)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=self.sigma)
        nn.init.normal_(self.output_layer.bias, mean=0.0, std=self.sigma)

    def select_activation(self, activation: str):
        if activation == "relu":
            return torch.relu
        elif activation == "tanh":
            return torch.tanh
        elif activation == "sigmoid":
            return torch.sigmoid
        else:
            raise ValueError(f"Activation function {activation} not supported")

    def forward(self, x):
        alpha = x
        for layer in self.layers:
            alpha_tilde = (1.0 / layer.weight.shape[1]) * layer(
                alpha
            ) + self.beta * layer.bias
            alpha = torch.relu(alpha_tilde)

        utility = self.output_layer(alpha)
        return utility

    def fit(
        self,
        S_train: torch.Tensor,
        y_train: torch.Tensor,
        S_test: torch.Tensor,
        y_test: torch.Tensor,
    ):
        self.train()
        self.S_train = S_train
        self.y_train = y_train
        self.S_test = S_test
        self.y_test = y_test

        self.total_train_items = torch.sum(S_train)
        self.total_test_items = torch.sum(S_test)

        self.dataset_train = TensorDataset(S_train, y_train)
        self.dataset_test = TensorDataset(S_test, y_test)
        if self.optimizer == "Adam":
            self.fit_with_Adam()
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")

    def fit_with_Adam(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.best_nll_train = float("inf")
        with tqdm(range(self.max_epochs)) as pbar:
            for epoch in range(self.max_epochs):
                optimizer.zero_grad()
                U = self.forward(self.S_train)
                loss = cross_entropy(
                    U=U, y=self.y_train, mask_tensor=self.S_train
                )
                if loss < self.best_nll_train:
                    self.best_nll_train = loss
                    self.best_model = self.state_dict().copy()

                loss.backward()
                optimizer.step()
                self.evaluate()
                pbar.set_postfix(
                    {"Train NLL": self.nll_train, "Test NLL": self.nll_test, "Train RMSE": self.rmse_train, "Test RMSE": self.rmse_test}
                )
                pbar.update(1)
        self.load_state_dict(self.best_model)

    def evaluate(self):
        with torch.no_grad():
            U_train = self.forward(self.S_train)
            U_test = self.forward(self.S_test)
            P_train = compute_P(U=U_train, mask_tensor=self.S_train)
            P_test = compute_P(U=U_test, mask_tensor=self.S_test)
            self.nll_train = cross_entropy(
                U=U_train, y=self.y_train, mask_tensor=self.S_train
            ).item()
            self.nll_test = cross_entropy(
                U=U_test, y=self.y_test, mask_tensor=self.S_test
            ).item()
            self.rmse_train = rmse(
                P=P_train, y=self.y_train, total_item=self.total_train_items
            )
            self.rmse_test = rmse(
                P=P_test, y=self.y_test, total_item=self.total_test_items
            )


# # 定义模型参数
# input_dim = 4
# hidden_dims = [10]  # 自定义层数和宽度
# output_dim = 4
# beta = 1.0

# # 创建模型
# model = NTKModel(input_dim, hidden_dims, output_dim, beta)

# x = torch.tensor([[0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)

# print(model(x))
# train_model(model, train_data, train_labels)
