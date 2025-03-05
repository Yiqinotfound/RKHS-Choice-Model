from HaloDataLoader import HaloDataset
import torch
import time
import csv
import gc
import os

eps = torch.tensor(1e-7)
safe_log = lambda x: torch.log(torch.clamp(x, eps, 1.0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CMNL:
    def __init__(self, dataset: HaloDataset, device: torch.device):
        self.dataset = dataset
        self.train_datasize = self.dataset.train_datasize
        self.product_number_train = self.dataset.S_train.sum().item()
        self.device = device

    def objective(self, mu: torch.Tensor, alphaset: torch.Tensor):
        U = self.compute_U(mu, alphaset, self.dataset.S_train)
        P = self.compute_P(U, self.dataset.S_train)
        return self.cross_entropy(P)

    def constraint1(self, U: torch.Tensor):
        return torch.sum(U[:, 0] ** 2) * 100

    def cross_entropy(self, P):
        return -torch.sum(self.dataset.y_train * safe_log(P)) / self.train_datasize

    def compute_U(self, mu: torch.Tensor, alphaset: torch.Tensor, S: torch.Tensor):
        interaction = torch.matmul(S, alphaset)
        return mu + interaction

    def zero_diagonal_gradients(self, alphaset: torch.Tensor):
        alphaset_grad = alphaset.grad
        if alphaset_grad is not None:
            alphaset_grad[
                torch.eye(self.dataset.item_number, device=self.device).bool()
            ] = 0
        return alphaset_grad

    def compute_P(self, U: torch.Tensor, S: torch.Tensor):
        U_max = torch.max(U, dim=1, keepdim=True).values
        U_stable = U - U_max
        exp_U = torch.exp(U_stable)
        exp_U_masked = exp_U * S
        sum_exp_utility = torch.sum(exp_U_masked, dim=1, keepdim=True)
        P = exp_U_masked / sum_exp_utility
        return P

    def run(self, grad_norm_threshold: float = 1e-3, lr: float = 0.01):
        self.grad_norm_threshold = grad_norm_threshold
        self.lr = lr
        time1 = time.time()
        self.mu = torch.randn(
            (self.dataset.item_number),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alphaset = torch.zeros(
            (self.dataset.item_number, self.dataset.item_number),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        self.optimizer = torch.optim.Adam([self.alphaset, self.mu], lr=0.01)
        self.best_loss = float("inf")
        self.best_alphaset = None
        self.best_mu = None

        epoch = 0
        while True:
            epoch += 1
            self.optimizer.zero_grad()

            loss_value = self.objective(self.mu, self.alphaset)
            if loss_value.item() < self.best_loss:
                self.best_loss = loss_value.item()
                self.best_alphaset = self.alphaset.clone().detach()
                self.best_mu = self.mu.clone().detach()
            loss_value.backward()
            self.zero_diagonal_gradients(self.alphaset)
            grad = self.alphaset.grad
            grad_norm = torch.norm(grad).item()
            if grad_norm < 1e-3:
                print(f"Early Stopping at Epoch {epoch}, Best Loss: {self.best_loss}")
                break
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}], Loss: {loss_value.item():.4f}")
        time2 = time.time()
        self.train_time = time2 - time1

    def evaluate(self):
        self.U = self.compute_U(self.best_mu, self.best_alphaset, self.dataset.S_train)
        self.P = self.compute_P(self.U, self.dataset.S_train)
        self.nll = self.cross_entropy(self.P).item()
        self.hard_rmse = torch.sqrt(
            torch.sum((self.P - self.dataset.y_train) ** 2) / self.product_number_train
        ).item()
        self.soft_rmse = torch.sqrt(
            torch.sum((self.P - self.dataset.F_train) ** 2) / self.product_number_train
        ).item()
        print(f"Test RMSE: {self.hard_rmse:.4f}, Test NLL: {self.nll:.4f}")

    def save_results(self, filepath: str):
        results = [[self.nll, self.hard_rmse, self.soft_rmse, self.train_time]]
        file_exists = os.path.exists(filepath)
        with open(filepath, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists or os.stat(filepath).st_size == 0:
                writer.writerow(
                    [
                        "nll",
                        "hard_rmse",
                        "soft_rmse",
                        "train_time",
                    ]
                )
            writer.writerows(results)
