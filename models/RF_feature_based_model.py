import torch
import numpy as np
import time
import os
import csv
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils.creterias_and_loss_utils import (
    compute_U_RF,
    compute_P,
    nll,
    accuracy,
    cross_entropy,
)


class GaussianRFFB:
    def __init__(
        self,
        model_args: dict,
        feature_length: int,
        device: torch.device,
    ):
        self.model_args = model_args
        self.kernel_type = model_args["kernel_type"]
        self.feature_length = feature_length
        self.sigma = model_args["kernel_params"]["sigma"]
        self.lengthscale = model_args["kernel_params"]["lengthscale"]
        self.Nw = model_args["Nw"]
        self.device = device
        self.init_samples()

    def init_samples(self):
        self.W = (
            torch.randn(self.Nw, self.feature_length, device=self.device)
            / self.lengthscale
        )
        self.b: torch.Tensor = (
            torch.rand(
                self.Nw,
                device=self.device,
            )
            * 2
            * torch.pi
        )

    def compute_features(self):
        """
        Compute the features for the training, validation, and test sets.
        """
        # W_X = torch.zeros(self.N_train, self.Nw, self.d, device=device)
        W_expanded = self.W.unsqueeze(1)
        W_X_train = torch.einsum(
            "nkl,mjl->mnj", W_expanded, self.X_train
        )  # N_train * Nw * d
        W_X_val = torch.einsum(
            "nkl,mjl->mnj", W_expanded, self.X_val
        )  # N_train * Nw * d
        W_X_test = torch.einsum(
            "nkl,mjl->mnj", W_expanded, self.X_test
        )  # N_train * Nw * d

        W_X_train_plus_b = W_X_train + self.b.unsqueeze(0).unsqueeze(2)
        W_X_val_plus_b = W_X_val + self.b.unsqueeze(0).unsqueeze(2)
        W_X_test_plus_b = W_X_test + self.b.unsqueeze(0).unsqueeze(2)

        self.Phi_train: torch.Tensor = (
            self.sigma * np.sqrt(2 / self.Nw) * torch.cos(W_X_train_plus_b)
        )  # N_train * Nw * d
        self.Phi_val: torch.Tensor = (
            self.sigma * np.sqrt(2 / self.Nw) * torch.cos(W_X_val_plus_b)
        )  # N_train * Nw * d
        self.Phi_test: torch.Tensor = (
            self.sigma * np.sqrt(2 / self.Nw) * torch.cos(W_X_test_plus_b)
        )  # N_train * Nw * d

    def optimize_distribution(
        self,
        dataset_train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dataset_val: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dataset_test: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """
        Optimize the distribution of the features.
        """

        print("Begin Optimizing Distribution")

        self.X_train, self.y_train, self.cardinality_train = dataset_train
        self.X_val, self.y_val, self.cardinality_val = dataset_val
        self.X_test, self.y_test, self.cardinality_test = dataset_test

        # move to device
        self.X_train = self.X_train.to(self.device)
        self.X_val = self.X_val.to(self.device)
        self.X_test = self.X_test.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.y_val = self.y_val.to(self.device)
        self.y_test = self.y_test.to(self.device)
        self.cardinality_train = self.cardinality_train.to(self.device)
        self.cardinality_val = self.cardinality_val.to(self.device)
        self.cardinality_test = self.cardinality_test.to(self.device)

        # datasize
        self.N_train = self.X_train.shape[0]
        self.N_val = self.X_val.shape[0]
        self.N_test = self.X_test.shape[0]

        self.d = self.X_train.shape[1]
        self.smoothing = self.model_args["smoothing"]

        self.y_train_smooth = (
            1 - self.smoothing
        ) * self.y_train + self.smoothing / self.d
        self.y_val_smooth = (1 - self.smoothing) * self.y_val + self.smoothing / self.d
        self.y_test_smooth = (
            1 - self.smoothing
        ) * self.y_test + self.smoothing / self.d

        self.similar_train = torch.where(self.y_train == 0, -1, self.y_train)
        self.similar_val = torch.where(self.y_val == 0, -1, self.y_val)
        self.similar_test = torch.where(self.y_test == 0, -1, self.y_test)

        self.S_train = (
            torch.arange(self.d, device=self.device)
            <= self.cardinality_train.unsqueeze(1) - 1
        )
        self.S_val = (
            torch.arange(self.d, device=self.device)
            <= self.cardinality_val.unsqueeze(1) - 1
        )
        self.S_test = (
            torch.arange(self.d, device=self.device)
            <= self.cardinality_test.unsqueeze(1) - 1
        )

        self.item_num_train = torch.sum(self.cardinality_train)
        self.item_num_val = torch.sum(self.cardinality_val)
        self.item_num_test = torch.sum(self.cardinality_test)

        self.Y = self.similar_train.reshape(-1, 1)

        self.rho = self.model_args["rho"]
        self.tol = self.model_args["tol"]
        self.init_q = torch.ones(self.Nw, device=self.device) / self.Nw
        self.eps = self.model_args["eps"]

        # compute the features
        self.compute_features()

        # mask out the features that are not in the offered choice set
        self.mask()

        self.Phi = self.Phi_train.reshape(self.Nw, -1)
        V = torch.matmul(self.Phi, self.Y)
        self.V_hadamard = V * V

        self.lambda_upper = float("inf")
        self.lambda_lower = 0
        self.lambda_s = 1
        self.init_q = torch.ones(self.Nw, device=self.device) / self.Nw

        alignment_start = time.time()

        while self.lambda_upper == float("inf"):
            q = self.find_q(self.lambda_s)
            if self.chi_2_divergence(q) < self.rho:
                self.lambda_upper = self.lambda_s
            else:
                self.lambda_s *= 2

        while self.lambda_upper - self.lambda_lower > self.eps * self.lambda_s:
            lambda_mid = (self.lambda_upper + self.lambda_lower) / 2
            q = self.find_q(lambda_mid)
            if self.chi_2_divergence(q) < self.rho:
                self.lambda_upper = lambda_mid
            else:
                self.lambda_lower = lambda_mid
            # print("Gap:", self.lambda_upper - self.lambda_lower)
        self.q_opt = q.view(-1)

        alignment_end = time.time()
        self.alignment_time = alignment_end - alignment_start

        self.Q_diag = torch.diag(self.q_opt)
        self.sqrt_Q_diag = torch.sqrt(self.Q_diag)  # Nw x Nw

        print("Optimization Distribution Finished")

    def mask(self):
        """
        Mask out the features that are not in the offered choice set.
        """
        j_indices = torch.arange(self.d, device=self.device)
        cardinality_mask = j_indices > (self.cardinality_train - 1).unsqueeze(1)
        cardinality_mask = cardinality_mask.unsqueeze(1)
        self.Phi_train[cardinality_mask.expand_as(self.Phi_train)] = 0

    def target_function(self, tau: float, labmda_: float):
        """
        Compute the target function for the optimization.
        """
        q = (self.V_hadamard / (labmda_ * self.Nw) + tau).clamp(min=0)
        return torch.sum(q) - 1

    def find_q(self, lambda_: float):
        """
        Find the optimal q for the optimization.
        """
        tau = self.find_tau(lambda_)
        q = (self.V_hadamard / (lambda_ * self.Nw) + tau).clamp(min=0)
        return q

    def find_tau(self, lambda_: float):
        """
        Find the optimal tau for the optimization.
        """
        tau_low, tau_high = -torch.max(self.V_hadamard / (lambda_ * self.Nw)), 1.0
        f1 = self.target_function(tau_low, lambda_)
        f2 = self.target_function(tau_high, lambda_)
        if f1 * f2 > 0:
            print("ValueError: f1 and f2 should have different signs!!")
            return None

        max_iteration = 1000
        iteration = 0
        while (tau_high - tau_low).item() > self.tol and iteration < max_iteration:
            tau_mid = (tau_low + tau_high) / 2
            f_mid = self.target_function(tau_mid, lambda_)
            if f_mid < 0:
                tau_low = tau_mid
            elif f_mid > 0:
                tau_high = tau_mid
            else:
                return tau_mid
            iteration += 1
            # print("Gap:", (tau_high - tau_low).item())
        if iteration >= max_iteration:
            print("Max Iteration Reached without Convergence")
        return (tau_low + tau_high) / 2

    def chi_2_divergence(self, q: torch.Tensor):
        """
        Compute the chi-2 divergence.
        """
        return torch.sum((q - self.init_q).pow(2) / self.init_q)

    def fit(self):
        """
        Fit the model with Adam optimizer.
        """
        self.lr = self.model_args["learning_rate"]
        self.lambda_ = self.model_args["lambda"]
        self.theta_std = self.model_args["theta_std"]
        self.theta = torch.randn(self.Nw, device=self.device) * self.theta_std
        self.theta.requires_grad = True
        self.best_loss_val = float("inf")
        self.best_theta = None
        self.patience = self.model_args["patience"]
        patience_counter = 0

        self.optimizer = torch.optim.Adam([self.theta], lr=self.lr)

        train_start = time.time()
        with tqdm(total=50000, desc="Training", unit="epoch") as pbar:
            for epoch in range(50000):
                self.optimizer.zero_grad()
                loss = self.objective()

                self.evaluate(self.theta)

                if self.nll_val < self.best_loss_val:
                    self.best_loss_val = self.nll_val
                    patience_counter = 0
                    self.best_theta = self.theta.clone().detach()
                else:
                    patience_counter += 1
                if patience_counter > self.patience:
                    pbar.close()
                    print(
                        "Early Stopping at Epoch:",
                        epoch,
                        "Validation NLL:",
                        self.nll_val,
                    )
                    break
                pbar.set_postfix(
                    {
                        "Train Loss": self.nll_train,
                        "Val Loss": self.nll_val,
                        "Test Loss": self.nll_test,
                        "Patience": patience_counter,
                    }
                )
                pbar.update(1)
                loss.backward()
                self.optimizer.step()
        train_end = time.time()
        self.train_time = train_end - train_start

    def objective(self):
        U = compute_U_RF(
            theta=self.theta, Phi=self.Phi_train, sqrt_Q_diag=self.sqrt_Q_diag
        )
        loss = (
            cross_entropy(U=U, y=self.y_train, mask_tensor=self.S_train)
            + self.lambda_ * self.l2_regularization()
        )
        return loss

    def l2_regularization(self):
        reg = torch.sum(self.theta.pow(2))
        return reg

    def evaluate(self, theta: torch.Tensor):
        self.U_train = compute_U_RF(
            theta=theta, Phi=self.Phi_train, sqrt_Q_diag=self.sqrt_Q_diag
        )
        self.U_val = compute_U_RF(
            theta=theta, Phi=self.Phi_val, sqrt_Q_diag=self.sqrt_Q_diag
        )
        self.U_test = compute_U_RF(
            theta=theta, Phi=self.Phi_test, sqrt_Q_diag=self.sqrt_Q_diag
        )

        self.P_train = compute_P(U=self.U_train, mask_tensor=self.S_train)
        self.P_val = compute_P(U=self.U_val, mask_tensor=self.S_val)
        self.P_test = compute_P(U=self.U_test, mask_tensor=self.S_test)

        self.acc_train = accuracy(Y=self.y_train, P=self.P_train)
        self.acc_val = accuracy(Y=self.y_val, P=self.P_val)
        self.acc_test = accuracy(Y=self.y_test, P=self.P_test)

        self.nll_train = nll(P=self.P_train, y=self.y_train_smooth)
        self.nll_val = nll(P=self.P_val, y=self.y_val_smooth)
        self.nll_test = nll(P=self.P_test, y=self.y_test_smooth)
