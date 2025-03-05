import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import time
import numpy as np
import os
from utils.creterias_and_loss_utils import *
from data_preprocess.hotel_data_loader import HotelDataset


class GaussainRFFF:
    def __init__(
        self,
        kernel_type: str,
        kernel_params: dict,
        d: int,
        Nw: int,
        mask: bool,
        device: torch.device,
    ):

        # kernel parameters
        self.kernel_type = kernel_type
        self.sigma = kernel_params["sigma"]
        self.lengthscale = kernel_params["lengthscale"]
        self.Nw = Nw
        self.d = d

        # mask or not
        self.mask = mask

        # device
        self.device = device

        # initial samples
        self.initial_sample()

    def initial_sample(self):
        self.init_W: torch.Tensor = (
            torch.randn(self.Nw, self.d, self.d, device=self.device) / self.lengthscale
        )
        self.init_b: torch.Tensor = (
            torch.rand(self.Nw, self.d, device=self.device) * 2 * torch.pi
        )

    def optimize_distribution(
        self,
        dataset: HotelDataset,
        rho: float,
        tol: float,
        eps: float,
        align_type: int = 0,
        smoothing: float = 1e-2,
    ):

        # initialize dataset
        self.dataset = dataset
        self.N_train = self.dataset.train_datasize
        self.N_test = self.dataset.test_datasize
        self.total_train_item = self.dataset.total_train_item
        self.total_test_item = self.dataset.total_test_item

        # smoothing parameter
        self.smoothing = smoothing

        # S,y and y_smoothing
        self.S_train = self.dataset.S_train
        self.y_train = self.dataset.y_train
        self.y_train_smoothing = (
            1 - self.smoothing
        ) * self.y_train + self.smoothing / self.d

        self.S_test = self.dataset.S_test
        self.y_test = self.dataset.y_test
        self.y_test_smoothing = (
            1 - self.smoothing
        ) * self.y_test + self.smoothing / self.d

        # similar vector for align type 1
        self.similar_train = torch.where(self.y_train == 0, -1, self.y_train)
        self.similar_test = torch.where(self.y_test == 0, -1, self.y_test)

        # align type
        self.align_type = align_type
        if self.align_type == 0:
            self.Y = self.y_train.reshape(-1, 1)
        elif self.align_type == 1:
            self.Y = self.similar_train.reshape(-1, 1)
        elif self.align_type == 2:
            self.Y = self.y_train_smoothing.reshape(-1, 1)

        # alignment parameters
        self.rho = rho
        self.tol = tol
        self.init_q = torch.ones(self.Nw, device=self.device) / self.Nw
        self.eps = eps
    
        init_W_samples = self.init_W.view(self.Nw, self.d, self.d)  # (Nw, d, d)
        init_b_samples = self.init_b.view(self.Nw, self.d)  # (N, d)

        W_train = torch.einsum(
            "wmd,nd->wnm", init_W_samples, self.S_train
        )  # (Nw, N_train, d)
        W_test = torch.einsum(
            "wmd,nd->wnm", init_W_samples, self.S_test
        )  # (Nw, N_test, d)

        B_train = init_b_samples.unsqueeze(1).expand(
            -1, self.N_train, -1
        )  # (Nw, N_train, d)

        B_test = init_b_samples.unsqueeze(1).expand(
            -1, self.N_test, -1
        )  # (Nw, N_test, d)

        self.initial_Phi_train = torch.tensor(np.sqrt(2)) * torch.cos(
            W_train + B_train
        )  # (Nw, N_train, d)
        self.initial_Phi_test = torch.tensor(np.sqrt(2)) * torch.cos(
            W_test + B_test
        )  # (Nw, N_test, d)

        self.mask_feature_maps()

        self.Phi_train = self.initial_Phi_train.permute(1, 0, 2)  # N_train x Nw x d
        self.Phi_test = self.initial_Phi_test.permute(1, 0, 2)  # N_test x Nw x d

        self.Phi = self.initial_Phi_train.reshape(self.Nw, -1)

        V = torch.matmul(self.Phi, self.Y)
        self.V_hadamard = V * V

        self.lambda_upper = float("inf")
        self.lambda_lower = 0
        self.lambda_s = 1
        self.init_q = torch.ones(self.Nw, device=self.device) / self.Nw
        print("Before Alignment: ", torch.matmul(self.init_q.T, self.V_hadamard).item())

        alignment_start = time.time()

        while self.lambda_upper == float("inf"):
            q = self.find_q(self.lambda_s)
            if self.chi_2_divergence(q) < self.rho:
                self.lambda_upper = self.lambda_s
            else:
                self.lambda_s *= 2
            # print("lambda_s:", self.lambda_s)

        # print("lambda_s:", self.lambda_s)
        while self.lambda_upper - self.lambda_lower > self.eps * self.lambda_s:
            lambda_mid = (self.lambda_upper + self.lambda_lower) / 2
            q = self.find_q(lambda_mid)
            print(torch.matmul(q.T, self.V_hadamard)[0].item())
            if self.chi_2_divergence(q) < self.rho:
                self.lambda_upper = lambda_mid
            else:
                self.lambda_lower = lambda_mid
            # print("Gap:", self.lambda_upper - self.lambda_lower)
        self.q_opt = q.view(-1)
        print("After Alignment: ", torch.matmul(self.q_opt.T, self.V_hadamard).item())

        alignment_end = time.time()
        self.alignment_time = alignment_end - alignment_start

        print("q_opt:", self.q_opt.shape)
        self.Q_diag = torch.diag(self.q_opt)
        self.sqrt_Q_diag = torch.sqrt(self.Q_diag)  # Nw x Nw

    def mask_feature_maps(self):
        mask_train = self.S_train == 0
        self.initial_Phi_train[:, mask_train] = 0
        mask_test = self.S_test == 0
        self.initial_Phi_test[:, mask_test] = 0

    def target_function(self, tau: float, labmda_: float):
        q = (self.V_hadamard / (labmda_ * self.Nw) + tau).clamp(min=0)
        return torch.sum(q) - 1

    def find_q(self, lambda_: float):
        tau = self.find_tau(lambda_)
        q = (self.V_hadamard / (lambda_ * self.Nw) + tau).clamp(min=0)
        return q

    def find_tau(self, lambda_: float):
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
        return torch.sum((q - self.init_q).pow(2) / self.init_q)

    def train_with_Adam(
        self,
        theta_std: float = 0.01,
        lambda_: float = 0.01,
        grad_norm_threshold: float = 1e-4,
    ):
        self.lambda_ = lambda_
        self.theta = torch.randn(self.Nw, device=self.device) * theta_std
        self.theta.requires_grad = True
        self.best_loss = float("inf")
        self.best_theta = None
        self.grad_norm_threshold = grad_norm_threshold

        self.optimizer = torch.optim.Adam([self.theta], lr=0.01)

        train_start = time.time()
        for epoch in range(4000):
            self.optimizer.zero_grad()
            loss = self.objective()
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_theta = self.theta.clone().detach()
            loss.backward()
            self.optimizer.step()
            grad_norm = torch.norm(self.theta.grad).item()
            if grad_norm < self.grad_norm_threshold:
                print(
                    "Early Stopping at Epoch:",
                    epoch,
                    "Loss:",
                    loss.item(),
                    "Grad Norm:",
                    grad_norm,
                )
                break
            if epoch % 10 == 0:
                print("Epoch:", epoch, "Loss:", loss.item(), "Grad Norm:", grad_norm)
        train_end = time.time()
        self.train_time = train_end - train_start

    def train_with_LBFGS(self, theta_std: float = 0.01, lambda_: float = 0.01):
        self.lambda_ = lambda_
        self.theta = torch.randn(self.Nw, device=self.device) * theta_std
        self.theta.requires_grad = True
        self.best_loss = float("inf")
        self.best_theta = None

        self.optimizer = torch.optim.LBFGS([self.theta], lr=0.01)

        def closure():
            self.optimizer.zero_grad()
            loss = self.objective()
            loss.backward()
            return loss

        for epoch in range(1000):
            loss = self.optimizer.step(closure)
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_theta = self.theta.clone().detach()
            grad_norm = torch.norm(self.theta.grad)
            print(grad_norm)
            if epoch % 10 == 0:
                print("Epoch:", epoch, "Loss:", closure().item())

    def objective(
        self,
    ):
        U = compute_U_RF(
            theta=self.theta, Phi=self.Phi_train, sqrt_Q_diag=self.sqrt_Q_diag
        )
        loss = cross_entropy_FF(
            U=U, S=self.S_train, y=self.y_train
        ) + self.lambda_ * l2_regularization_RF(self.theta)
        return loss

    def calculate_P(self, U: torch.Tensor, S: torch.Tensor):
        U_max = torch.max(U, dim=1, keepdim=True).values
        U_stable = U - U_max
        exp_U = torch.exp(U_stable)
        exp_U_masked = exp_U * S
        exp_U_sum = torch.sum(exp_U_masked, dim=1, keepdim=True)
        P = exp_U_masked / exp_U_sum
        return P

    def evaluate(self):
        self.U_train = compute_U_RF(
            theta=self.best_theta, Phi=self.Phi_train, sqrt_Q_diag=self.sqrt_Q_diag
        )
        self.P_train = self.calculate_P(self.U_train, self.S_train)
        self.U_test = compute_U_RF(
            theta=self.best_theta, Phi=self.Phi_test, sqrt_Q_diag=self.sqrt_Q_diag
        )
        self.P_test = self.calculate_P(self.U_test, self.S_test)

        self.rmse_train = torch.sqrt(
            torch.sum((self.P_train - self.y_train) ** 2) / self.total_train_item
        ).item()
        self.rmse_test = torch.sqrt(
            torch.sum((self.P_test - self.y_test) ** 2) / self.total_test_item
        ).item()

        self.nll_train = (
            -torch.sum(self.y_train * safe_log(self.P_train)).item() / self.N_train
        )
        self.nll_test = (
            -torch.sum(self.y_test * safe_log(self.P_test)).item() / self.N_test
        )

        print("RMSE Train:", self.rmse_train)
        print("RMSE Test:", self.rmse_test)
        print("Train NLL:", self.nll_train)
        print("Test NLL:", self.nll_test)
