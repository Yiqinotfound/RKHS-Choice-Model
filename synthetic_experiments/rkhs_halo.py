import torch
import gc
import time
import os
import csv
from scipy.special import gamma, kv
from HaloDataLoader import HaloDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


eps = torch.tensor(1e-7)
safe_log = lambda x: torch.log(torch.clamp(x, eps, 1.0))


def report_memory(device):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
    reserved_memory = torch.cuda.memory_reserved(device) / (1024**3)
    print(
        f" {device} | Allocated Memory: {allocated_memory:.2f} GB | Reserved Memory: {reserved_memory:.2f} GB"
    )


class RKHSHalo:
    def __init__(
        self,
        dataset: HaloDataset,
        kernel_type: str = "outer",
        kernel_params: dict = {},
        mask: bool = True,
        device: torch.device = "cpu",
    ):
        self.dataset = dataset
        self.train_datasize = self.dataset.train_datasize
        self.test_datasize = self.dataset.test_datasize
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.mask = mask
        self.device = device

        self.product_number_train = self.dataset.S_train.sum().item()
        if self.dataset.test_datasize > 0:
            self.product_number_test = self.dataset.S_test.sum().item()

        print(
            f"Item Number: {self.dataset.item_number}, Set Number: {self.dataset.set_number}"
        )
        print(
            f"Train Data Size: {self.train_datasize}, Test Data Size: {self.test_datasize}"
        )
        if self.kernel_type == "linear" or self.kernel_type == "poly":
            self.compute_kernel = self.compute_poly_kernel
        elif self.kernel_type == "gaussian":
            self.compute_kernel = self.compute_gaussian_kernel
        elif self.kernel_type.endswith("matern"):
            self.compute_kernel = self.compute_matern_kernel

        self.precompute()

    def precompute(self):
        kernel_data_train_train = self.compute_kernel(
            S1_set=self.dataset.S_train,
            S2_set=self.dataset.S_train,
            kernel_params=self.kernel_params,
        )
        kernel_data_test_train = self.compute_kernel(
            S1_set=self.dataset.S_test,
            S2_set=self.dataset.S_train,
            kernel_params=self.kernel_params,
        )
        if self.mask:
            self.masked_kernel_data_train_train = self.mask_kernel_tensor(
                kernel_data_train_train, self.dataset.S_train, self.dataset.S_train
            )
            self.masked_kernel_data_test_train = self.mask_kernel_tensor(
                kernel_data_test_train, self.dataset.S_test, self.dataset.S_train
            )
            del kernel_data_train_train
            del kernel_data_test_train
            gc.collect()
        else:
            self.masked_kernel_data_train_train = kernel_data_train_train
            self.masked_kernel_data_test_train = kernel_data_test_train
        report_memory(self.device)

    def compute_poly_kernel(
        self, S1_set: torch.Tensor, S2_set: torch.Tensor, kernel_params: dict
    ):
        degree = kernel_params["degree"]
        c = kernel_params["c"]
        K = torch.eye(
            self.dataset.item_number,
            device=self.device,
            dtype=torch.float32,
        )
        Kg = torch.mm(S1_set, S2_set.T)
        Kg += c
        Kg = torch.pow(Kg, degree)
        Kg /= torch.max(Kg)
        return  (Kg.unsqueeze(-1).unsqueeze(-1)) * (K.unsqueeze(0).unsqueeze(0))

    def compute_gaussian_kernel(
        self,
        S1_set: torch.Tensor,
        S2_set: torch.Tensor,
        kernel_params: dict,
    ):
        K = self.dataset.cov
        length_scale = kernel_params["length_scale"]
        sigma = kernel_params["sigma"]
        sq_dist = torch.cdist(S1_set, S2_set, p=2) ** 2
        dist = torch.sqrt(sq_dist)
        length_scale = torch.median(dist)
        Kg: torch.Tensor = sigma**2 * torch.exp(-sq_dist / (2 * length_scale**2))

        return (Kg.unsqueeze(-1).unsqueeze(-1)) * (K.unsqueeze(0).unsqueeze(0))

    def compute_matern_kernel(
        self,
        S1_set: torch.Tensor,
        S2_set: torch.Tensor,
        kernel_params: dict,
    ):
        K = self.dataset.cov
        nu = kernel_params["nu"]
        sigma = kernel_params["sigma"]
        length_scale = kernel_params["length_scale"]
        dist = torch.cdist(S1_set, S2_set, p=2)
        dist = torch.where(dist == 0, torch.tensor(1e-7, device=self.device), dist)
        scaled_dist = torch.sqrt(torch.tensor(2 * nu)) * dist / length_scale
        factor = (2 ** (1 - nu)) / gamma(nu)

        scaled_dist = scaled_dist

        Kg: torch.Tensor = (
            sigma**2
            * factor
            * (scaled_dist**nu)
            * torch.as_tensor(
                kv(nu, scaled_dist.cpu().numpy()),
                device=S1_set.device,
                dtype=torch.float32,
            )
        )

        return (Kg.unsqueeze(-1).unsqueeze(-1)) * (K.unsqueeze(0).unsqueeze(0))

    def mask_kernel_tensor(
        self, Kg: torch.Tensor, S_1: torch.Tensor, S_2: torch.Tensor
    ):

        mask1 = (S_1 != 0).unsqueeze(1).unsqueeze(3)
        mask2 = (S_2 != 0).unsqueeze(0).unsqueeze(2)
        mask = mask1 & mask2
        return Kg * mask

    def objective(self):

        U = torch.einsum(
            "jikl, il -> jk", self.masked_kernel_data_train_train, self.alphaset
        )
        if torch.any(torch.isnan(U)):
            print("NaN detected in U")
            print("U: ", U)

        if self.mask:
            loss = self.cross_entropy(
                U=U, S=self.dataset.S_train, y=self.dataset.y_train
            )
        else:
            # print("Using unmasked cross entropy")
            loss = self.cross_entropy_unmasked(U=U, y=self.dataset.y_train)
        # print(loss)
        r = self.regularization()
        # print(r)

        # del U
        # gc.collect()

        return loss + self.lambda_ * r

    def cross_entropy(self, U: torch.Tensor, S: torch.Tensor, y: torch.Tensor):
        # print(U)

        datasize = len(S)
        U_max = torch.max(U, dim=1, keepdim=True).values
        U_stable = U - U_max

        exp_U = torch.exp(U_stable)
        if torch.any(torch.isnan(exp_U)):
            print("NaN detected in exp(U)")
        # print(exp_U)
        exp_U_masked = exp_U * S
        sum_exp_utility = torch.sum(exp_U_masked, dim=1, keepdim=True)
        # print(exp_U_masked)
        # print(sum_exp_utility)
        P = exp_U_masked / (sum_exp_utility)
        if torch.any(torch.isnan(P)):
            print("NaN detected in P")
            print(P)
            # print(sum_exp_utility)
            # print(exp_U_masked)

        log_P = safe_log(P)
        if torch.any(torch.isnan(log_P)):
            print("NaN detected in log_P")

        loss_matrix = -y * log_P
        if torch.any(torch.isnan(loss_matrix)):
            print("NaN detected in loss matrix")

        loss_value = torch.sum(loss_matrix) / datasize
        return loss_value

    def cross_entropy_unmasked(self, U: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.CrossEntropyLoss()(U, y)

    def regularization(self):
        return torch.einsum(
            "id,ijdl,jl->",
            self.alphaset,
            self.masked_kernel_data_train_train,
            self.alphaset,
        )

    def run(
        self,
        alpha_std,
        lr: float = 0.01,
        lambda_: float = 1e-5,
        grad_norm_threshold: float = 1e-3,
    ):
        self.alpha_std = alpha_std
        self.lambda_ = lambda_
        self.grad_norm_threshold = grad_norm_threshold
        self.lr = lr

        self.alphaset = torch.randn(
            (self.train_datasize, self.dataset.item_number),
            dtype=torch.float32,
            device=self.device,
        )
        self.alphaset = (self.alphaset * self.alpha_std).detach().requires_grad_(True)
        self.original_alphaset = self.alphaset.detach().clone()

        self.best_loss = float("inf")
        self.best_alphaset = None
        self.optimizer = torch.optim.Adam([self.alphaset], lr=self.lr)
        self.run_start = time.time()

        for epoch in range(1500):
            self.optimizer.zero_grad()
            loss = self.objective()
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_alphaset = self.alphaset.detach().clone()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_([self.alphaset], max_norm=1.0)
            grad_norm = torch.norm(self.alphaset.grad)
            print(
                "Epoch: ", epoch, "Loss: ", loss.item(), "Grad Norm: ", grad_norm.item()
            )
            if grad_norm < self.grad_norm_threshold:
                print(
                    "Early Stopping at Epoch: ",
                    epoch,
                    "Grad Norm: ",
                    grad_norm.item(),
                    "Loss: ",
                    loss.item(),
                )
                break
            # print(self.alphaset)
            self.optimizer.step()
            # print(self.alphaset)
        self.run_end = time.time()
        self.train_time = self.run_end - self.run_start

    def evaluate_limited(self):
        self.U_train = torch.einsum(
            "jikl,il->jk",
            self.masked_kernel_data_train_train,
            self.best_alphaset,
        )
        
        self.P_train = self.compute_P(self.U_train)
        self.soft_rmse_train = torch.sqrt(
            torch.sum((self.P_train - self.dataset.P_train) ** 2)
            / self.product_number_train
        ).item()
        print("Soft RMSE: ", self.soft_rmse_train)
        
        
    
    def evaluate(self):

        self.U_train = torch.einsum(
            "jikl,il->jk",
            self.masked_kernel_data_train_train,
            self.best_alphaset,
        )
        if self.mask:
            self.P_train = self.compute_P(self.U_train)
            self.nll_train = self.cross_entropy(self.U_train, self.datast.y_train).item()

            self.hard_rmse_train = torch.sqrt(
                torch.sum((self.P_train - self.dataset.y_train) ** 2)
                / self.product_number_train
            ).item()
            self.soft_rmse_train = torch.sqrt(
                torch.sum((self.P_train - self.dataset.F_train) ** 2)
                / self.product_number_train
            ).item()

        else:
            self.P_train = self.compute_P_unmasked(self.U_train)
            self.nll_train = self.cross_entropy_unmasked(self.U_train, self.dataset.y_train).item()
            self.hard_rmse_train = torch.sqrt(
                torch.sum((self.P_train - self.dataset.y_train) ** 2)
                / self.product_number_train
            ).item()
            self.soft_rmse_train = torch.sqrt(
                torch.sum((self.P_train - self.dataset.F_train) ** 2)
                / self.product_number_train
            ).item()


        print(
            "Train NLL: ",
            self.nll_train,
            "Train Hard RMSE: ",
            self.hard_rmse_train,
            "Train Soft RMSE: ",
            self.soft_rmse_train,
        )

    def compute_P(self, U: torch.Tensor):
        U_max = torch.max(U, dim=1, keepdim=True).values
        U_stable = U - U_max
        exp_U = torch.exp(U_stable)
        exp_U_masked = exp_U * self.dataset.S_train
        sum_exp_U = torch.sum(exp_U_masked, dim=1, keepdim=True)
        P = exp_U_masked / sum_exp_U
        return P

    def compute_P_unmasked(self, U: torch.Tensor):
        U_max = torch.max(U, dim=1, keepdim=True).values
        U_stable = U - U_max
        exp_U = torch.exp(U_stable)
        sum_exp_U = torch.sum(exp_U, dim=1, keepdim=True)
        P = exp_U / sum_exp_U
        return P

    def retrive_base_utility(self):
        base_sets = self.dataset.base_S
        base_kernel = self.compute_kernel(
            S1_set=base_sets,
            S2_set=self.dataset.S_train,
            kernel_params=self.kernel_params,
        )
        masked_base_kernel = self.mask_kernel_tensor(
            base_kernel, base_sets, self.dataset.S_train
        )
        self.base_utility = torch.einsum(
            "jikl,il->jk",
            masked_base_kernel,
            self.best_alphaset,
        )
        self.base_utility = self.base_utility.diagonal()
        print(self.base_utility)

    def retrive_one_order_interaction(self):
        one_order_S = self.dataset.one_order_S
        ond_order_kernel = self.compute_kernel(
            S1_set=one_order_S,
            S2_set=self.dataset.S_train,
            kernel_params=self.kernel_params,
        )
        masked_one_order_kernel = self.mask_kernel_tensor(
            ond_order_kernel, one_order_S, self.dataset.S_train
        )
        self.one_order_utility = torch.einsum(
            "jikl,il->jk",
            masked_one_order_kernel,
            self.best_alphaset,
        )
        print(self.one_order_utility)
        interaction = torch.zeros(
            self.dataset.item_number, self.dataset.item_number, device=self.device
        )
        for i in range(len(self.one_order_utility)):
            # 找出 one_order_utility[i]非零的两个位置
            for j in range(self.dataset.item_number):
                if one_order_S[i, j] != 0:
                    break
            for k in range(j + 1, self.dataset.item_number):
                if one_order_S[i, k] != 0:
                    break

            interaction[j, k] = self.one_order_utility[i, k] - self.base_utility[k]
            interaction[k, j] = self.one_order_utility[i, j] - self.base_utility[j]
            print(j, k, interaction[j, k])
        return interaction

    def save_results(self, filepath: str):
        results = [
            [
                self.kernel_type,
                self.kernel_params,
                self.mask,
                self.nll_train,
                self.hard_rmse_train,
                self.soft_rmse_train,
                self.lambda_,
                self.alpha_std,
                self.lr,
                self.train_time,
            ]
        ]
        file_exists = os.path.exists(filepath)
        with open(filepath, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists or os.stat(filepath).st_size == 0:
                writer.writerow(
                    [
                        "kernel_type",
                        "kernel_params",
                        "mask",
                        "nll",
                        "hard_rmse",
                        "soft_rmse",
                        "lambda",
                        "alpha_std",
                        "lr",
                        "train_time",
                    ]
                )
            writer.writerows(results)
