import sys,os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from scipy.special import gamma, kv
import torch
import time
import csv, os

from utils.creterias_and_loss_utils import *
from utils.model_utils import *
from utils.kernel_utils import *

from data_preprocess.hotel_data_loader import HotelDataset


class FKFFModel:
    def __init__(
        self,
        dataset: HotelDataset,
        kernel_type: str,
        kernel_params: dict,
        mask: bool,
        precompute_device,
        train_device,
    ):

        self.dataset = dataset
        self.instance_id = self.dataset.instance_id
        self.d = self.dataset.d
        self.train_datasize = self.dataset.train_datasize
        self.test_datasize = self.dataset.test_datasize

        self.S_train = self.dataset.S_train
        self.y_train = self.dataset.y_train

        self.S_test = self.dataset.S_test
        self.y_test = self.dataset.y_test

        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.kernel_params_str = "_".join(
            [f"{key}={value}" for key, value in self.kernel_params.items()]
        )

        # whether to mask the kernel tensor
        self.mask = mask

        # select the precompute function based on the kernel type
        self.compute_kernel_tensor = self.select_precompute_function()

        # device
        self.precompute_device = precompute_device
        self.train_device = train_device

        # K is set to the identity matrix by default
        self.K = torch.eye((self.d), dtype=torch.float16, device=self.precompute_device)

        self.precompute()

    def select_precompute_function(self):
        """select precompute function based on the kernel type

        Raises:
            ValueError: Wrong kernel type

        Returns:
            _type_: func
        """

        if self.kernel_type == "matern":
            return compute_matern_kernel_tensor_FF
        elif self.kernel_type == "gaussian":
            return compute_gaussian_kernel_tensor_FF
        elif self.kernel_type == "poly":
            return compute_polynomial_kernel_tensor_FF
        elif self.kernel_type == "linear":
            return compute_polynomial_kernel_tensor_FF
        else:
            raise ValueError("Invalid kernel type")

    def precompute(self):
        """precompute kernel tensors and mask them (if self.mask = True)"""
        print("Begin Precomputing Kernel Tensor.")

        # precompute the diagonal kernel tensor
        self.precompute_start = time.time()
        self.kernel_tensor_train = self.compute_kernel_tensor(
            self.S_train, self.S_train, self.kernel_params, self.K
        )
        self.kernel_tensor_test = self.compute_kernel_tensor(
            self.S_test, self.S_train, self.kernel_params, self.K
        )

        # mask
        if self.mask:
            self.kernel_tensor_train = mask_kernel_tensor_FF(
                self.kernel_tensor_train, self.S_train, self.S_train
            )
            self.kernel_tensor_test = mask_kernel_tensor_FF(
                self.kernel_tensor_test, self.S_test, self.S_train
            )
        self.precompute_end = time.time()
        self.precompute_time = self.precompute_end - self.precompute_start
        print("Precompute Completed in ", self.precompute_time, "s.")
        self.move_data_semi_precision()
        print("Data Moved to Train Device.")
        report_memory(self.train_device)

    def move_data_semi_precision(self):
        """move the data to self.train_device and convert the kernel tensors to semi-precison"""
        self.S_train = self.S_train.to(self.train_device)
        self.S_test = self.S_test.to(self.train_device)
        self.y_train = self.y_train.to(self.train_device)
        self.y_test = self.y_test.to(self.train_device)
        self.kernel_tensor_train = self.kernel_tensor_train.to(
            self.train_device, dtype=torch.float16
        )
        self.kernel_tensor_test = self.kernel_tensor_test.to(
            self.train_device, dtype=torch.float16
        )

    def train_param_init(
        self,
        lambda_: float,
        grad_norm_threshold: float,
        alpha_std: float,
        lr: float,
        batch_size: int,
    ):
        """initialize the training parameters

        Args:
            lambda_ (float): reg param
            grad_norm_threshold (float): grad norm threshold for early stopping
            alpha_std (float): alpha intializing std
            lr (float): learning rate
        """
        self.lambda_ = lambda_
        self.grad_norm_threshold = grad_norm_threshold
        self.alpha_std = alpha_std
        self.lr = lr
        self.batch_size = batch_size

    def run(self):
        """train function"""

        # record the start time of the training process
        self.run_start = time.time()

        # initialize the alpha set
        self.alphaset = torch.randn(
            (self.train_datasize, self.d), dtype=torch.float32, device=self.train_device
        )
        self.alphaset = (self.alphaset * self.alpha_std).detach().requires_grad_(True)

        # initialize the best loss and best alpha set
        self.best_loss = float("inf")
        self.best_alphaset = None

        # initialize the optimizer and scaler
        self.scaler = torch.amp.GradScaler()
        self.optimizer = torch.optim.Adam([self.alphaset], lr=self.lr)

        # train the model
        for epoch in range(50):
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda"):
                loss = self.objective()

            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_alphaset = self.alphaset.clone().detach()

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)

            grad_norm = torch.norm(self.alphaset.grad).item()
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}, Best Loss: {self.best_loss}, Grad Norm: {grad_norm}"
                )
            if grad_norm < self.grad_norm_threshold:
                print(
                    f"Early stopping at epoch {epoch}, Gradient Norm: {grad_norm}, Best Loss: {self.best_loss}"
                )
                break

            self.scaler.update()

        self.run_end = time.time()
        self.train_time = self.run_end - self.run_start

        self.optimizer.state.clear()

        del self.optimizer
        del self.scaler
        report_memory(self.train_device)

    def objective(self):
        """the objective function, including cross entropy loss and the regularization"""

        U = compute_U(alphaset=self.alphaset, kernel_tensor=self.kernel_tensor_train)
        if torch.any(torch.isnan(U)):
            print("NaN detected in U")

        nll = cross_entropy_FF(U=U, S=self.S_train, y=self.y_train, mask=True)
        reg = regularization_with_batch(
            alphaset=self.alphaset,
            kernel_tensor=self.kernel_tensor_train,
            batch_size=self.batch_size,
        )

        return nll + self.lambda_ * reg

    def evaluate(self):
        """evalute function"""

        # compute U
        self.U_train = compute_U(
            alphaset=self.alphaset.to(self.kernel_tensor_train.dtype),
            kernel_tensor=self.kernel_tensor_train,
        )
        self.U_test = compute_U(
            alphaset=self.alphaset.to(self.kernel_tensor_train.dtype),
            kernel_tensor=self.kernel_tensor_test,
        )

        # compute P
        self.P_train = compute_P_FF(
            self.U_train.to(self.S_train.dtype), self.S_train, self.mask
        )
        self.P_test = compute_P_FF(
            self.U_test.to(self.S_test.dtype), self.S_test, self.mask
        )

        # compute hard rmse
        self.hard_rmse_train = rmse(
            P=self.P_train, y=self.y_train, total_item=self.dataset.total_train_item
        )
        self.hard_rmse_test = rmse(
            P=self.P_test, y=self.y_test, total_item=self.dataset.total_test_item
        )

        # compute nll
        self.nll_train = nll(P=self.P_train, y=self.y_train)
        self.nll_test = nll(P=self.P_test, y=self.y_test)

        print(f"Hard Train RMSE: {self.hard_rmse_train}")
        print(f"Hard Test RMSE: {self.hard_rmse_test}")
        print(f"Train NNL: {self.nll_train}")
        print(f"Test NNL: {self.nll_test}")
