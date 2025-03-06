import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from scipy.special import gamma, kv
import torch
import time
import csv, os

from utils.creterias_and_loss_utils import *
from utils.model_utils import *
from utils.kernel_utils import *

from data_preprocess.hotel_data_loader import HotelDataset
from tqdm import tqdm


class FKFFModel:
    def __init__(
        self,
        dataset: HotelDataset,
        model_args: dict,
        kernel_args: dict,
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

        self.kernel_type = model_args["kernel_type"]
        self.kernel_params = kernel_args[self.kernel_type]
        self.kernel_params_str = "_".join(
            [f"{key}={value}" for key, value in self.kernel_params.items()]
        )

        self.model_args = model_args
        self.kernel_args = kernel_args

        # select the precompute function based on the kernel type
        self.compute_kernel_tensor = self.select_precompute_function()

        # device
        self.device = train_device

        # training parameters
        self.lambda_ = self.model_args["lambda"]
        self.grad_norm_threshold = self.model_args["grad_norm_threshold"]
        self.alpha_std = self.model_args["alpha_std"]
        self.lr = self.model_args["learning_rate"]
        self.reg_batch = self.model_args["reg_batch"]

        # K is set to the identity matrix by default
        self.K = torch.eye((self.d))

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

        print("Precomputing Kernel Tensor.")

        # precompute the diagonal kernel tensor
        self.precompute_start = time.time()
        self.kernel_tensor_train = self.compute_kernel_tensor(
            self.S_train, self.S_train, self.kernel_params, self.K
        )
        self.kernel_tensor_test = self.compute_kernel_tensor(
            self.S_test, self.S_train, self.kernel_params, self.K
        )

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
        print("Data Moved to GPU.")
        report_memory(self.device)

    def move_data_semi_precision(self):
        """move the data to self.train_device and convert the kernel tensors to semi-precison"""
        self.S_train = self.S_train.to(self.device)
        self.S_test = self.S_test.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.y_test = self.y_test.to(self.device)
        self.kernel_tensor_train = self.kernel_tensor_train.to(
            self.device, dtype=torch.float16
        )
        self.kernel_tensor_test = self.kernel_tensor_test.to(
            self.device, dtype=torch.float16
        )

    def train_param_init(self):
        """initialize the training parameters

        Args:
            lambda_ (float): reg param
            grad_norm_threshold (float): grad norm threshold for early stopping
            alpha_std (float): alpha intializing std
            lr (float): learning rate
        """

    def fit(self):
        """train function"""

        # record the start time of the training process
        self.run_start = time.time()

        # initialize the alpha set
        self.alphaset = torch.randn(
            (self.train_datasize, self.d), dtype=torch.float32, device=self.device
        )
        self.alphaset = (self.alphaset * self.alpha_std).detach().requires_grad_(True)

        # initialize the best loss and best alpha set
        self.best_loss = float("inf")
        self.best_alphaset = None

        # initialize the optimizer and scaler
        self.scaler = torch.amp.GradScaler()
        self.optimizer = torch.optim.Adam([self.alphaset], lr=self.lr)

        # train the model
        with tqdm(range(self.model_args["max_epochs"])) as pbar:
            for epoch in range(self.model_args["max_epochs"]):
                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda"):
                    loss = self.objective()

                if loss.item() < self.best_loss:
                    self.best_loss = loss.item()
                    self.best_alphaset = self.alphaset.clone().detach()

                self.evaluate(self.alphaset)
                
                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)

                grad_norm = torch.norm(self.alphaset.grad).item()
                if grad_norm < self.grad_norm_threshold:
                    print(
                        f"Early stopping at epoch {epoch}, Gradient Norm: {grad_norm}, Best Loss: {self.best_loss}"
                    )
                    break
                
                pbar.set_postfix({"Train Loss": self.nll_train,  "Test Loss": self.nll_test, "Grad Norm": grad_norm})
                pbar.update(1)

                self.scaler.update()

        self.run_end = time.time()
        self.train_time = self.run_end - self.run_start

        self.optimizer.state.clear()

        del self.optimizer
        del self.scaler

    def objective(self):
        """the objective function, including cross entropy loss and the regularization"""

        U = compute_U(alphaset=self.alphaset, kernel_tensor=self.kernel_tensor_train)
        if torch.any(torch.isnan(U)):
            print("NaN detected in U")

        nll = cross_entropy_FF(U=U, S=self.S_train, y=self.y_train, mask=True)
        reg = regularization_with_batch(
            alphaset=self.alphaset,
            kernel_tensor=self.kernel_tensor_train,
            batch_size=self.reg_batch,
        )

        return nll + self.lambda_ * reg

    def evaluate(self, alphaset: torch.Tensor):
        """evalute function"""
        with torch.no_grad():
            # compute U
            self.U_train = compute_U(
                alphaset=alphaset.to(self.kernel_tensor_train.dtype),
                kernel_tensor=self.kernel_tensor_train,
            )
            self.U_test = compute_U(
                alphaset=alphaset.to(self.kernel_tensor_train.dtype),
                kernel_tensor=self.kernel_tensor_test,
            )

            # compute P
            self.P_train = compute_P(
                U=self.U_train.to(self.S_train.dtype), mask_tensor=self.S_train
            )
            self.P_test = compute_P(
                U=self.U_test.to(self.S_test.dtype), mask_tensor=self.S_test
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
