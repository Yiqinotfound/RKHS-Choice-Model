import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import time
from utils.model_utils import *
from utils.creterias_and_loss_utils import (
    compute_U,
    compute_P,
    compute_mask_from_card,
    nll,
    accuracy,
    regularization,
    cross_entropy,
)
from utils.kernel_utils import (
    mask_kernel_tensor_FB,
    compute_gaussian_kernel_tensor_FB,
    compute_matern_kernel_tensor_FB,
)


from sklearn.model_selection import KFold
from tqdm import tqdm


class FKFBModel:
    def __init__(
        self,
        dataset_train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dataset_val: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dataset_test: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        model_args: dict,
        kernel_args: dict,
        precompute_device: torch.device,
        train_device: torch.device,
    ):

        self.model_args = model_args
        self.kernel_args = kernel_args

        self.X_train, self.y_train, self.cardinality_train = dataset_train
        self.X_val, self.y_val, self.cardinality_val = dataset_val
        self.X_test, self.y_test, self.cardinality_test = dataset_test

        self.X_train = self.X_train.to(precompute_device)
        self.X_val = self.X_val.to(precompute_device)
        self.X_test = self.X_test.to(precompute_device)
        self.y_train = self.y_train.to(precompute_device)
        self.y_val = self.y_val.to(precompute_device)
        self.y_test = self.y_test.to(precompute_device)
        self.cardinality_train = self.cardinality_train.to(precompute_device)
        self.cardinality_val = self.cardinality_val.to(precompute_device)
        self.cardinality_test = self.cardinality_test.to(precompute_device)

        # max cardinality
        self.d = self.X_train.shape[1]

        # compute mask tensors
        self.mask_train = compute_mask_from_card(
            cardinality=self.cardinality_train, d=self.d
        )
        self.mask_val = compute_mask_from_card(
            cardinality=self.cardinality_val, d=self.d
        )
        self.mask_test = compute_mask_from_card(
            cardinality=self.cardinality_test, d=self.d
        )

        self.train_datasize = self.X_train.shape[0]
        self.val_datasize = self.X_val.shape[0]
        self.test_datasize = self.X_test.shape[0]

        # devices
        self.precompute_device = precompute_device
        self.train_device = train_device

        # precompute kernel tensor
        self.kernel_type: str = self.model_args["kernel_type"]
        self.kernel_params = self.kernel_args[self.kernel_type]
        self.random_seed = self.model_args["SEED"]

        self.kernel_params_str = "_".join(
            [f"{key}={value}" for key, value in self.kernel_params.items()]
        )

        self.compute_kernel_tensor = self.select_precompute_function()

        self.lambda_ = self.model_args["lambda"]
        self.patience = self.model_args["patience"]
        self.lr = self.model_args["learning_rate"]
        self.alpha_std = self.model_args["alpha_std"]
        self.smoothing = self.model_args["smoothing"]

        # smoothed labels
        self.y_train_smooth = (
            1 - self.smoothing
        ) * self.y_train + self.smoothing / self.d
        self.y_val_smooth = (1 - self.smoothing) * self.y_val + self.smoothing / self.d
        self.y_test_smooth = (
            1 - self.smoothing
        ) * self.y_test + self.smoothing / self.d

    def select_precompute_function(self):
        if self.kernel_type == "gaussian":
            return compute_gaussian_kernel_tensor_FB
        elif self.kernel_type.endswith("matern"):
            return compute_matern_kernel_tensor_FB
        else:
            raise ValueError("Invalid kernel type")

    def precompute(self):
        print("Begin Precomputing Kernel Tensors...")

        precompute_start = time.time()

        print(f"Precomputing kernel tensor for train set...")
        self.kernel_tensor_train = self.compute_kernel_tensor(
            X=self.X_train,
            Y=self.X_train,
            kernel_params=self.kernel_params,
            batch_size=self.model_args["precompute_batch_size"],
        )
        t1 = time.time()
        self.precompute_time_train = t1 - precompute_start
        print(f"Precompute time for train set: {self.precompute_time_train}s")
        print(f"Precomputing kernel tensor for val set...")
        self.kernel_tensor_val = self.compute_kernel_tensor(
            X=self.X_val,
            Y=self.X_train,
            kernel_params=self.kernel_params,
            batch_size=self.model_args["precompute_batch_size"],
        )
        t2 = time.time()
        self.precompute_time_val = t2 - t1
        print(f"Precompute time for val set: {self.precompute_time_val}s")
        print(f"Precomputing kernel tensor for test set...")
        self.kernel_tensor_test = self.compute_kernel_tensor(
            X=self.X_test,
            Y=self.X_train,
            kernel_params=self.kernel_params,
            batch_size=self.model_args["precompute_batch_size"],
        )
        t3 = time.time()
        self.precompute_time_test = t3 - t2
        print(f"Precompute time for test set: {self.precompute_time_test}s")

        self.kernel_tensor_train = mask_kernel_tensor_FB(
            kernel_tensor=self.kernel_tensor_train,
            cardinality_1=self.cardinality_train,
            cardinality_2=self.cardinality_train,
        )
        self.kernel_tensor_val = mask_kernel_tensor_FB(
            kernel_tensor=self.kernel_tensor_val,
            cardinality_1=self.cardinality_val,
            cardinality_2=self.cardinality_train,
        )
        self.kernel_tensor_test = mask_kernel_tensor_FB(
            kernel_tensor=self.kernel_tensor_test,
            cardinality_1=self.cardinality_test,
            cardinality_2=self.cardinality_train,
        )

        self.precompute_time = time.time() - precompute_start
        print("Precompute Completed in ", self.precompute_time, "s.")
        report_memory(self.precompute_device)

    def objective(self):
        U = compute_U(alphaset=self.alphaset, kernel_tensor=self.kernel_tensor_train)
        nll = cross_entropy(U=U, y=self.y_train, mask_tensor=self.mask_train)
        reg = regularization(
            alphaset=self.alphaset, kernel_tensor=self.kernel_tensor_train
        )
        return nll + self.lambda_ * reg

    def fit(self):

        # move data to train device
        self.kernel_tensor_train = self.kernel_tensor_train.to(self.train_device)
        self.kernel_tensor_val = self.kernel_tensor_val.to(self.train_device)
        self.kernel_tensor_test = self.kernel_tensor_test.to(self.train_device)
        self.mask_train = self.mask_train.to(self.train_device)
        self.mask_val = self.mask_val.to(self.train_device)
        self.mask_test = self.mask_test.to(self.train_device)
        self.y_train_smooth = self.y_train_smooth.to(self.train_device)
        self.y_val_smooth = self.y_val_smooth.to(self.train_device)
        self.y_test_smooth = self.y_test_smooth.to(self.train_device)
        self.y_train = self.y_train.to(self.train_device)
        self.y_val = self.y_val.to(self.train_device)
        self.y_test = self.y_test.to(self.train_device)
        print(f"Data moved to train device successfully")

        report_memory(self.train_device)

        # record time
        run_start = time.time()

        # initialize alphaset
        self.alphaset = torch.randn(
            (self.train_datasize, self.d),
            device=self.train_device,
            dtype=torch.float32,
        )
        self.alphaset = (self.alphaset * self.alpha_std).detach().requires_grad_(True)

        # intialize best loss, best alphaset
        self.best_val_loss = float("inf")
        self.best_alphaset = None

        # val loss patience
        self.patience_counter = 0

        if self.model_args["optimizer"] == "LBFGS":
            self.fit_with_LBFGS()
        elif self.model_args["optimizer"] == "Adam":
            self.fit_with_Adam()
        else:
            raise ValueError("Invalid optimizer")

        self.train_time = time.time() - run_start

        self.optimizer.state.clear()
        del self.optimizer
        report_memory(self.train_device)

    def fit_with_LBFGS(self):
        """
        fit the model with LBFGS optimizer
        """
        self.optimizer = torch.optim.LBFGS([self.alphaset], lr=self.lr)

        def closure():
            self.optimizer.zero_grad()
            loss_value = self.objective()
            loss_value.backward()
            return loss_value

        with tqdm(total=self.model_args["max_epochs"], desc="Training", unit="epoch") as pbar:
            for epoch in range(self.model_args["max_epochs"]):
                self.optimizer.step(closure)
                self.evaluate(self.alphaset)
                if self.nll_val < self.best_val_loss:
                    self.best_val_loss = self.nll_val
                    self.best_alphaset = self.alphaset.clone().detach()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                pbar.set_postfix(
                    {
                        "Train Loss": self.nll_train,
                        "Val Loss": self.nll_val,
                        "Test Loss": self.nll_test,
                        "Patience": self.patience_counter,
                    }
                )
                if self.patience_counter > self.patience:
                    pbar.close()
                    break
                pbar.update(1)
    def fit_with_Adam(self):

        # use Adam optimizer
        self.optimizer = torch.optim.Adam([self.alphaset], lr=self.lr)

        with tqdm(total=self.model_args["max_epochs"], desc="Training", unit="epoch") as pbar:
            for epoch in range(self.model_args["max_epochs"]):
                self.optimizer.zero_grad()
                loss = self.objective()
                self.evaluate(self.alphaset)
                if self.nll_val < self.best_val_loss:
                    self.best_val_loss = self.nll_val
                    self.best_alphaset = self.alphaset.clone().detach()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                pbar.set_postfix(
                    {
                        "Train Loss": self.nll_train,
                        "Val Loss": self.nll_val,
                        "Test Loss": self.nll_test,
                        "Patience": self.patience_counter,
                    }
                )
                if self.patience_counter > self.patience:
                    pbar.close()
                    break
                loss.backward()
                self.optimizer.step()
                pbar.update(1)

    def evaluate(self, alphaset):

        with torch.no_grad():
            # compute U
            self.U_train = compute_U(
                alphaset=alphaset, kernel_tensor=self.kernel_tensor_train
            )
            self.U_val = compute_U(
                alphaset=alphaset, kernel_tensor=self.kernel_tensor_val
            )
            self.U_test = compute_U(
                alphaset=alphaset, kernel_tensor=self.kernel_tensor_test
            )

            # compute P
            self.P_train = compute_P(U=self.U_train, mask_tensor=self.mask_train)
            self.P_val = compute_P(U=self.U_val, mask_tensor=self.mask_val)
            self.P_test = compute_P(U=self.U_test, mask_tensor=self.mask_test)

            # nll
            self.nll_train = nll(P=self.P_train, y=self.y_train_smooth)
            self.nll_val = nll(P=self.P_val, y=self.y_val_smooth)
            self.nll_test = nll(P=self.P_test, y=self.y_test_smooth)

            # accuracy
            self.acc_train = accuracy(P=self.P_train, Y=self.y_train)
            self.acc_val = accuracy(P=self.P_val, Y=self.y_val)
            self.acc_test = accuracy(P=self.P_test, Y=self.y_test)
