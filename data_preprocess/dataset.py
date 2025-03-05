import torch


class FBDataset:
    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        cardinality_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        cardinality_val: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        cardinality_test: torch.Tensor,
        device: torch.device,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.cardinality_train = cardinality_train

        self.X_val = X_val
        self.y_val = y_val
        self.cardinality_val = cardinality_val

        self.X_test = X_test
        self.y_test = y_test
        self.cardinality_test = cardinality_test

        self.train_datasize = self.X_train.shape[0]
        self.val_datasize = self.X_val.shape[0]
        self.test_datasize = self.X_test.shape[0]
        
        self.d = self.X_train.shape[1]
        self.feature_length = self.X_train.shape[2]

        self.device = device
        
