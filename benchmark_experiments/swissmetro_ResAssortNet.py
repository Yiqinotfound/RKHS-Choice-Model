import sys, os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from context_benchmark_models.ResAssortNet import ResAssortNet
from data_preprocess.swissmetros_data_loader import SwissMetroDataset
from utils.creterias_and_loss_utils import compute_mask_from_card
from utils.model_utils import load_config
from torch.utils.data import dataloader, TensorDataset
from sklearn.model_selection import train_test_split, KFold


def data_preprocess(dataset: SwissMetroDataset) -> tuple[torch.Tensor, torch.Tensor]:
    X_cust = torch.tensor(np.array(dataset.cust_df), dtype=torch.float32)
    X_prod = torch.tensor(np.array(dataset.trans_df), dtype=torch.float32)
    X_last_nine_dynamic_features = X_prod[:, -9:]
    X_cust = torch.cat((X_cust, X_last_nine_dynamic_features), axis=-1)

    products = torch.eye(dataset.d, dtype=torch.float32)
    assort = compute_mask_from_card(cardinality=dataset.cardinality, d=dataset.d).to(
        dtype=torch.float32
    )

    data = torch.cat((assort, X_cust), axis=-1)

    return products, data


def cross_validate(
    data: torch.Tensor, products: torch.Tensor, original_dataset: SwissMetroDataset
):
    datasize = len(data)
    temp_idx, test_idx = train_test_split(
        np.arange(datasize),
        test_size=model_args["test_size"],  # First split 90-10
        random_state=model_args["SEED"],
    )
    kf = KFold(
        n_splits=model_args["n_splits"],
        shuffle=True,
        random_state=model_args["SEED"],
    )

    fold = 0

    for train_idx, val_idx in kf.split(temp_idx):
        data_train = data[train_idx]
        data_val = data[val_idx]
        data_test = data[test_idx]

        y_train = original_dataset.y[train_idx]
        y_val = original_dataset.y[val_idx]
        y_test = original_dataset.y[test_idx]

        dataset_train = (data_train, y_train)
        dataset_val = (data_val, y_val)
        dataset_test = (data_test, y_test)

        model = ResAssortNet(model_args=model_args, products=products, device=device)
        model.fit(
            dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test
        )
        return 


device = torch.device("cuda")



swiss_path = "data/swissmetro/swissmetro.dat"
config_path = "configs/benchmark_config.yaml"

dataset = SwissMetroDataset(filepath=swiss_path, preprocess_mode="rumnet")
model_args = load_config(config_path=config_path)["swissmetro_ResAssortNet"]

products, data = data_preprocess(dataset=dataset)

cross_validate(data=data, products=products, original_dataset=dataset)
# model = ResAssortNet(model_args=model_args, products=products)
