import sys, os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from context_benchmark_models.dlcl import DLCL
from data_preprocess.swissmetros_data_loader import SwissMetroDataset
from utils.model_utils import load_config, save_to_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np


def cross_validate(dataset: SwissMetroDataset, model_args: dict):
    datasize = len(dataset)
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
        dataset_train = dataset[train_idx]
        dataset_val = dataset[val_idx]
        dataset_test = dataset[test_idx]

        model = DLCL(num_features=dataset.feature_vec_length, model_args=model_args).to(
            device
        )
        # print(next(model.parameters()).device)
        model.fit(dataset_train, dataset_val, dataset_test, device)
        model.evaluate(device=device)

        fold += 1

        data_rows = [
            [fold]
            + list(model_args.values())
            + [
                model.nll_train,
                model.nll_val,
                model.nll_test,
                model.acc_train,
                model.acc_val,
                model.acc_test,
                model.train_time,
            ]
        ]

        header_columns = (
            ["fold"]
            + list(model_args.keys())
            + [
                "nll_train",
                "nll_val",
                "nll_test",
                "acc_train",
                "acc_val",
                "acc_test",
                "train_time",
            ]
        )
        save_to_csv(data_rows, header_columns, save_path)


device = torch.device("cuda")


swiss_path = "data/swissmetro/swissmetro.dat"
config_path = "configs/benchmark_config.yaml"
save_path = "benchmark_experiments/swissmetro_DLCL_results/swissmetro_DLCL_results.csv"

dataset = SwissMetroDataset(filepath=swiss_path, preprocess_mode="rumnet")
model_args = load_config(config_path=config_path)["swissmetro_DLCL"]
# n_features = dataset.feature_vec_length

# # model = FATEScoring(n_features=n_features)
# # utils = model(dataset.X)
# # print(utils)
# # print(utils.shape)

# model = DLCL(num_features=dataset.feature_vec_length, model_args=model_args)
# print(model(dataset.X, dataset.cardinality)[9])
cross_validate(dataset=dataset, model_args=model_args)
# print(model_args)
