import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
from context_benchmark_models.TCNet import TransformerChoiceNet
from utils.model_utils import load_config, save_to_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from data_preprocess.swissmetros_data_loader import SwissMetroDataset


def cross_validate(dataset: SwissMetroDataset, model_args: dict):
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

        model = TransformerChoiceNet(model_args=model_args, device=device)
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
config_path = "configs/benchmark_config.yaml"

swiss_path = "data/swissmetro/swissmetro.dat"
model_args = load_config(config_path=config_path)["swissmetro_TCNet"]
save_path = (
    "benchmark_experiments/swissmetro_TCNet_results/swissmetro_TCNet_results.csv"
)


dataset = SwissMetroDataset(
    filepath=swiss_path, preprocess_mode=model_args["preprocess_mode"]
)

datasize = len(dataset)
input_dim = dataset.X.shape[-1]
model_args["input_dim"] = int(input_dim)
hidden_dim_list = [128, 256, 512]
for hidden_dim in hidden_dim_list:
    model_args["hidden_dim"] = hidden_dim
    cross_validate(dataset=dataset, model_args=model_args)

# print(model_args)
# print(dataset.X.shape)
