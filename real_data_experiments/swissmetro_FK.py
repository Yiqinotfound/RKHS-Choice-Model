import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
from data_preprocess.swissmetros_data_loader import SwissMetroDataset
from models.FK_feature_based_model import FKFBModel
from utils.model_utils import load_config, save_to_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np


def cross_validate(
    dataset: SwissMetroDataset,
    model_args: dict,
    kernel_args: dict,
):
    datasize = dataset.X.shape[0]
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

    dataset_test = dataset[test_idx]
    for fold, (train_idx, val_idx) in enumerate(kf.split(temp_idx)):
        dataset_train = dataset[train_idx]
        dataset_val = dataset[val_idx]

        model = FKFBModel(
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test,
            model_args=model_args,
            kernel_args=kernel_args,
            precompute_device=precompute_device,
            train_device=train_device,
        )
        model.precompute()
        model.fit()
        data_rows = [
            [fold + 1]
            + list(model_args.values())
            + [model.kernel_params_str]
            + [
                model.nll_train,
                model.nll_val,
                model.nll_test,
                model.acc_train,
                model.acc_val,
                model.acc_test,
                model.train_time,
                model.precompute_time,
                model.precompute_time_train,
                model.precompute_time_val,
                model.precompute_time_test,
            ]
        ]
        header_columns = (
            ["fold"]
            + list(model_args.keys())
            + ["kernel_params"]
            + ["nll_train", "nll_val", "nll_test", "acc_train", "acc_val", "acc_test"]
            + [
                "train_time",
                "precompute_time",
                "precompute_time_train",
                "precompute_time_val",
                "precompute_time_test",
            ]
        )
        save_to_csv(data_rows, header_columns, save_path)


if __name__ == "__main__":
    # precompute_device = torch.device("cpu")
    precompute_device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    train_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data_path = "data/swissmetro/swissmetro.dat"
    exp_config_path = "configs/exp_config.yaml"
    kernel_config_path = "configs/swissmetro_kernel_config.yaml"
    save_path = "real_data_experiments/swissmetro_FK_results/swissmetro_FK_results.csv"
    model_args = load_config(exp_config_path)["swiss_metro_fk_args"]
    kernel_args = load_config(kernel_config_path)
    dataset = SwissMetroDataset(
        filepath=data_path,
        preprocess_mode="rumnet",
    )

    datasize = dataset.X.shape[0]
    print(f"Has total {datasize} samples")

    cross_validate(
        dataset=dataset,
        model_args=model_args,
        kernel_args=kernel_args,
    )
