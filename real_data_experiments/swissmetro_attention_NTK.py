import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
from models.AttentionNTKModel import AttentionNTKChoiceModel
from utils.model_utils import load_config, save_to_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from data_preprocess.swissmetros_data_loader import SwissMetroDataset


def cross_validate(dataset: SwissMetroDataset, model_config: dict):
    temp_idx, test_idx = train_test_split(
        np.arange(datasize),
        test_size=model_config["test_size"],  # First split 90-10
        random_state=model_config["SEED"],
    )
    kf = KFold(
        n_splits=model_config["n_splits"],
        shuffle=True,
        random_state=model_config["SEED"],
    )

    nll_train_log = []
    nll_val_log = []
    nll_test_log = []
    acc_train_log = []
    acc_val_log = []
    acc_test_log = []

    fold = 0
    for train_idx, val_idx in kf.split(temp_idx):
        dataset_train = dataset[train_idx]
        dataset_val = dataset[val_idx]
        dataset_test = dataset[test_idx]

        model = AttentionNTKChoiceModel(
            d=d, d0=d0, d2=d2, model_config=model_config
        ).to(device)
        model.fit(dataset_train, dataset_val, dataset_test, device)
        model.evaluate(device)

        nll_train_log.append(model.nll_train)
        nll_val_log.append(model.nll_val)
        nll_test_log.append(model.nll_test)
        acc_train_log.append(model.acc_train)
        acc_val_log.append(model.acc_val)
        acc_test_log.append(model.acc_test)
        fold += 1

        data_rows = [
            [fold]
            + list(model_config.values())
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
            + list(model_config.keys())
            + ["nll_train", "nll_val", "nll_test", "acc_train", "acc_val", "acc_test", "train_time"]
        )
        save_to_csv(
            data_rows, header_columns, save_path
        )



device = torch.device("cuda")
config_path = "configs/exp_config.yaml"
swiss_path = "data/swissmetro/swissmetro.dat"
model_config = load_config(config_path=config_path)["attention_ntk_model_args"]
save_path = "real_data_experiments/swissmetro_NTK_results/swissmetro_NTK_results.csv"

dataset = SwissMetroDataset(
    filepath=swiss_path, preprocess_mode=model_config["preprocess_mode"]
)
datasize = len(dataset)

d = dataset.feature_vec_length
d0 = dataset.feature_vec_length
d2 = dataset.feature_vec_length
print(f"Has total {datasize} samples.")

cross_validate(dataset=dataset, model_config=model_config)
