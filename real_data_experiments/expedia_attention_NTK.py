import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from data_preprocess.expedia_data_loader import ExpediaDataset
from models.AttentionNTKModel import AttentionNTKChoiceModel
from utils.model_utils import report_memory, load_config
from sklearn.model_selection import KFold
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from utils.creterias_and_loss_utils import compute_mask_from_card


def cross_validate(dataset: ExpediaDataset, model_config: dict):
    temp_idx, test_idx = train_test_split(
        np.arange(datasize),
        test_size=model_args["test_size"],
        random_state=model_args["SEED"],
    )

    kf = KFold(
        n_splits=model_config["n_splits"],
        shuffle=True,
        random_state=model_config["SEED"],
    )
    fold = 0
    for train_idx, val_idx in kf.split(temp_idx):
        dataset_train = dataset[train_idx]
        dataset_val = dataset[val_idx]
        dataset_test = dataset[test_idx]

        model = AttentionNTKChoiceModel(d=d, d0=d0, d2=d2, model_config=model_args).to(
            device
        )
        model.fit(dataset_train, dataset_val, dataset_test, device)
        model.evaluate()
        return


device = torch.device("cuda")
config_path = "configs/exp_config.yaml"
data_path = "data/expedia"
model_args = load_config(config_path=config_path)["expedia_ntk_model_args"]

dataset = ExpediaDataset(data_path=data_path)
datasize = len(dataset)
print(f"Has total {datasize} samples!")
d = dataset.feature_vec_length
d0 = 50
d2 = 50

print(d)
cross_validate(dataset=dataset, model_config=model_args)

# model = AttentionNTKChoiceModel(d=d, d0=d0, d2=d2, model_config=model_args).to(device)


# X = dataset.X[[0]]
# card = dataset.cardinality[[0]]
# # print(dataset.d)
# mask = compute_mask_from_card(card, dataset.d)
# # print(mask.shape)
# print(X.shape)
# print(model(X.to(dtype=torch.float32, device=device), mask.to(device=device)))
