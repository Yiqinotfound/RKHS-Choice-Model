import sys, os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data_preprocess.hotel_data_loader import HotelDataset
from models.NTKModel import NTKChoiceModel
from utils.model_utils import load_config, save_to_csv


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = "configs/exp_config.yaml"
    save_path = "real_data_experiments/hotel_NTK_results/hotel_NTK_results.csv"
    model_args = load_config(config_path=config_path)["hotel_ntk_args"]

    dataset = HotelDataset(instance_id=model_args["instance_id"])
    input_dim = dataset.d
    output_dim = dataset.d
    hidden_dims = model_args["H"]
    model = NTKChoiceModel(
        input_dim=input_dim, output_dim=output_dim, model_args=model_args
    ).to(device)

    model.fit(
        S_train=dataset.S_train.to(device),
        y_train=dataset.y_train.to(device),
        S_test=dataset.S_test.to(device),
        y_test=dataset.y_test.to(device),
    )
    model.evaluate()

    data_rows = [
        list(model_args.values())
        + [model.nll_train, model.nll_test, model.rmse_train, model.rmse_test]
    ]

    header_columns = list(model_args.keys()) + [
        "nll_train",
        "nll_test",
        "rmse_train",
        "rmse_test",
    ]
    save_to_csv(data_rows=data_rows, header_columns=header_columns, file_path=save_path)

    # print(model(dataset.S_train.to(device)))
    # print(model(dataset.S_test.to(device)))
