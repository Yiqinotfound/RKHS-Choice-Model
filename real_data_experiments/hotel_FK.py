import torch
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from data_preprocess.hotel_data_loader import HotelDataset
from models.FK_feature_free_model import FKFFModel
from utils.model_utils import load_config, save_to_csv


if __name__ == "__main__":
    exp_config_path = "configs/exp_config.yaml"
    kernel_config_path = "configs/hotel_kernel_config.yaml"
    save_path = "real_data_experiments/hotel_FK_results/hotel_FK_results.csv"
    model_args = load_config(config_path=exp_config_path)["hotel_fk_args"]
    kernel_args = load_config(config_path=kernel_config_path)

    train_device = torch.device("cuda")

    dataset = HotelDataset(instance_id=model_args["instance_id"])

    model = FKFFModel(
        dataset=dataset,
        model_args=model_args,
        kernel_args=kernel_args,
        train_device=train_device,
    )
    model.precompute()
    model.train_param_init()
    model.fit()
    model.evaluate(model.best_alphaset)
    data_rows = [
        list(model_args.values())
        + [model.kernel_params_str]
        + [
            model.nll_train,
            model.nll_test,
            model.hard_rmse_train,
            model.hard_rmse_test,
            model.train_time,
            model.precompute_time,
        ]
    ]
    header_columns = (
        list(model_args.keys())
        + ["kernel_params"]
        + [
            "nll_train",
            "nll_test",
            "hard_rmse_train",
            "hard_rmse_test",
            "train_time",
            "precompute_time",
        ]
    )
    save_to_csv(
        data_rows=data_rows,
        header_columns=header_columns,
        file_path=save_path,
    )

    # # devices
    # precompute_device = "cpu"
    # train_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # print("precompute device:", precompute_device)
    # print("train device:", train_device)

    # # instance id

    # # load the dataset
    # dataset = HotelDataset(instance_id=instance_id)

    # # initialze the model
    # model = FKFFModel(
    #     dataset=dataset,
    #     kernel_type="gaussian",
    #     kernel_params=kernel_type_params_dict["gaussian"],
    #     mask=True,
    #     precompute_device=precompute_device,
    #     train_device=train_device,
    # )

    # # move the data for following training process
    # model.move_data_semi_precision()

    # # init training parameters
    # model.train_param_init(
    #     lambda_=1e-5,
    #     grad_norm_threshold=1e-4,
    #     alpha_std=0.001,
    #     lr=0.001,
    #     batch_size=None,
    # )

    # model.run()
    # model.evaluate()
