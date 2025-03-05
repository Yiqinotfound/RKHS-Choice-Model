import sys
import torch

sys.path.append("/data/yiqi/RKHSChoiceModel/")
from data_preprocess.hotel_data_loader import HotelDataset
from models.RF_feauture_free_model import GaussainRFFF

kernel_type_params_dict = {
    "gaussian": {"length_scale": 1.0, "sigma": 1.0},
    "sigmoid": {"alpha": 0.5, "c": 1.0, "sigma": 1.0},
    "poly": {"c": 1.0, "degree": 2},
    "linear": {"c": 1.0, "degree": 1},
}

if __name__ == "__main__":

    # devices
    precompute_device = "cpu"
    train_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("precompute device:", precompute_device)
    print("train device:", train_device)

    # instance id
    instance_id = 3

    # load the dataset
    dataset = HotelDataset(instance_id=instance_id)

    # initialze the model
    model = GaussainRFFF()

    # move the data for following training process
    model.move_data_semi_precision()

    # init training parameters
    model.train_param_init(
        lambda_=1e-5,
        grad_norm_threshold=1e-4,
        alpha_std=0.001,
        lr=0.001,
        batch_size=None,
    )

    model.run()
    model.evaluate()
