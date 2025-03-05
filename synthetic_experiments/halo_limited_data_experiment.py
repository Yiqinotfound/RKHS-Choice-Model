import numpy as np
from cmnl_halo import CMNL
from rkhs_halo import RKHSHalo
from HaloDataLoader import HaloDataset
import torch
from scipy.special import gamma, kv
import gc
from RegressionModel import RegressionChoiceModel


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
item_number = 20
set_number = 2000
grad_norm_threshold = 0
# grad_norm_threshold = 0
alpha_std = 0.01
lr = 0.01
lambda_ = 1e-9
run_num = 10
kernel_params_dict = {
    # "gaussian": {"length_scale": 1, "sigma": 5},
    "0.5matern": {"sigma": 5, "nu": 0.5, "length_scale": 1},
    # "1.5matern": {"sigma": 1, "nu": 1.5, "length_scale": 1},
    # "2.5matern": {"sigma": 1, "nu": 2.5, "length_scale": 1},
    # "poly": {"sigma": 5, "degree": 3, "c": 2},
    # "linear": {"degree": 1, "c": 0},
}
rkhs_filepath = (
    "/data/yiqi/RKHSChoiceModel/Halo/results/rkhs_first_order_results_2_8.csv"
)
cmnl_filepath = (
    "/data/yiqi/RKHSChoiceModel/Halo/results/cmnl_first_order_results_2_8.csv"
)
mask_list = [False, True]

dataset = HaloDataset(
    item_number=item_number,
    set_number=set_number,
    utility_mean=0,
    utility_std=1,
    first_order_interaction_mean=0,
    first_order_interaction_std=0.5,
    second_order_interaction_mean=0,
    second_order_interaction_std=0.25,
    third_order_interaction_mean=0,
    third_order_interaction_std=0.125,
    test_size=0.25,
    random_seed=0,
    device=device,
)

# regression_model = RegressionChoiceModel(dataset, device)
# regression_model.fit(lr=0.01)
# regression_model.evaluate()

for kernel_type, kernel_params in kernel_params_dict.items():
    rkhs_model = RKHSHalo(
        dataset=dataset,
        kernel_type=kernel_type,
        kernel_params=kernel_params,
        mask=True,
        device=device,
    )
    rkhs_model.run(
        alpha_std=alpha_std,
        lr=lr,
        lambda_=lambda_,
        grad_norm_threshold=grad_norm_threshold,
    )
    rkhs_model.evaluate_limited()


# for _ in range(run_num):
#     for kernel_type, kernel_params in kernel_params_dict.items():
#         for mask in mask_list:
#             print(f"Kernel Type: {kernel_type}")
#             rkhs_model = RKHSHalo(
#                 dataset=dataset,
#                 kernel_type=kernel_type,
#                 kernel_params=kernel_params,
#                 mask=mask,
#                 device=device,
#             )
#             rkhs_model.run(
#                 alpha_std=alpha_std,
#                 lr=lr,
#                 lambda_=lambda_,
#                 grad_norm_threshold=grad_norm_threshold,
#             )
#             rkhs_model.evaluate()
#             rkhs_model.save_results(filepath=rkhs_filepath)


#             del rkhs_model
#             gc.collect()

#     cmnl_model = CMNL(dataset, device)
#     cmnl_model.run(grad_norm_threshold=1e-4)
#     cmnl_model.evaluate()
#     cmnl_model.save_results(cmnl_filepath)
#     del cmnl_model
#     gc.collect()
