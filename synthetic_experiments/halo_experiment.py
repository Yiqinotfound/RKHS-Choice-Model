import numpy as np
from cmnl_halo import CMNL
from rkhs_halo import RKHSHalo
from HaloDataLoader import HaloDataset
import torch
from scipy.special import gamma, kv
import gc


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
item_number = 5
set_number = 8000
grad_norm_threshold = 1e-3
# grad_norm_threshold = 0
alpha_std = 0.01
lr = 0.001
lambda_ = 1e-9
run_num = 10
kernel_params_dict = {
    # "gaussian": {"length_scale": 1, "sigma": 1},
    # "0.5matern": {"sigma": 1, "nu": 0.5, "length_scale": 1},
    # "1.5matern": {"sigma": 1, "nu": 1.5, "length_scale": 1},
    # "2.5matern": {"sigma": 1, "nu": 2.5, "length_scale": 1},
    "poly": {"degree": 2, "c": 1,"sigma":1},
    # "linear": {"degree": 1, "c": 0, "sigma":1},
}
rkhs_filepath = "/data/yiqi/RKHSChoiceModel/Halo/results/rkhs_first_order_results_2_9.csv"
cmnl_filepath = "/data/yiqi/RKHSChoiceModel/Halo/results/cmnl_first_order_results_2_9.csv"
mask_list = [True]

dataset = HaloDataset(
    item_number=item_number,
    set_number=set_number,
    utility_mean=0,
    utility_std=1,
    first_order_interaction_mean=0,
    first_order_interaction_std=0.2,
    second_order_interaction_mean=0,
    second_order_interaction_std=0.1,
    third_order_interaction_mean=0,
    third_order_interaction_std=0,
    test_size=0.2,
    random_seed=0,
    device=device,
)
print(dataset.base_utilities)
print(dataset.first_order_interaction)
print(dataset.second_order_interaction)
torch.save(dataset.second_order_interaction, "/data/yiqi/RKHSChoiceModel/Halo/results/second_order_interaction.pt")

# print(dataset.base_S)
# print(dataset.interaction)
# # save the interaction matrix
# torch.save(dataset.interaction, "/data/yiqi/RKHSChoiceModel/Halo/results/interaction.pt")
# torch.save(dataset.base_utilities, "/data/yiqi/RKHSChoiceModel/Halo/results/base_utilities.pt")
# rkhs_model = RKHSHalo(
#     dataset=dataset,
#     kernel_type="linear",
#     kernel_params=kernel_params_dict["linear"],
#     mask=True,
#     device=device,
# )
# rkhs_model.run(
#     alpha_std=alpha_std,
#     lr=lr,
#     lambda_=lambda_,
#     grad_norm_threshold=grad_norm_threshold,
# )
# rkhs_model.evaluate()
# rkhs_model.retrive_base_utility()
# interaction = rkhs_model.retrive_one_order_interaction()
# print(interaction)
# torch.save(
#     rkhs_model.base_utility,
#     "/data/yiqi/RKHSChoiceModel/Halo/results/rkhs_base_utility.pt",
# )
# torch.save(interaction, "/data/yiqi/RKHSChoiceModel/Halo/results/rkhs_interaction.pt")
# rkhs_model.save_results(filepath=rkhs_filepath)



# for _ in range(run_num):
#     for kernel_type, kernel_params in kernel_params_dict.items():
#         for mask in mask_list:
#             pass
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

    # cmnl_model = CMNL(dataset, device)
    # cmnl_model.run(grad_norm_threshold=1e-4)
    # cmnl_model.evaluate()
    # cmnl_model.save_results(cmnl_filepath)
    # del cmnl_model
    # gc.collect()
