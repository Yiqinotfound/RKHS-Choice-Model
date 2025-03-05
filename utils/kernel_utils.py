from scipy.special import gamma, kv
import torch
import numpy as np
from tqdm import tqdm
def compute_polynomial_kernel_tensor_FF(
    S1_set: torch.Tensor, S2_set: torch.Tensor, kernel_params: dict, K: torch.Tensor
):
    c = kernel_params["c"]
    degree = kernel_params["degree"]
    Kg = torch.mm(S1_set, S2_set.T)
    Kg += c

    Kg = torch.pow(Kg, degree)

    Kg /= torch.max(Kg)
    return (Kg.unsqueeze(-1).unsqueeze(-1)) * (K.unsqueeze(0).unsqueeze(0))


def compute_gaussian_kernel_tensor_FF(
    S1_set: torch.Tensor,
    S2_set: torch.Tensor,
    kernel_params: dict,
    K: torch.Tensor,
):
    length_scale = kernel_params["length_scale"]
    sq_dist = torch.cdist(S1_set, S2_set, p=2) ** 2
    dist = torch.sqrt(sq_dist)
    length_scale = torch.median(dist)

    sigma = kernel_params["sigma"]
    Kg: torch.Tensor = sigma**2 * torch.exp(-sq_dist / (2 * length_scale**2))

    return (Kg.unsqueeze(-1).unsqueeze(-1)) * (K.unsqueeze(0).unsqueeze(0))


def compute_matern_kernel_tensor_FF(
    S1_set: torch.Tensor,
    S2_set: torch.Tensor,
    kernel_params: dict,
    K: torch.Tensor,
):
    nu = kernel_params["nu"]
    sigma = kernel_params["sigma"]
    length_scale = kernel_params["length_scale"]
    dist = torch.cdist(S1_set, S2_set, p=2)
    dist = torch.where(dist == 0, torch.tensor(1e-6), dist)
    scaled_dist = (
        torch.sqrt(torch.tensor(2 * nu), device=S1_set.device) * dist / length_scale
    )
    factor = (2 ** (1 - nu)) / gamma(nu)

    Kg: torch.Tensor = (
        sigma**2
        * factor
        * (scaled_dist**nu)
        * torch.as_tensor(
            kv(nu, scaled_dist), device=S1_set.device, dtype=torch.float32
        )
    )

    return (Kg.unsqueeze(-1).unsqueeze(-1)) * (K.unsqueeze(0).unsqueeze(0))


def mask_kernel_tensor_FF(
    kernel_tensor: torch.Tensor, S1_set: torch.Tensor, S2_set: torch.Tensor
):
    mask1 = (S1_set != 0).unsqueeze(1).unsqueeze(3)
    mask2 = (S2_set != 0).unsqueeze(0).unsqueeze(2)
    mask = mask1 & mask2
    return kernel_tensor * mask



def compute_matern_kernel_tensor_FB(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel_params: dict,
    batch_size: int = 1024,
) -> torch.Tensor:
    """
    Compute the row-wise Matern kernel between two 3D tensors with batch processing.

    Args:
        X: torch.Tensor of shape (m, d, p)
        Y: torch.Tensor of shape (n, d, p)
        sigma: float, the bandwidth parameter for the Matern kernel
        nu: float, the smoothness parameter of the Matern kernel
        length_scale: float, the length scale of the Matern kernel
        batch_size: int, size of the batch for processing

    Returns:
        torch.Tensor: Matern kernel matrix of shape (m, n, d, d)
    """
    m = X.shape[0]
    n = Y.shape[0]

    d = X.shape[1]
    sigma = kernel_params["sigma"]
    nu = kernel_params["nu"]
    length_scale = kernel_params["length_scale"]

    # Initialize an empty tensor to store the final result
    result = torch.zeros((m, n, d, d), dtype=torch.float32, device=X.device)

    from tqdm import tqdm

    # Loop over batches of X
    for i in tqdm(range(0, m, batch_size), desc="Processing X batches"):
        X_batch = X[i : i + batch_size]  # Shape: (batch_size, d, p)

        # Compute the difference tensor for this batch, in a vectorized manner
        for j in tqdm(range(0, n, batch_size), desc="Processing Y batches", leave=False):
            Y_batch = Y[j : j + batch_size]  # Shape: (batch_size, d, p)

            # Calculate the squared L2 norm (difference of rows)
            diff = X_batch.unsqueeze(1).unsqueeze(3) - Y_batch.unsqueeze(0).unsqueeze(2)
            squared_l2_norm = torch.sum(
                diff**2, dim=-1
            )  # Shape: (batch_size, batch_size, d, d)

            # Compute the Matern kernel
            dist = torch.sqrt(squared_l2_norm)  # Shape: (batch_size, batch_size, d, d)
            matern = matern_kernel_FBhelper(dist, nu, length_scale)

            # Scale by the bandwidth parameter sigma
            result[i : i + batch_size, j : j + batch_size] = sigma**2 * matern

    return result


def matern_kernel_FBhelper(
    dist: torch.Tensor, nu: float, length_scale: float
) -> torch.Tensor:
    """
    Computes the Matern kernel for a given distance matrix.

    Args:
        dist: torch.Tensor, distance matrix of shape (batch_size, batch_size, d, d)
        nu: float, the smoothness parameter of the Matern kernel
        length_scale: float, the length scale of the Matern kernel

    Returns:
        torch.Tensor: Matern kernel matrix of shape (batch_size, batch_size, d, d)
    """
    # Compute the scaling factor
    scaled_dist = torch.sqrt(torch.tensor(2 * nu)) * dist / length_scale
    scaled_dist = torch.where(
        dist == 0, torch.tensor(1e-7, device=dist.device), scaled_dist
    )
    factor = (2 ** (1 - nu)) / gamma(nu)

    # Compute the Matern kernel using the formula involving the Bessel function (K_nu)

    matern = (
        factor
        * (scaled_dist**nu)
        * torch.as_tensor(kv(nu, scaled_dist.cpu().numpy()), device=dist.device)
    )

    return matern


def compute_gaussian_kernel_RF(
    S1: torch.Tensor, S2: torch.Tensor, kernel_params: dict, K: torch.Tensor, Nw=1000
):
    d = S1.shape[0]
    sigma = kernel_params["sigma"]
    lengthscale = kernel_params["lengthscale"]
    W: torch.Tensor = torch.randn(Nw, d, d) / lengthscale

    b = torch.rand(Nw, d) * 2 * torch.pi

    mask1 = S1.unsqueeze(1)
    mask2 = S2.unsqueeze(0)
    mask = mask1 * mask2

    phi_1 = torch.tensor(np.sqrt(2 / Nw)) * torch.cos(W @ S1 + b).reshape(d, Nw)
    phi_2 = torch.tensor(np.sqrt(2 / Nw)) * torch.cos(W @ S2 + b).reshape(d, Nw)

    return sigma**2 * (phi_1 @ phi_2.T) * K * mask

def compute_gaussian_kernel_tensor_FB(
    X: torch.Tensor, Y: torch.Tensor, kernel_params: dict, batch_size: int = 1024
) -> torch.Tensor:
    """
    Compute the row-wise Gaussian (RBF) kernel between two 3D tensors.

    Args:
        X: torch.Tensor of shape (m, d, p)
        Y: torch.Tensor of shape (n, d, p)
        sigma: float, the bandwidth parameter for the Gaussian kernel

    Returns:
        torch.Tensor: Gaussian kernel matrix of shape (m, n, d, d)
    """
    # Compute squared L2 norm between rows of X and Y
    length_scale = kernel_params["length_scale"]
    sigma = kernel_params["sigma"]
    m = X.shape[0]
    n = Y.shape[0]
    d = X.shape[2]

    result = torch.zeros((m, n, d, d), dtype=torch.float32, device=X.device)
    # print(result.shape)

    for i in tqdm(range(0, m, batch_size), desc="Processing X batches"):
        X_batch = X[i : i + batch_size]  # (batchsize, d, p)

        for j in tqdm(range(0, n, batch_size), desc="Processing Y batches", leave=False):
            Y_batch = Y[j : j + batch_size]  # (batchsize, d, p)

            diff = X_batch.unsqueeze(1).unsqueeze(3) - Y_batch.unsqueeze(0).unsqueeze(2)
            squared_l2_norm = torch.sum(diff**2, dim=-1)
            kernel = torch.exp(-squared_l2_norm / (2 * length_scale**2))
            result[i : i + batch_size, j : j + batch_size] = kernel

    return sigma**2 * result


def mask_kernel_tensor_FB(
    kernel_tensor: torch.Tensor,
    cardinality_1: torch.Tensor,
    cardinality_2: torch.Tensor,
) -> torch.Tensor:

    # max cardinality
    d = kernel_tensor.shape[2]

    # mask1
    mask1 = torch.arange(d, device=kernel_tensor.device).unsqueeze(
        0
    ) < cardinality_1.unsqueeze(
        1
    )  # (m, d)
    mask1 = mask1.unsqueeze(1).unsqueeze(-1)  # (m, 1, d, 1)

    # mask2
    mask2 = torch.arange(d, device=kernel_tensor.device).unsqueeze(
        0
    ) < cardinality_2.unsqueeze(
        1
    )  # (n, d)
    mask2 = mask2.unsqueeze(0).unsqueeze(-2)  # (1, n, 1, d)

    # the final mask
    mask = mask1 & mask2  # (m, n, d, d)

    # the masked kernel tensor
    masked_kernel_tesnor = kernel_tensor * mask  # (m, n, d, d)
    
    del kernel_tensor, mask

    return masked_kernel_tesnor