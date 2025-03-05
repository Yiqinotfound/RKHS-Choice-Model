import torch
import gc

safe_log = lambda x: torch.log(torch.clamp(x, 1e-6, 1.0))


# compute utilities
def compute_U(alphaset: torch.Tensor, kernel_tensor: torch.Tensor):
    U = torch.einsum("jikl, il -> jk", kernel_tensor, alphaset)
    return U


# compute utilities random feature
def compute_U_RF(theta: torch.Tensor, Phi: torch.Tensor, sqrt_Q_diag: torch.Tensor):
    Q_Theta = sqrt_Q_diag @ theta
    U = torch.einsum("nwd,w->nd", Phi, Q_Theta)
    return U


# compute probs for feature-free model
def compute_P_FF(U: torch.Tensor, S: torch.Tensor, mask: bool):
    U_max = torch.max(U)
    U_stable = U - U_max
    exp_U = torch.exp(U_stable)
    if mask:
        exp_U = exp_U * S
    sum_exp_U = torch.sum(exp_U, dim=1, keepdim=True)
    P = exp_U / sum_exp_U
    return P


# compute cross entropy loss for feature-free model
def cross_entropy_FF(U: torch.Tensor, S: torch.Tensor, y: torch.Tensor, mask: bool):
    P = compute_P_FF(U=U, S=S, mask=mask)
    log_P = safe_log(P)
    loss_matrix = -y * log_P
    loss = torch.sum(loss_matrix) / len(y)
    return loss


# general cross entropy loss
def cross_entropy(U: torch.Tensor, y: torch.Tensor, mask_tensor: torch.Tensor):
    P = compute_P(U=U, mask_tensor=mask_tensor)
    log_P = safe_log(P)
    loss_matrix = -y * log_P
    loss = torch.sum(loss_matrix) / len(y)
    return loss


# compute probs for feature-based model
def compute_P_FB(U: torch.Tensor, cardinality: torch.Tensor, mask: bool):
    d = U.shape[1]
    U_max = torch.max(U)
    U_stable = U - U_max
    exp_U = torch.exp(U_stable)

    if mask:
        cardinality_mask = torch.arange(d, device=U.device).unsqueeze(
            0
        ) < cardinality.unsqueeze(1)
        exp_U = exp_U * cardinality_mask

    sum_exp_U = torch.sum(exp_U, dim=1, keepdim=True)
    P = exp_U / sum_exp_U
    return P


# general compute P function
def compute_P(U: torch.Tensor, mask_tensor: torch.Tensor):
    U_max = torch.max(U)
    U_stable = U - U_max
    exp_U = torch.exp(U_stable)
    exp_U = exp_U * mask_tensor
    sum_exp_U = torch.sum(exp_U, dim=1, keepdim=True)
    P = exp_U / sum_exp_U
    return P


# compute mask from card
def compute_mask_from_card(cardinality: torch.Tensor, d: int):
    return torch.arange(d, device=cardinality.device).unsqueeze(
        0
    ) < cardinality.unsqueeze(1)


# compute cross entropy loss for feature-based model
def cross_entropy_FB(
    U: torch.Tensor, cardinality: torch.Tensor, y: torch.Tensor, mask: bool
):
    P = compute_P_FB(U=U, cardinality=cardinality, mask=mask)
    if torch.any(torch.isnan(P)):
        print("P has NaN")
    log_P = safe_log(P)
    loss_matrix = -y * log_P
    loss = torch.sum(loss_matrix) / len(y)
    return loss


# compute rkhs norm regularizatoin
def regularization(alphaset: torch.Tensor, kernel_tensor: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "id,ijdl,jl->",
        alphaset,
        kernel_tensor,
        alphaset,
    )


# use batch to compute reg
def regularization_with_batch(
    alphaset: torch.Tensor, kernel_tensor: torch.Tensor, batch_size: int = None
) -> torch.Tensor:
    if batch_size is None:
        return regularization(alphaset, kernel_tensor)
    datasize = alphaset.shape[0]  # 数据规模
    num_batches = (datasize + batch_size - 1) // batch_size  # 计算总批次数

    total_reg = torch.tensor(
        0.0,
        device=alphaset.device,
        dtype=alphaset.dtype,
        requires_grad=True,
    )

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, datasize)

        with torch.no_grad():
            kernel_tensor_batch = kernel_tensor[start_idx:end_idx]

        alphaset_batch = alphaset[start_idx:end_idx]  # Shape: [batch_size, d]

        reg_batch = torch.einsum(
            "id,ijdl,jl->", alphaset_batch, kernel_tensor_batch, alphaset
        )
        total_reg = total_reg + reg_batch
        del kernel_tensor_batch, alphaset_batch, reg_batch
        gc.collect()

    return total_reg


def l2_regularization_RF(theta: torch.Tensor) -> torch.Tensor:
    reg = torch.sum(theta.pow(2))
    return reg


# rmse creteira
def rmse(P: torch.Tensor, y: torch.Tensor, total_item: int) -> float:
    return torch.sqrt(torch.sum((P - y) ** 2) / total_item).item()


# negative log likelihood
def nll(P: torch.Tensor, y: torch.Tensor) -> float:
    datasize = y.shape[0]
    return -torch.sum(y * safe_log(P)).item() / datasize


# accuracy
def accuracy(P: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute the accuracy for a choice model.

    Args:
        y (torch.Tensor): Actual choices, one-hot encoded, shape (datasize, d).
        p (torch.Tensor): Predicted probabilities, shape (datasize, d).

    Returns:
        float: Accuracy of the model.
    """
    with torch.no_grad():
        # Convert one-hot encoded `y` to the indices of the actual choices
        y_indices = torch.argmax(Y, dim=1)

        # Get the predicted choice indices from the probabilities
        p_indices = torch.argmax(P, dim=1)

        # Calculate the number of correct predictions
        correct_predictions = torch.sum(y_indices == p_indices).item()

        # Calculate accuracy as the ratio of correct predictions to total samples
        accuracy = correct_predictions / Y.size(0)

    return accuracy
