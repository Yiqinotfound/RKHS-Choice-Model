import torch
import torch.nn as nn
import torch.optim as optim
import math

def xavier_initialize_weights(model:nn.Module):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)

def warmup_lr_scheduler(optimizer, warmup_epochs, num_epochs=100):
    def lambda_function(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1 - (epoch - warmup_epochs) / (num_epochs - warmup_epochs + 1)
    return optim.lr_scheduler.LambdaLR(optimizer, lambda_function)
    
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )