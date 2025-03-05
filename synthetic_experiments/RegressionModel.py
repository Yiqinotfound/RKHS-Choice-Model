import torch
from HaloDataLoader import HaloDataset
import time

eps = torch.tensor(1e-7)
safe_log = lambda x: torch.log(torch.clamp(x, eps, 1.0))

class RegressionChoiceModel:
    def __init__(self, dataset: HaloDataset, device):
        self.dataset = dataset
        self.device = device
    
    def zero_diagonal_gradients(self, first_order_interaction: torch.Tensor):
        first_order_interaction_grad = first_order_interaction.grad
        if first_order_interaction_grad is not None:
            first_order_interaction_grad[
                torch.eye(self.dataset.item_number, device=self.device).bool()
            ] = 0
        return first_order_interaction_grad

    def fit(self, lr: float):
        self.lr = lr
        self.base_utilities = torch.zeros(self.dataset.item_number, device=self.device)
        self.first_order_interaction = torch.zeros(
            self.dataset.item_number, self.dataset.item_number, device=self.device
        )

        self.second_order_interaction = torch.zeros(
            self.dataset.item_number,
            self.dataset.item_number,
            self.dataset.item_number,
            device=self.device,
        )
        self.third_order_interaction = torch.zeros(
            self.dataset.item_number,
            self.dataset.item_number,
            self.dataset.item_number,
            self.dataset.item_number,
            device=self.device,
        )

        self.base_utilities.requires_grad = True
        self.first_order_interaction.requires_grad = True
        self.second_order_interaction.requires_grad = True
        self.third_order_interaction.requires_grad = True

        self.best_base_utility = None
        self.best_first_order_interaction = None
        self.best_second_order_interaction = None
        self.best_third_order_interaction = None
        self.best_loss = float("inf")
        self.optimizer = torch.optim.Adam(
            [
                self.base_utilities,
                self.first_order_interaction,
                self.second_order_interaction,
                self.third_order_interaction,
            ],
            lr=self.lr,
        )
        self.run_start = time.time()

        for epoch in range(50):
            self.optimizer.zero_grad()
            loss = self.objective()
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_base_utility = self.base_utilities.detach().clone()
                self.best_first_order_interaction = self.first_order_interaction.detach().clone()
                self.best_second_order_interaction = self.second_order_interaction.detach().clone()
                self.best_third_order_interaction = self.third_order_interaction.detach().clone()
            loss.backward()
            self.zero_diagonal_gradients(self.first_order_interaction)
            print(f"Epoch {epoch} Loss: {loss.item()}")
            self.optimizer.step()
        
    def objective(self):
        U = self.compute_U(
            self.dataset.S_train,
            self.base_utilities,
            self.first_order_interaction,
            self.second_order_interaction,
            self.third_order_interaction,
        )
        return torch.nn.CrossEntropyLoss()(U, self.dataset.y_train)

    def compute_U(
        self,
        S:torch.Tensor,
        base_utilities: torch.Tensor,
        first_order_interaction: torch.Tensor,
        second_order_interaction: torch.Tensor,
        third_order_interaction: torch.Tensor,
    ):
        U = base_utilities + torch.matmul(S, first_order_interaction)
        U = U * S

        j, k = torch.triu_indices(
            self.dataset.item_number,
            self.dataset.item_number,
            offset=1,
            device=self.device,
        )

        # 创建有效(j,k)对的掩码（基于S的非零条件）
        valid_mask = (S[:, j] != 0) & (
            S[:, k] != 0
        )  # [set_num, num_pairs]

        # 重塑维度用于广播
        l = torch.arange(self.dataset.item_number, device=self.device).view(
            1, 1, -1
        )  # [1,1,item_num]
        j_exp = j.view(1, -1, 1)  # [1,num_pairs,1]
        k_exp = k.view(1, -1, 1)  # [1,num_pairs,1]

        # 创建三维有效性掩码（转换为float）
        cond_j = (l != j_exp).float()  # [1,num_pairs,item_num]
        cond_k = (l != k_exp).float()  # [1,num_pairs,item_num]
        cond_S = (S[:, None, :] != 0).float()  # [set_num,1,item_num]

        l_valid_mask = cond_j * cond_k * cond_S  # [set_num,num_pairs,item_num]

        # 获取交互项（确保数值类型）
        interactions = second_order_interaction[j, k, :]  # [num_pairs,item_num]
        interactions = interactions[None, ...].float()  # [1,num_pairs,item_num]

        # 应用有效掩码
        valid_mask_exp = valid_mask[:, :, None].float()  # [set_num,num_pairs,1]
        combined_mask = l_valid_mask * valid_mask_exp  # [set_num,num_pairs,item_num]

        # 爱因斯坦求和
        U += torch.einsum("spl,spl->sl", interactions, combined_mask)

        i, j, k = torch.combinations(
            torch.arange(self.dataset.item_number, device=self.device), r=3
        ).t()
        num_triples = i.size(0)  # 当item_number=20时，num_triples=1140

        # 创建三元组有效性掩码 (S[i], S[j], S[k] 均非零)
        triple_valid_mask = (
            (S[:, i] != 0)
            & (S[:, j] != 0)
            & (S[:, k] != 0)
        )  # [set_num, num_triples]

        # 重塑维度对齐二阶代码结构
        l = torch.arange(self.dataset.item_number, device=self.device).view(
            1, 1, -1
        )  # [1, 1, item_num]
        i_exp = i.view(1, -1, 1)  # [1, num_triples, 1]
        j_exp = j.view(1, -1, 1)  # [1, num_triples, 1]
        k_exp = k.view(1, -1, 1)  # [1, num_triples, 1]

        # 创建三维有效性掩码（与二阶保持相同维度结构）
        cond_i = (l != i_exp).float()  # [1, num_triples, item_num]
        cond_j = (l != j_exp).float()  # [1, num_triples, item_num]
        cond_k = (l != k_exp).float()  # [1, num_triples, item_num]
        cond_S = (
            S[:, None, :] != 0
        ).float()  # [set_num, 1, item_num]

        # 合并条件（三维张量）
        l_valid_mask = (
            cond_i * cond_j * cond_k * cond_S
        )  # [set_num, num_triples, item_num]

        # 获取三阶交互项（假设third_order_interaction形状为[item_num, item_num, item_num, item_num]）
        interactions = third_order_interaction[i, j, k, :]  # [num_triples, item_num]
        interactions = interactions[None, ...].float()  # [1, num_triples, item_num]

        # 扩展有效掩码（保持三维结构）
        triple_mask_exp = triple_valid_mask[
            :, :, None
        ].float()  # [set_num, num_triples, 1]

        # 统一维度后进行逐元素相乘
        combined_mask = (
            l_valid_mask * triple_mask_exp
        )  # [set_num, num_triples, item_num]

        # 爱因斯坦求和（与二阶保持相同模式）
        U_third_order = torch.einsum("spl,spl->sl", interactions, combined_mask)
        U = U + U_third_order
        return U

    def compute_P(self, U):
        U_max = torch.max(U, dim=1, keepdim=True).values
        U_stable = U - U_max
        exp_U = torch.exp(U_stable)
        sum_exp_utility = torch.sum(exp_U, dim=1, keepdim=True)
        P = exp_U / sum_exp_utility
        return P

    def cross_entropy(self, U, y):
        P = self.compute_P(U)
        log_P = safe_log(P)
        loss_matrix = -y * log_P
        loss_value = torch.sum(loss_matrix) / loss_matrix.size(0)
        return loss_value
    
    def evaluate(self):
        U_train = self.compute_U(
            self.dataset.S_train,
            self.best_base_utility,
            self.best_first_order_interaction,
            self.best_second_order_interaction,
            self.best_third_order_interaction,
        )
        P_train = self.compute_P(U_train)
        self.total_prduct_train  = torch.sum(self.dataset.S_train)
        self.soft_rmse = torch.sqrt(
            torch.sum((P_train - self.dataset.P_train) ** 2) / self.total_prduct_train
        ) 
        print(f"Soft RMSE: {self.soft_rmse}")