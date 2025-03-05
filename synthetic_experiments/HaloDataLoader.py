import torch
from sklearn.model_selection import train_test_split
import itertools

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class HaloDataset:
    def __init__(
        self,
        item_number: int,
        set_number: int,
        utility_mean: float = 0,
        utility_std: float = 0,
        first_order_interaction_mean: float = 0,
        first_order_interaction_std: float = 0,
        second_order_interaction_mean: float = 0,
        second_order_interaction_std: float = 0,
        third_order_interaction_mean: float = 0,
        third_order_interaction_std: float = 0,
        test_size: float = 0.2,
        random_seed: int = 0,
        device: torch.device = "cpu",
        new_generator=False,
    ):
        self.device = device
        self.item_number = item_number
        self.set_number = set_number
        self.utility_mean = utility_mean
        self.utility_std = utility_std
        self.first_order_interaction_mean = first_order_interaction_mean
        self.first_order_interaction_std = first_order_interaction_std
        self.second_order_interaction_mean = second_order_interaction_mean
        self.second_order_interaction_std = second_order_interaction_std
        self.third_order_interaction_mean = third_order_interaction_mean
        self.third_order_interaction_std = third_order_interaction_std
        self.new_generator = new_generator

        self.test_size = test_size
        self.random_seed = random_seed

        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)

        self.base_utilities = (
            torch.randn(item_number, device=device) * utility_std + utility_mean
        )
        # self.base_utilities = self.base_utilities - self.base_utilities[0]
        self.first_order_interaction = (
            torch.randn(item_number, item_number, device=device)
            * first_order_interaction_std
            + first_order_interaction_mean
        )
        self.first_order_interaction.fill_diagonal_(0)

        self.second_order_interaction = (
            torch.randn(item_number, item_number, item_number, device=device)
            * second_order_interaction_std
            + second_order_interaction_mean
        )
        self.second_order_interaction = (
            self.second_order_interaction
            + self.second_order_interaction.permute(1, 0, 2)
        ) / 2
        self.mask_second_order_interaction()
        # self.second_order_interaction.fill_diagonal_(0)
        print("second order interaction finished")

        self.generate_third_order_interaction()
        print("third order interaction finished")

        self.S = self.generate_S(self.set_number)
        self.U = self.compute_U()
        self.P = self.compute_set_total_prob(self.U, self.S)
        self.sample_choices()
        self.F = self.compute_freq()

        self.split_data(test_size=self.test_size)
        y_train = self.y_train
        y_train_mean = y_train.mean(dim=0)
        y_train_centered = y_train - y_train_mean
        self.cov = y_train_centered.t() @ y_train_centered / (y_train.shape[0] - 1)

        # self.base_S = torch.eye(self.item_number, device=self.device)
        # self.compute_one_order_S()

    def mask_second_order_interaction(self):
        # when i == j or i == k or j == k, set the second order interaction to 0
        # for i in range(self.item_number):
        #     for j in range(self.item_number):
        #         for k in range(self.item_number):
        #             if i == j or i == k or j == k:
        #                 self.second_order_interaction[i, j, k] = 0
        indices = torch.arange(self.item_number, device=self.device)
        i = indices.view(-1, 1, 1)
        j = indices.view(1, -1, 1)
        k = indices.view(1, 1, -1)

        mask = (i == j) | (i == k) | (j == k)
        self.second_order_interaction[mask] = 0

    def generate_third_order_interaction(self):
        self.third_order_interaction = (
            torch.randn(
                self.item_number,
                self.item_number,
                self.item_number,
                self.item_number,
                device=self.device,
            )
            * self.third_order_interaction_std
            + self.third_order_interaction_mean
        )
        self.third_order_interaction = (
            self.third_order_interaction
            + self.third_order_interaction.permute(0, 2, 1, 3)
            + self.third_order_interaction.permute(1, 0, 2, 3)
            + self.third_order_interaction.permute(2, 0, 1, 3)
            + self.third_order_interaction.permute(2, 1, 0, 3)
            + self.third_order_interaction.permute(1, 2, 0, 3)
        ) / 6

        # print(
        #     self.third_order_interaction[0, 1, 2, 3],
        #     self.third_order_interaction[1, 0, 2, 3],
        # )
        indices = torch.arange(self.item_number, device=self.device)
        i = indices.view(-1, 1, 1, 1)
        j = indices.view(1, -1, 1, 1)
        k = indices.view(1, 1, -1, 1)
        l = indices.view(1, 1, 1, -1)

        mask = (i == j) | (i == k) | (i == l) | (j == k) | (j == l) | (k == l)
        self.third_order_interaction[mask] = 0

    def compute_one_order_S(self):
        indices = range(5)

        # 获取所有两两组合
        combinations = list(itertools.combinations(indices, 2))

        # 生成对应的one-hot向量并存储在列表中
        one_order_S = []
        for comb in combinations:
            vector = [1 if i in comb else 0 for i in range(5)]
            one_order_S.append(vector)

        # 转换为torch.tensor
        self.one_order_S = torch.tensor(
            one_order_S, device=self.device, dtype=torch.float32
        )
        self.one_order_utilities = self.base_utilities + torch.matmul(
            self.one_order_S, self.first_order_interaction
        )
        self.one_order_utilities = self.one_order_utilities * self.one_order_S
        self.one_order_prob = self.compute_set_total_prob(
            self.one_order_utilities, self.one_order_S
        )

    def compute_U(self):
        U = self.base_utilities + torch.matmul(self.S, self.first_order_interaction)
        U = U * self.S
        if self.second_order_interaction_std == 0:
            print("no second order interaction")
            return U

        # 生成所有j < k的组合索引对
        j, k = torch.triu_indices(
            self.item_number, self.item_number, offset=1, device=self.device
        )

        # 创建有效(j,k)对的掩码（基于S的非零条件）
        valid_mask = (self.S[:, j] != 0) & (self.S[:, k] != 0)  # [set_num, num_pairs]

        # 重塑维度用于广播
        l = torch.arange(self.item_number, device=self.S.device).view(
            1, 1, -1
        )  # [1,1,item_num]
        j_exp = j.view(1, -1, 1)  # [1,num_pairs,1]
        k_exp = k.view(1, -1, 1)  # [1,num_pairs,1]

        # 创建三维有效性掩码（转换为float）
        cond_j = (l != j_exp).float()  # [1,num_pairs,item_num]
        cond_k = (l != k_exp).float()  # [1,num_pairs,item_num]
        cond_S = (self.S[:, None, :] != 0).float()  # [set_num,1,item_num]

        l_valid_mask = cond_j * cond_k * cond_S  # [set_num,num_pairs,item_num]

        # 获取交互项（确保数值类型）
        interactions = self.second_order_interaction[j, k, :]  # [num_pairs,item_num]
        interactions = interactions[None, ...].float()  # [1,num_pairs,item_num]

        # 应用有效掩码
        valid_mask_exp = valid_mask[:, :, None].float()  # [set_num,num_pairs,1]
        combined_mask = l_valid_mask * valid_mask_exp  # [set_num,num_pairs,item_num]

        # 爱因斯坦求和
        U = U + torch.einsum("spl,spl->sl", interactions, combined_mask)

        if self.third_order_interaction_std == 0:
            return U

        # 生成所有 i < j < k 的三元组索引 (组合数 C(n,3))
        i, j, k = torch.combinations(
            torch.arange(self.item_number, device=self.device), r=3
        ).t()
        num_triples = i.size(0)  # 当item_number=20时，num_triples=1140

        # 创建三元组有效性掩码 (S[i], S[j], S[k] 均非零)
        triple_valid_mask = (
            (self.S[:, i] != 0) & (self.S[:, j] != 0) & (self.S[:, k] != 0)
        )  # [set_num, num_triples]

        # 重塑维度对齐二阶代码结构
        l = torch.arange(self.item_number, device=self.S.device).view(
            1, 1, -1
        )  # [1, 1, item_num]
        i_exp = i.view(1, -1, 1)  # [1, num_triples, 1]
        j_exp = j.view(1, -1, 1)  # [1, num_triples, 1]
        k_exp = k.view(1, -1, 1)  # [1, num_triples, 1]

        # 创建三维有效性掩码（与二阶保持相同维度结构）
        cond_i = (l != i_exp).float()  # [1, num_triples, item_num]
        cond_j = (l != j_exp).float()  # [1, num_triples, item_num]
        cond_k = (l != k_exp).float()  # [1, num_triples, item_num]
        cond_S = (self.S[:, None, :] != 0).float()  # [set_num, 1, item_num]

        # 合并条件（三维张量）
        l_valid_mask = (
            cond_i * cond_j * cond_k * cond_S
        )  # [set_num, num_triples, item_num]

        # 获取三阶交互项（假设third_order_interaction形状为[item_num, item_num, item_num, item_num]）
        interactions = self.third_order_interaction[
            i, j, k, :
        ]  # [num_triples, item_num]
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

        # U = U * self.S
        print(U[0])
        return U

    def compute_set_prob(self, utility: torch.Tensor, S: torch.Tensor):
        utility_max = torch.max(utility)
        utility_stable = utility - utility_max
        utility_exp = torch.exp(utility_stable)
        utility_exp_masked = utility_exp * S
        sum_exp_utility = torch.sum(utility_exp_masked)
        prob = utility_exp_masked / sum_exp_utility
        return prob

    def compute_set_total_prob(self, U: torch.Tensor, S_batch: torch.Tensor):
        U_max = torch.max(U, dim=1, keepdim=True).values
        U_stable = U - U_max
        exp_U = torch.exp(U_stable)
        exp_U_masked = exp_U * S_batch
        sum_exp_U = torch.sum(exp_U_masked, dim=1, keepdim=True)
        P = exp_U_masked / sum_exp_U
        return P

    def generate_S(self, datasize):
        k = torch.randint(1, self.item_number + 1, (datasize,), device=self.device)

        S = torch.zeros(datasize, self.item_number, device=self.device)

        indices = torch.rand(datasize, self.item_number, device=self.device).argsort(
            dim=1
        )

        mask = torch.arange(self.item_number, device=self.device).reshape(
            1, self.item_number
        ) < k.reshape(datasize, 1)

        cols = indices[mask]
        rows = torch.arange(datasize, device=self.device).repeat_interleave(k)

        S[rows, cols] = 1

        return S

    def sample_choices(self):
        self.choices = torch.multinomial(self.P, 1)
        # convert to one-hot
        self.y = torch.zeros_like(self.P)
        self.y.scatter_(1, self.choices, 1)

    def split_data(self, test_size=0.2):

        if test_size == 0:
            self.S_train = self.S
            self.U_train = self.U
            self.P_train = self.P
            self.y_train = self.y
            self.F_train = self.F
            self.F_train = self.S_test = self.U_test = self.P_test = self.y_test = None
            self.train_datasize = self.set_number
            self.test_datasize = 0
            return
        (
            self.S_train,
            self.S_test,
            self.U_train,
            self.U_test,
            self.P_train,
            self.P_test,
            self.y_train,
            self.y_test,
            self.F_train,
            self.F_test,
        ) = train_test_split(
            self.S,
            self.U,
            self.P,
            self.y,
            self.F,
            test_size=test_size,
            random_state=self.random_seed,
        )
        self.train_datasize = len(self.S_train)
        self.test_datasize = len(self.S_test)

    def compute_freq(self):

        weights = 2 ** torch.arange(self.item_number - 1, -1, -1, device=device)
        ids = (self.S * weights).sum(dim=1).long()

        # Get inverse indices for group mapping
        unique_ids, inverse_indices = torch.unique(ids, return_inverse=True)

        # Calculate group sums and counts
        M = unique_ids.size(0)
        sum_y = torch.zeros((M, self.item_number), device=device)
        sum_y.index_add_(0, inverse_indices, self.y)
        counts = torch.bincount(inverse_indices, minlength=M).float().unsqueeze(1)

        # Compute averages and expand back to original shape
        avg_y = sum_y / counts
        return avg_y[inverse_indices]


# item_number = 5
# set_number = 8000
# dataset = HaloDataset(
#     item_number=item_number,
#     set_number=set_number,
#     utility_mean=0,
#     utility_std=1,
#     first_order_interaction_mean=0,
#     first_order_interaction_std=0.2,
#     second_order_interaction_mean=0,
#     second_order_interaction_std=0,
#     third_order_interaction_mean=0,
#     third_order_interaction_std=0,
#     test_size=0,
#     random_seed=0,
#     device=device,
#     new_generator=True,
# )
# print(dataset.base_utilities)
# print(dataset.S)
# print(dataset.compute_freq())
# print(dataset.P_train)
# print(dataset.F_train)
# torch.save(dataset.base_utilities, "base_utilities.pt")
# torch.save(dataset.interaction, "interaction.pt")
# print(dataset.base_utilities)
# print(dataset.one_order_S)
# print(dataset.one_order_S.shape)
# print(dataset.second_order_interaction)
# print(dataset.first_order_interaction)
# # compute y_train's covariance matrix


# y_train = dataset.y_train
# y_train_mean = y_train.mean(dim=0)
# y_train_centered = y_train - y_train_mean
# cov = y_train_centered.t() @ y_train_centered / (y_train.shape[0] - 1)
# print(cov)

# print(dataset.base_utilities)
# print(dataset.S[1])
# print(dataset.P[1])
# for i in range(dataset.set_number):
#     if torch.equal(dataset.S[i], dataset.S[1]):
#         print(dataset.y[i])

# print(dataset.base_utilities)
# print(dataset.interaction)
# print(dataset.S)
# print(dataset.U)
# print(dataset.P)
