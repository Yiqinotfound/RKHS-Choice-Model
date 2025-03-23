import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from utils.creterias_and_loss_utils import safe_log


class ResAssortNet(nn.Module):

    def __init__(
        self,
        model_args: dict,
        products: np.array,
        device: torch.device,
        **kwargs
    ):
        """
        product_num : int , including the no_purchase option
        products: shape = (product_num-1, product_feature_length) if no_purchase is not included, else shape = (product_num, product_feature_length)
        """

        super(ResAssortNet, self).__init__()

        self.device = device

        self.model_args = model_args

        self.product_num = model_args["product_num"]

        self.product_feature_length = model_args["product_feature_length"]
        self.customer_feature_length = model_args["customer_feature_length"]

        if model_args["add_no_purchase"]:
            # add the no purchase option to the product set
            no_purchase = np.zeros((1, self.product_feature_length))
            self.products = torch.Tensor(
                np.concatenate((no_purchase, products), axis=0)
            )
        else:
            self.products = products

        self.products = self.products.to(self.device)

        if self.product_num != len(self.products):

            raise Exception("product amount not match!")

        # customer encoder
        self.customer_encoder_midlayers = model_args["customer_encoder_midlayers"]
        self.customer_encoder_midlayer_num = len(self.customer_encoder_midlayers)

        # product encoder
        self.product_encoder_midlayers = model_args["product_encoder_midlayers"]
        self.product_encoder_midlayer_num = len(self.product_encoder_midlayers)

        # cross effect
        self.cross_effect_layers = model_args["cross_effect_layers"]
        self.cross_effect_layer_num = len(self.cross_effect_layers)

        if self.customer_encoder_midlayers[-1] != self.product_encoder_midlayers[-1]:

            raise Exception("cross utility layer not match!")

        ## cusEncoder network
        self.cusEncoder = self.generate_sequential_layers(
            feature_dim=self.customer_feature_length,
            mid_layers=self.customer_encoder_midlayers,
        )

        ## prodEncoder network
        self.prodEncoder = self.generate_sequential_layers(
            feature_dim=self.product_feature_length,
            mid_layers=self.product_encoder_midlayers,
        )

        ## res layer
        self.Res = nn.Linear(2, 1, device=self.device)

        ## activation function
        self.sigmoid = nn.Sigmoid()

        ## CrossEffect network
        self.cross_effect = self.generate_sequential_layers(
            self.product_num, np.append(self.cross_effect_layers, self.product_num)
        )

        ## normalize
        self.softmax = nn.Softmax(dim=-1)

    def generate_sequential_layers(self, feature_dim: int, mid_layers: list):

        modulist = []
        from_dim = feature_dim
        for layer in range(len(mid_layers)):

            to_dim = mid_layers[layer]

            sub_net = nn.Sequential(
                nn.Linear(from_dim, to_dim),
                nn.BatchNorm1d(to_dim),
                nn.LeakyReLU(0.2, inplace=True),
            )
            modulist.append(sub_net)

            from_dim = to_dim

        return nn.ModuleList(modulist).to(self.device)

    # def misc_to_gpu(self):
    #     self.products = self.products.cuda()

    def forward(self, input, **kwargs):

        assorts = input[:, : self.product_num]
        cusFs = input[:, self.product_num :]

        ## encode product features
        prod_features = self.products
        for m in self.prodEncoder:
            prod_features = m(prod_features)

        ## encode customer features
        cus_features = cusFs
        for m in self.cusEncoder:
            cus_features = m(cus_features)

        ## utility
        cus_features = torch.unsqueeze(cus_features, 1)

        encoded_utils = torch.matmul(cus_features, prod_features.T)

        residual = torch.unsqueeze(assorts, 1)

        encoded_utils = torch.cat((encoded_utils, residual), 1).permute(0, 2, 1)

        encoded_utils = self.sigmoid(torch.squeeze(self.Res(encoded_utils), 2))

        ## cross effect
        prob = encoded_utils
        for m in self.cross_effect:
            prob = m(prob)

        prob = self.softmax(prob)

        ## regularize

        prob = prob * assorts
        prob = prob / prob.sum(dim=-1).unsqueeze(-1)

        return prob

    def compute_prob_batch(
        self,
        data_full: torch.Tensor,
        batch_size: int = 1024,
        device: torch.device = "cpu",
    ):
        with torch.no_grad():
            all_probs = []
            for i in range(0, len(data_full), batch_size):
                data_batch = data_full[i : i + batch_size].to(device)
                probs_batch = self.forward(input=data_batch.to(torch.float32))
                all_probs.append(probs_batch.cpu())

        return torch.cat(all_probs, dim=0).to(device)

    def compute_acc(self, P: torch.Tensor, Y: torch.Tensor):
        P_max = torch.argmax(P, dim=1)
        Y_max = torch.argmax(Y, dim=1)
        return torch.sum(P_max == Y_max).item() / Y.size(0)

    def cross_entropy(self, P: torch.Tensor, Y: torch.Tensor):
        return -torch.sum(Y * safe_log(P)) / Y.size(0)

    def fit(
        self,
        dataset_train: tuple[torch.Tensor, torch.Tensor],
        dataset_val: tuple[torch.Tensor, torch.Tensor],
        dataset_test: tuple[torch.Tensor, torch.Tensor],
    ):

        self.train()

        # record time
        t1 = time.time()

        # train, val test data and max_items
        self.data_train, self.y_train = dataset_train
        self.data_val, self.y_val = dataset_val
        self.data_test, self.y_test = dataset_test
        self.max_items = self.data_train.shape[1]

        self.train_dataset = TensorDataset(self.data_train, self.y_train)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.model_args["batch_size"]
        )

        # smoothed labels
        self.y_train_smooth = (
            1 - self.model_args["smoothing"]
        ) * self.y_train + self.model_args["smoothing"] / self.max_items
        self.y_val_smooth = (
            1 - self.model_args["smoothing"]
        ) * self.y_val + self.model_args["smoothing"] / self.max_items
        self.y_test_smooth = (
            1 - self.model_args["smoothing"]
        ) * self.y_test + self.model_args["smoothing"] / self.max_items

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.model_args["learning_rate"],
            weight_decay=self.model_args["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.model_args["scheduler_gamma"]
        )

        self.best_val_loss = float("inf")
        self.best_parameters = None

        with tqdm(
            total=self.model_args["max_epochs"], desc="Training", unit="epoch"
        ) as pbar:
            for epoch in range(self.model_args["max_epochs"]):
                for data_batch, y_batch in self.train_loader:
                    data_batch = data_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    optimizer.zero_grad()
                    prob_train_batch = self.forward(input=data_batch)
                    nll_train_batch = self.cross_entropy(P=prob_train_batch, Y=y_batch)
                    nll_train_batch.backward()
                    optimizer.step()

                self.evaluate(self.device)
                if self.nll_val < self.best_val_loss:
                    self.best_val_loss = self.nll_val
                    self.best_parameters = {
                        k: v.clone() for k, v in self.state_dict().items()
                    }

                scheduler.step()
                pbar.set_postfix(
                    {
                        "Train Loss": self.nll_train,
                        "Val Loss": self.nll_val,
                        "Test Loss": self.nll_test,
                    }
                )
                pbar.update(1)
                self.train()

        self.train_time = time.time() - t1
        self.load_state_dict(self.best_parameters)

    def evaluate(self, device):
        with torch.no_grad():

            P_train = self.compute_prob_batch(data_full=self.data_train, device=device)
            self.nll_train = self.cross_entropy(
                P=P_train, Y=self.y_train_smooth.to(device)
            ).item()
            self.acc_train = self.compute_acc(P=P_train, Y=self.y_train.to(device))

            P_val = self.compute_prob_batch(data_full=self.data_val, device=device)
            self.nll_val = self.cross_entropy(
                P=P_val, Y=self.y_val_smooth.to(device)
            ).item()
            self.acc_val = self.compute_acc(P=P_val, Y=self.y_val.to(device))

            P_test = self.compute_prob_batch(data_full=self.data_test, device=device)
            self.nll_test = self.cross_entropy(
                P=P_test, Y=self.y_test_smooth.to(device)
            ).item()
            self.acc_test = self.compute_acc(P=P_test, Y=self.y_test.to(device))


if __name__ == "__main__":

    product_num = 5
    SEED = 1234

    product_feature_length = 2
    customer_feature_length = 3
    products = np.array([[1, 0], [0, 1], [0.5, 0.5], [0.3, 0.2]])

    model = ResAssortNet(
        product_num=product_num,
        product_feature_length=product_feature_length,
        customer_feature_length=customer_feature_length,
        products=products,
    )

    print(model.products)
    data_batch = torch.Tensor(
        [
            [1, 0, 0, 1, 0, 0.1, 0.4, 0.3],
            [1, 1, 1, 1, 1, 0.5, 0.7, 0.2],
            [1, 0, 1, 1, 1, 0.6, 0.1, 0.1],
            [1, 0, 1, 0, 1, 0.9, 0.1, 0.1],
            [1, 0, 1, 0, 0, 0.6, 0.8, 0.1],
            [1, 0, 0, 0, 1, 0.6, 0.1, 0.3],
        ]
    )
    # 就是把 S和 customer feature concat到一起
    ## print(model(data_batch))
    print("---------------debug-------------------")
    print(model(data_batch))
    # print(model.products)
