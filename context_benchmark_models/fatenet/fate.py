"""An implementation of the scoring module for FATE estimators."""

import functools

import torch
import torch.nn as nn

from context_benchmark_models.fatenet.instance_reduction import DeepSet
from context_benchmark_models.fatenet.object_mapping import DenseNeuralNetwork
import time
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from utils.creterias_and_loss_utils import safe_log, cross_entropy
from torch.optim.lr_scheduler import StepLR


class FATEScoring(nn.Module):
    r"""Map instances to scores with the FATE approach.

    Let's show the FATE approach on an example. To simplify things, we'll use a
    simply identity-embedding. The FATE module will then aggregate the context
    by simply taking the average of the objects (feature-wise). To further
    simplify things the actual pairwise utility is just computed by the sum of
    all features of the object and the context.

    >>> import torch.nn as nn
    >>> from csrank.modules.object_mapping import DeterministicSumming
    >>> scoring = FATEScoring(
    ...     n_features=2,
    ...     pairwise_utility_module=DeterministicSumming,
    ...     embedding_module=nn.Identity,
    ... )

    Now let's define some problem instances.

    >>> object_a = [0.5, 0.8]
    >>> object_b = [1.5, 1.8]
    >>> object_c = [2.5, 2.8]
    >>> object_d = [3.5, 3.6]
    >>> object_e = [4.5, 4.6]
    >>> object_f = [5.5, 5.6]
    >>> # instance = list of objects to rank
    >>> instance_a = [object_a, object_b, object_c]
    >>> instance_b = [object_d, object_e, object_f]
    >>> import torch
    >>> instances = torch.tensor([instance_a, instance_b])

    Let's focus on the first instance in this example. The aggregated identity
    embedding is

    >>> embedding_1 = (object_a[0] + object_b[0] + object_c[0]) / 3
    >>> embedding_2 = (object_a[1] + object_b[1] + object_c[1]) / 3
    >>> (embedding_1, embedding_2)
    (1.5, 1.8)

    for the first and second feature respectively. So the utility of object_a
    within the context (defined by the mock sum utility) should be

    >>> embedding_1 + embedding_2 + object_a[0] + object_a[1]
    4.6

    Let's verify this:

    >>> scoring(instances)
    tensor([[ 4.6000,  6.6000,  8.6000],
            [16.2000, 18.2000, 20.2000]])

    As you can see, the scoring comes to the same result for the first object
    of the first instance.

    Parameters
    ----------
    n_features: int
        The number of features each object has.
    embedding_size: int
        The size of the embeddings that should be generated. Defaults to
        ``n_features`` if not specified.
    pairwise_utility_module: pytorch module with one integer parameter
        The module that should be used for pairwise utility estimations. Uses a
        simple linear mapping not specified. You likely want to replace this
        with something more expressive such as a ``DenseNeuralNetwork``. This
        should take the size of the input values as its only parameter. You can
        use ``functools.partial`` if necessary. This corresponds to
        :math:`U` in Figure 2 of [1]_.
    embedding_module: pytorch module with one integer parameter
        The module that should be used for the object embeddings. Its
        constructor should take two parameters: The size of the input and the
        size of the output. This corresponds to :math:`\Phi` in Figure 2 of
        [1]_. The default is a ``DenseNeuralNetwork`` with 5 hidden layers and
        64 units per hidden layer.

    References
    ----------
    .. [1] Pfannschmidt, K., Gupta, P., & Hüllermeier, E. (2019). Learning
    choice functions: Concepts and architectures. arXiv preprint
    arXiv:1901.10860.
    """

    def __init__(
        self,
        n_features,
        embedding_size=None,
        pairwise_utility_module=None,
        embedding_module=None,
        model_args: dict = None,
    ):
        super().__init__()
        self.model_args = model_args
        if embedding_size is None:
            embedding_size = n_features
        if pairwise_utility_module is None:
            pairwise_utility_module = functools.partial(
                DenseNeuralNetwork,
                hidden_layers=5,
                units_per_hidden=64,
                output_size=1,
            )
        if embedding_module is None:
            embedding_module = functools.partial(
                DenseNeuralNetwork, hidden_layers=5, units_per_hidden=64
            )

        self.embedding = DeepSet(
            n_features,
            embedding_size,
            embedding_module=embedding_module,
        )

        self.pairwise_utility_module = pairwise_utility_module(
            n_features + embedding_size
        )

    def forward(self, instances, **kwargs):
        n_objects = instances.size(1)
        contexts = self.embedding(instances)
        # Repeat each context for each object within the instance; This is then
        # a flat list of contexts. Then reshape to have a list of contexts per
        # instance.
        context_per_object = contexts.repeat_interleave(n_objects, dim=0).reshape_as(
            instances
        )
        pairs = torch.stack((instances, context_per_object), dim=-1)
        utilities = self.pairwise_utility_module(pairs.flatten(start_dim=-2)).squeeze()
        prob = torch.nn.functional.softmax(utilities, dim=-1)
        return prob

    def compute_prob_batch(
        self,
        X_full: torch.Tensor,
        batch_size: int = 1024,
        device: torch.device = None,
    ):
        with torch.no_grad():
            all_probs = []
            for i in range(0, len(X_full), batch_size):
                X_batch = X_full[i : i + batch_size].to(device)
                probs_batch = self.forward(instances=X_batch.to(torch.float32))
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
        dataset_train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dataset_val: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        dataset_test: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        device:torch.device="cpu"
    ):

        self.train()

        # record time
        t1 = time.time()

        self.X_train, self.y_train, self.cardinality_train = dataset_train
        self.X_test, self.y_test, self.cardinality_test = dataset_test
        self.X_val, self.y_val, self.cardinality_val = dataset_val
        self.max_items = self.X_train.shape[1]

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

        # train dataset and train loader
        self.train_dataset = TensorDataset(
            self.X_train, self.y_train, self.cardinality_train
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.model_args["batch_size"], shuffle=True
        )

        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.model_args["learning_rate"],
            weight_decay=self.model_args["weight_decay"],
        )
        scheduler = StepLR(
            optimizer=optimizer,
            step_size=self.model_args["step_size"],
            gamma=self.model_args["scheduler_gamma"],
        )


        self.best_val_loss = float("inf")

        with tqdm(
            total=self.model_args["max_epochs"], desc="Training", unit="epoch"
        ) as pbar:
            for epoch in range(self.model_args["max_epochs"]):
                total_train_loss = 0.0
                for X_batch, y_batch, cardinality_batch in self.train_loader:

                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    optimizer.zero_grad()
                    prob_train_batch = self.forward(instances=X_batch)
                    nll_train_batch = self.cross_entropy(P=prob_train_batch, Y=y_batch)
                    nll_train_batch.backward()
                    optimizer.step()
                    total_train_loss += nll_train_batch.item()

                # evaluate
                self.evaluate(device)
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

            P_train = self.compute_prob_batch(X_full=self.X_train, device=device)
            self.nll_train = self.cross_entropy(
                P=P_train, Y=self.y_train_smooth.to(device)
            ).item()
            self.acc_train = self.compute_acc(P=P_train, Y=self.y_train.to(device))

            P_val = self.compute_prob_batch(X_full=self.X_val, device=device)
            self.nll_val = self.cross_entropy(
                P=P_val, Y=self.y_val_smooth.to(device)
            ).item()
            self.acc_val = self.compute_acc(P=P_val, Y=self.y_val.to(device))

            P_test = self.compute_prob_batch(X_full=self.X_test, device=device)
            self.nll_test = self.cross_entropy(
                P=P_test, Y=self.y_test_smooth.to(device)
            ).item()
            self.acc_test = self.compute_acc(P=P_test, Y=self.y_test.to(device))
