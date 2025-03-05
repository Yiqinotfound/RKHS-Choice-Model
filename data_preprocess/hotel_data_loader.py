import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def convert_json_to_df(json_file_path: str):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    in_sample_transactions = data["transactions"]["in_sample_transactions"]
    out_sample_transactions = data["transactions"]["out_of_sample_transactions"]
    product_labels = data["product_labels"]

    in_sample_transactions = pd.DataFrame(in_sample_transactions)
    out_sample_transactions = pd.DataFrame(out_sample_transactions)

    # rename 'prodcut' to 'choice'
    in_sample_transactions.rename(columns={"product": "choice"}, inplace=True)
    out_sample_transactions.rename(columns={"product": "choice"}, inplace=True)
    product_labels = pd.DataFrame(
        list(product_labels.items()), columns=["product_id", "product_name"]
    )
    return in_sample_transactions, out_sample_transactions, product_labels


def convert_list_to_one_hot(transaction: list, d):
    one_hot = np.zeros(d)
    for item in transaction:
        one_hot[item] = 1
    return one_hot


def convert_to_one_hot(transactions: pd.DataFrame, d):
    transactions["offered_product_one_hot"] = transactions["offered_products"].apply(
        lambda x: convert_list_to_one_hot(x, d)
    )
    transactions["choice_one_hot"] = transactions["choice"].apply(
        lambda x: convert_list_to_one_hot([x], d)
    )
    return transactions


class HotelDataset:
    def __init__(self, instance_id: int):
        self.instance_id = instance_id
        self.data_path = (
            f"data/hotel_json/instance_{instance_id}.json"
        )
        (
            self.in_sample_transactions,
            self.out_sample_transactions,
            self.product_labels,
        ) = convert_json_to_df(self.data_path)
        self.d = len(self.product_labels) + 1
        self.train_datasize = len(self.in_sample_transactions)
        self.test_datasize = len(self.out_sample_transactions)
        self.in_sample_transactions = convert_to_one_hot(
            self.in_sample_transactions, self.d
        )
        self.out_sample_transactions = convert_to_one_hot(
            self.out_sample_transactions, self.d
        )
        self.S_train = torch.stack(
            [
                torch.tensor(sample, dtype=torch.float32)
                for sample in self.in_sample_transactions["offered_product_one_hot"]
            ]
        )
        self.y_train = torch.stack(
            [
                torch.tensor(sample, dtype=torch.float32)
                for sample in self.in_sample_transactions["choice_one_hot"]
            ]
        )
        self.S_test = torch.stack(
            [
                torch.tensor(sample, dtype=torch.float32)
                for sample in self.out_sample_transactions["offered_product_one_hot"]
            ]
        )
        self.y_test = torch.stack(
            [
                torch.tensor(sample, dtype=torch.float32)
                for sample in self.out_sample_transactions["choice_one_hot"]
            ]
        )

        self.total_train_item = sum(
            len(products)
            for products in self.in_sample_transactions["offered_products"]
        )
        self.total_test_item = sum(
            len(products)
            for products in self.out_sample_transactions["offered_products"]
        )

    def save_transaction_data(self):
        self.in_sample_transactions.to_csv('')