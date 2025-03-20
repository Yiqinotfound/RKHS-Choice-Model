import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.model_utils import create_one_hot_from_choices


class ExpediaDataset(Dataset):
    def __init__(
        self,
        raw_data_path: str = None,
        preprocessed_path: str = None,
        data_path: str = None,
        mode: str = "direct_load",
    ):
        if mode == "direct_load":
            print("Directly Loading X.pt, y.pt and cardinality.pt...")
            self.X = torch.load(f"{data_path}/X.pt", weights_only=True).to(
                torch.float16
            )
            self.y = torch.load(f"{data_path}/y.pt", weights_only=True).to(
                torch.float16
            )
            self.cardinality = torch.load(
                f"{data_path}/cardinality.pt", weights_only=True
            ).to(torch.float16)
            self.N, self.d, self.feature_vec_length = self.X.shape
        if mode == "clean" and raw_data_path is not None:
            self.preprocess_dataframe(raw_data_path)
            print("Cleaning is Done!")
            return
        elif mode == "preprocess" and preprocessed_path is not None:
            print("Loading preprocessed dataframe...")
            self.expedia_df = pd.read_csv(
                preprocessed_path, engine="pyarrow", dtype=float
            )
            print("Creating choices and y...")
            self.choices_df = self.expedia_df.groupby("srch_id").apply(
                lambda x: x.booking_bool.argmax()
            )
            # convert the choices to dummy
            self.d = 39
            self.choices = torch.tensor(self.choices_df.values, dtype=torch.int64)
            self.y = create_one_hot_from_choices(self.choices, self.d)

            print("Creating cardinalities")
            self.cardinality_df = self.expedia_df.groupby(["srch_id"])["av"].sum()
            self.cardinality = torch.tensor(
                self.cardinality_df.values, dtype=torch.float32
            )

            print("Dropping columns and converting it to tensor...")
            drop_cols = [
                "is_no_purchase",
                "av",
                "booking_bool",
                "srch_id",
                "prop_id",
                "site_id_0",
                "visitor_location_country_id_0",
                "srch_destination_id_0",
                "prop_country_id_0",
            ]
            self.expedia_df.drop(columns=drop_cols, inplace=True)
            self.X = torch.tensor(self.expedia_df.values, dtype=torch.float32)
            self.N = int(self.X.shape[0] / self.d)
            self.X = self.X.reshape(self.N, self.d, -1)

            print("Saving data to data/expedia/X.pt, y.pt and cardinality.pt")
            
            torch.save(self.X, "../data/expedia/X.pt")
            torch.save(self.y, "../data/expedia/y.pt")
            torch.save(self.cardinality, "../data/expedia/cardinality.pt")

            print("Data saved to data/expedia/X.pt, y.pt and cardinality.pt")

    def preprocess_dataframe(self, data_path: str):
        print("Loading Expedia data")
        self.expedia_df = pd.read_csv(data_path, engine="pyarrow")

        self.expedia_df.date_time = pd.to_datetime(
            self.expedia_df.date_time, format="%Y-%m-%d %H:%M:%S"
        )
        self.expedia_df.loc[:, "day_of_week"] = self.expedia_df.loc[
            :, "date_time"
        ].dt.dayofweek
        self.expedia_df.loc[:, "month"] = self.expedia_df.loc[:, "date_time"].dt.month
        self.expedia_df.loc[:, "hour"] = self.expedia_df.loc[:, "date_time"].dt.hour

        print("Filtering ids with less than 1000 occurrences")
        for id_col in [
            "site_id",
            "visitor_location_country_id",
            "prop_country_id",
            "srch_destination_id",
        ]:
            value_counts = (
                self.expedia_df[["srch_id", id_col]]
                .drop_duplicates()[id_col]
                .value_counts()
            )
            kept_ids = value_counts.index[value_counts.gt(1000)]
            for id_ in self.expedia_df[id_col].unique():
                if id_ not in kept_ids:
                    self.expedia_df.loc[self.expedia_df[id_col] == id_, id_col] = -1

        print("Filtering for price, stay length, booking window, etc.")
        self.expedia_df = self.expedia_df[self.expedia_df.price_usd <= 1000]
        self.expedia_df = self.expedia_df[self.expedia_df.price_usd >= 10]
        self.expedia_df["log_price"] = self.expedia_df.price_usd.apply(np.log)
        self.expedia_df = self.expedia_df[self.expedia_df.srch_length_of_stay <= 14]
        self.expedia_df = self.expedia_df[self.expedia_df.srch_booking_window <= 365]
        self.expedia_df["booking_window"] = np.log(
            self.expedia_df["srch_booking_window"] + 1
        )
        self.expedia_df = self.expedia_df.fillna(-1)

        # sort the df columns
        print("Sorting DF columns")
        order_cols = [
            "srch_id",
            "prop_id",
            "prop_starrating",
            "prop_review_score",
            "prop_brand_bool",
            "prop_location_score1",
            "prop_location_score2",
            "prop_log_historical_price",
            "position",
            "promotion_flag",
            "srch_length_of_stay",
            "srch_adults_count",
            "srch_children_count",
            "srch_room_count",
            "srch_saturday_night_bool",
            "orig_destination_distance",
            "random_bool",
            "day_of_week",
            "month",
            "hour",
            "log_price",
            "booking_window",
            "site_id",
            "visitor_location_country_id",
            "prop_country_id",
            "srch_destination_id",
            "click_bool",
            "booking_bool",
        ]
        self.expedia_df = self.expedia_df[order_cols]

        # create availabilities
        print("Creating dummy availabilities")
        self.expedia_df["av"] = 1
        asst_size = 38  # Fixed number of items in the assortment

        # create dummy products to reach assortment size, fill with 0 except for srch_id
        print("Creating dummy products to reach assortment size")
        # return
        feature_names = self.expedia_df.columns.tolist()
        feature_names.remove("srch_id")
        print("feature_names: ", feature_names)
        # Loop to fill the data frame with dummy products
        for _ in tqdm(range(asst_size)):
            dum = (
                self.expedia_df.groupby("srch_id")
                .filter(lambda x: len(x) < asst_size)
                .groupby("srch_id")
                .first()
                .reset_index(drop=False)
            )
            dum.loc[:, feature_names] = 0
            dum.loc[:, "av"] = 0
            self.expedia_df = pd.concat([self.expedia_df, dum])

        # create the no purchase option
        print("Creating the no purchase option")
        df1 = (
            self.expedia_df.groupby("srch_id")
            .filter(lambda x: x.booking_bool.sum() == 1)  # booked
            .groupby("srch_id")
            .first()
            .reset_index(drop=False)
        )
        df1.loc[:, feature_names] = 0
        df1.loc[:, "is_no_purchase"] = 1
        # df1.loc[:, "log_price"] = 0
        # df1.loc[:, "booking_bool"] = 0

        df2 = (
            self.expedia_df.groupby("srch_id")
            .filter(lambda x: x.booking_bool.sum() == 0)  # not booked
            .groupby("srch_id")
            .first()
            .reset_index(drop=False)
        )
        df2.loc[:, feature_names] = 0
        df2.loc[:, "is_no_purchase"] = 1
        # df2.loc[:, "log_price"] = 0
        df2.loc[:, "booking_bool"] = 1
        self.expedia_df = pd.concat([self.expedia_df, df1, df2])

        # fill no purchase with 0
        self.expedia_df.loc[:, "is_no_purchase"] = self.expedia_df.loc[
            :, "is_no_purchase"
        ].fillna(0)
        print("Sorting the data frame")
        self.expedia_df = self.expedia_df.sort_values(
            ["srch_id", "av", "is_no_purchase"],
            ascending=[True, False, True],
            inplace=False,
        )
        # preprocessed_path = "preprocessed_expedia_rumnet.csv"
        # self.expedia_df.to_csv(preprocessed_path, index=False)

        # create one_hot for categorical fetures
        self.one_hot_features = [
            "site_id",
            "visitor_location_country_id",
            "srch_destination_id",
            "prop_country_id",
        ]
        self.expedia_df = pd.get_dummies(self.expedia_df, columns=self.one_hot_features)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.cardinality[idx]
