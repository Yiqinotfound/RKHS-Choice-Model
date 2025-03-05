import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import warnings
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")


class SwissMetroDataset(Dataset):
    def __init__(
        self,
        filepath: str = None,
        preprocess_mode: str = None,
    ):

        super().__init__()

        # maximum offer choice set cardinality
        self.d = 3

        if filepath is not None:
            # read data
            self.filepath = filepath
            self.data = pd.read_csv(self.filepath, sep="\t")

            # preprocess the data
            self.preprocess_mode = preprocess_mode
            if self.preprocess_mode == "rkhs":
                self.rkhs_preprocess()
            elif self.preprocess_mode == "rumnet":
                self.rumnet_preprocess()
        else:
            print("No data provided")
    def describe(self):
        print(f"Sample Size: {self.__len__()}")
        print(f"Max Offer Choice Set Size: {self.d}")
        print(f"Feature Vec Length: {self.feature_vec_length}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.cardinality[idx],

 
    def rumnet_preprocess(self):

        print("Preprocessing Raw Data Using RUMnet Setup...")
        encoder = LabelEncoder()
        scaler = MinMaxScaler()

        # drop the columns that are not used
        drop = ["ID", "SP", "SM_SEATS", "GROUP"]

        # self.data.loc[self.data["INCOME"] == 1, ["INCOME"]] = 0
        self.data.drop(drop, axis=1, inplace=True)

        # clean the unknown data
        self.data = self.data[
            (self.data["CHOICE"] != 0)
            # & (self.data["WHO"] != 0)
            # & (self.data["AGE"] != 6)
            # & (self.data["INCOME"] != 4)
        ]

        # customer data
        self.cust_df = self.data.copy()
        self.cust_df = self.cust_df.drop(
            columns=[
                "TRAIN_TT",
                "TRAIN_CO",
                "TRAIN_HE",
                "SM_TT",
                "SM_CO",
                "SM_HE",
                "CAR_TT",
                "CAR_CO",
                "CHOICE",
                "TRAIN_AV",
                "CAR_AV",
                "SM_AV",
            ]
        )

        # choice data
        choice_df = self.data[["CHOICE"]]
        choice_df.loc[:, "CHOICE"] = encoder.fit_transform(choice_df["CHOICE"])

        # transportation feature data
        self.trans_df = self.data[
            [
                "TRAIN_AV",
                "CAR_AV",
                "SM_AV",
                "TRAIN_TT",
                "TRAIN_CO",
                "TRAIN_HE",
                "SM_TT",
                "SM_CO",
                "SM_HE",
                "CAR_TT",
                "CAR_CO",
            ]
        ]

        # these columns are converted to one-hot encoding
        self.cust_df = pd.get_dummies(
            self.cust_df,
            columns=[
                "SURVEY",
                "AGE",
                "LUGGAGE",
                "INCOME",
                "PURPOSE",
                "FIRST",
                "TICKET",
                "WHO",
                "MALE",
                "GA",
                "ORIGIN",
                "DEST",
            ],
        )

        # # age, luggage, income are scaled
        # self.cust_df["AGE"] = encoder.fit_transform(self.cust_df["AGE"])
        # self.cust_df["AGE"] = scaler.fit_transform(self.cust_df[["AGE"]])
        # self.cust_df["LUGGAGE"] = encoder.fit_transform(self.cust_df["LUGGAGE"])
        # self.cust_df["INCOME"] = encoder.fit_transform(self.cust_df["INCOME"])
        self.cust_df = self.cust_df.astype("float")

        # Set car's HE to zero
        self.trans_df.loc[:, ["CAR_HE"]] = 0

        # scale the trainsportation features
        self.trans_df[["TRAIN_TT", "SM_TT", "CAR_TT"]] = (
            self.trans_df[["TRAIN_TT", "SM_TT", "CAR_TT"]]
            .astype(float)
            .apply(lambda x: x / 1000)
        )
        self.trans_df[["CAR_CO", "TRAIN_CO", "SM_CO"]] = (
            self.trans_df[["CAR_CO", "TRAIN_CO", "SM_CO"]]
            .astype(float)
            .apply(lambda x: x / 5000)
        )

        self.trans_df[["TRAIN_HE", "SM_HE", "CAR_HE"]] = (
            self.trans_df[["TRAIN_HE", "SM_HE", "CAR_HE"]]
            .astype(float)
            .apply(lambda x: x / 100)
        )

        # The following ugly code are used to transform the feature vecs to feature matrix , concat the customer and transportation feature vectors
        feature_matrix_list = []
        choice_list = []
        cardinality = []
        self.cust_feature_length = len(self.cust_df.columns)
        self.trans_feature_length = len(self.trans_df.columns)
        self.cust_feature_matrix = np.array(self.cust_df)
        self.trans_feature_matrix = np.array(self.trans_df)
        self.feature_tensor = torch.tensor(
            np.hstack((self.cust_feature_matrix, self.trans_feature_matrix)),
            dtype=torch.float32,
        )
        for i in tqdm(range(len(self.cust_df)), desc="Preprocessing", unit="sample"):
            cust_feature_vec = self.cust_df.iloc[i].values
            trans_feature_vec = self.trans_df.iloc[i].values
            train_feature_vec = trans_feature_vec[3:6]
            sm_feature_vec = trans_feature_vec[6:9]
            car_feature_vec = trans_feature_vec[9:]
            choice = choice_df.iloc[i].values[0]
            feature_vec_length = len(cust_feature_vec) + len(train_feature_vec) + self.d

            feature_matrix = np.zeros((3, feature_vec_length))

            cnt = 0

            train_idx = -1
            sm_idx = -1
            car_idx = -1

            if self.trans_df.iloc[i]["TRAIN_AV"] == 1:
                feature_vec = np.zeros(feature_vec_length)
                feature_vec[len(cust_feature_vec)] = 1
                feature_vec[: len(cust_feature_vec)] = cust_feature_vec
                feature_vec[self.d + len(cust_feature_vec) :] = train_feature_vec
                feature_matrix[cnt] = feature_vec
                train_idx = cnt
                cnt += 1

            if self.trans_df.iloc[i]["SM_AV"] == 1:
                feature_vec = np.zeros(feature_vec_length)
                feature_vec[len(cust_feature_vec) + 1] = 1
                feature_vec[: len(cust_feature_vec)] = cust_feature_vec
                feature_vec[self.d + len(cust_feature_vec) :] = sm_feature_vec
                feature_matrix[cnt] = feature_vec
                sm_idx = cnt
                cnt += 1

            if self.trans_df.iloc[i]["CAR_AV"] == 1:
                feature_vec = np.zeros(feature_vec_length)
                feature_vec[len(cust_feature_vec) + 2] = 1
                feature_vec[: len(cust_feature_vec)] = cust_feature_vec
                feature_vec[self.d + len(cust_feature_vec) :] = car_feature_vec
                feature_matrix[cnt] = feature_vec
                car_idx = cnt
                cnt += 1

            if choice_df.iloc[i]["CHOICE"] == 0:
                choice_list.append(train_idx)
            elif choice_df.iloc[i]["CHOICE"] == 1:
                choice_list.append(sm_idx)
            else:
                choice_list.append(car_idx)

            feature_matrix_list.append(feature_matrix)
            cardinality.append(cnt)

        feature_matrix_list = np.array(feature_matrix_list)
        self.X = torch.tensor(feature_matrix_list, dtype=torch.float32)

        choice_list = torch.tensor(choice_list)
        self.y = torch.zeros(len(choice_list), self.d, dtype=torch.float32)
        self.y[torch.arange(len(choice_list)), choice_list] = 1
        self.cardinality = torch.tensor(cardinality, dtype=torch.float32)
        self.N = len(self.X)

        self.feature_vec_length = feature_vec_length

        print("Preprocessing is Done!")
        

        
    def rkhs_preprocess(self):

        print("Preprocessing Raw Data Using RKHS Setup...")
        encoder = LabelEncoder()
        scaler = MinMaxScaler()

        # drop the columns that are not used
        drop = ["ID", "SP", "SM_SEATS", "GROUP"]

        self.data.loc[self.data["INCOME"] == 1, ["INCOME"]] = 0
        self.data.drop(drop, axis=1, inplace=True)

        # clean the unknown data
        self.data = self.data[
            (self.data["CHOICE"] != 0)
            & (self.data["WHO"] != 0)
            & (self.data["AGE"] != 6)
            & (self.data["INCOME"] != 4)
        ]

        # customer data
        self.cust_df = self.data.copy()
        self.cust_df = self.cust_df.drop(
            columns=[
                "TRAIN_TT",
                "TRAIN_CO",
                "TRAIN_HE",
                "SM_TT",
                "SM_CO",
                "SM_HE",
                "CAR_TT",
                "CAR_CO",
                "CHOICE",
                "TRAIN_AV",
                "CAR_AV",
                "SM_AV",
            ]
        )

        # choice data
        choice_df = self.data[["CHOICE"]]
        choice_df.loc[:, "CHOICE"] = encoder.fit_transform(choice_df["CHOICE"])

        # transportation feature data
        self.trans_df = self.data[
            [
                "TRAIN_AV",
                "CAR_AV",
                "SM_AV",
                "TRAIN_TT",
                "TRAIN_CO",
                "TRAIN_HE",
                "SM_TT",
                "SM_CO",
                "SM_HE",
                "CAR_TT",
                "CAR_CO",
            ]
        ]

        # these columns are converted to one-hot encoding
        self.cust_df = pd.get_dummies(
            self.cust_df,
            columns=[
                "PURPOSE",
                "FIRST",
                "TICKET",
                "WHO",
                "MALE",
                "GA",
                "ORIGIN",
                "DEST",
            ],
        )

        # age, luggage, income are scaled
        self.cust_df["AGE"] = encoder.fit_transform(self.cust_df["AGE"])
        self.cust_df["AGE"] = scaler.fit_transform(self.cust_df[["AGE"]])
        self.cust_df["LUGGAGE"] = encoder.fit_transform(self.cust_df["LUGGAGE"])
        self.cust_df["INCOME"] = encoder.fit_transform(self.cust_df["INCOME"])
        self.cust_df = self.cust_df.astype("float")

        # Set car's HE to zero
        self.trans_df.loc[:, ["CAR_HE"]] = 0

        # scale the trainsportation features
        self.trans_df[["TRAIN_TT", "SM_TT", "CAR_TT"]] = (
            self.trans_df[["TRAIN_TT", "SM_TT", "CAR_TT"]]
            .astype(float)
            .apply(lambda x: x / 1000)
        )
        self.trans_df[["CAR_CO", "TRAIN_CO", "SM_CO"]] = (
            self.trans_df[["CAR_CO", "TRAIN_CO", "SM_CO"]]
            .astype(float)
            .apply(lambda x: x / 5000)
        )

        self.trans_df[["TRAIN_HE", "SM_HE", "CAR_HE"]] = (
            self.trans_df[["TRAIN_HE", "SM_HE", "CAR_HE"]]
            .astype(float)
            .apply(lambda x: x / 100)
        )

        # The following ugly code are used to transform the feature vecs to feature matrix , concat the customer and transportation feature vectors
        feature_matrix_list = []
        choice_list = []
        cardinality = []
        self.cust_feature_length = len(self.cust_df.columns)
        self.trans_feature_length = len(self.trans_df.columns)
        self.cust_feature_matrix = np.array(self.cust_df)
        self.trans_feature_matrix = np.array(self.trans_df)
        self.feature_tensor = torch.tensor(
            np.hstack((self.cust_feature_matrix, self.trans_feature_matrix)),
            dtype=torch.float32,
        )
        for i in tqdm(range(len(self.cust_df)), desc="Preprocessing", unit="sample"):
            cust_feature_vec = self.cust_df.iloc[i].values
            trans_feature_vec = self.trans_df.iloc[i].values
            train_feature_vec = trans_feature_vec[3:6]
            sm_feature_vec = trans_feature_vec[6:9]
            car_feature_vec = trans_feature_vec[9:]
            choice = choice_df.iloc[i].values[0]
            feature_vec_length = len(cust_feature_vec) + len(train_feature_vec) + self.d

            feature_matrix = np.zeros((3, feature_vec_length))

            cnt = 0

            train_idx = -1
            sm_idx = -1
            car_idx = -1

            if self.trans_df.iloc[i]["TRAIN_AV"] == 1:
                feature_vec = np.zeros(feature_vec_length)
                feature_vec[len(cust_feature_vec)] = 1
                feature_vec[: len(cust_feature_vec)] = cust_feature_vec
                feature_vec[self.d + len(cust_feature_vec) :] = train_feature_vec
                feature_matrix[cnt] = feature_vec
                train_idx = cnt
                cnt += 1

            if self.trans_df.iloc[i]["SM_AV"] == 1:
                feature_vec = np.zeros(feature_vec_length)
                feature_vec[len(cust_feature_vec) + 1] = 1
                feature_vec[: len(cust_feature_vec)] = cust_feature_vec
                feature_vec[self.d + len(cust_feature_vec) :] = sm_feature_vec
                feature_matrix[cnt] = feature_vec
                sm_idx = cnt
                cnt += 1

            if self.trans_df.iloc[i]["CAR_AV"] == 1:
                feature_vec = np.zeros(feature_vec_length)
                feature_vec[len(cust_feature_vec) + 2] = 1
                feature_vec[: len(cust_feature_vec)] = cust_feature_vec
                feature_vec[self.d + len(cust_feature_vec) :] = car_feature_vec
                feature_matrix[cnt] = feature_vec
                car_idx = cnt
                cnt += 1

            if choice_df.iloc[i]["CHOICE"] == 0:
                choice_list.append(train_idx)
            elif choice_df.iloc[i]["CHOICE"] == 1:
                choice_list.append(sm_idx)
            else:
                choice_list.append(car_idx)

            feature_matrix_list.append(feature_matrix)
            cardinality.append(cnt)

        feature_matrix_list = np.array(feature_matrix_list)
        self.X = torch.tensor(feature_matrix_list, dtype=torch.float32)

        choice_list = torch.tensor(choice_list)
        self.y = torch.zeros(len(choice_list), self.d, dtype=torch.float32)
        self.y[torch.arange(len(choice_list)), choice_list] = 1
        self.cardinality = torch.tensor(cardinality, dtype=torch.float32)
        self.N = len(self.X)

        self.feature_vec_length = feature_vec_length

        print("Preprocessing is Done!")
