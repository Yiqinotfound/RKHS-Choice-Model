import torch
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from models.RF_feature_based_model import GaussianRFFB
from data_preprocess.swissmetros_data_loader import SwissMetroDataset
from sklearn.model_selection import KFold, train_test_split
from utils.model_utils import load_config, save_to_csv





def cross_validate(dataset: SwissMetroDataset, model_args: dict):
    temp_idx, test_idx = train_test_split(
        range(datasize), test_size=0.1, random_state=model_args["SEED"]
    )

    kf = KFold(
        n_splits=model_args["n_splits"], shuffle=True, random_state=model_args["SEED"]
    )

    fold=1
    for train_idx, val_idx in kf.split(temp_idx):
        dataset_train = dataset[train_idx]
        dataset_val = dataset[val_idx]
        dataset_test = dataset[test_idx]
        print(f"Train size: {len(dataset_train[0])}, Val size: {len(dataset_val[0])}, Test size: {len(dataset_test[0])}")

        model = GaussianRFFB(model_args=model_args, feature_length=feature_length, device=device)
        model.optimize_distribution(
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test,
        )   
        model.fit()
        model.evaluate(model.best_theta)
        data_rows = [
            [fold]+ list(model_args.values()) + [model.nll_train, model.nll_val, model.nll_test, model.acc_train, model.acc_val, model.acc_test, model.train_time]
        ] 
        header_columns = ["fold"] + list(model_args.keys()) + ["nll_train", "nll_val", "nll_test", "acc_train", "acc_val", "acc_test", "train_time"]
        save_to_csv(data_rows, header_columns, save_path)
        fold += 1



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_args = load_config("configs/exp_config.yaml")["swiss_metro_rf_args"]
data_path = "data/swissmetro/swissmetro.dat"
save_path = "real_data_experiments/swissmetro_RF_results/swissmetro_RF_results_with_train_time.csv"
dataset = SwissMetroDataset(filepath=data_path, preprocess_mode="rumnet")

datasize, d, feature_length = dataset.X.shape

cross_validate(dataset=dataset, model_args=model_args)

