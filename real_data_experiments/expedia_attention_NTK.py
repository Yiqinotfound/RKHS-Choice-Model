import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from data_preprocess.Expedia_data_loader import ExpediaDataset
dataset = ExpediaDataset(data_path="data/expedia")
from models.AttentionNTKModel import AttentionNTKChoiceModel
from utils.model_utils import report_memory
import torch

device = torch.device("cuda")
datasize = len(dataset)
print(f"Has total {datasize} samples!")
d = dataset.feature_vec_length
d0 = dataset.feature_vec_length
d2 = dataset.feature_vec_length




