import os
import csv
import torch
import gc
import sys
import yaml
sys.path.append("/data/yiqi/RKHSChoiceModel/")


def save_to_csv(data_rows: list, header_columns: list, file_path: str):
    if len(header_columns) != len(data_rows[0]):
        raise ValueError("pikabu")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    file_exists = os.path.exists(file_path)
    file_is_empty = not os.path.isfile(file_path) or os.stat(file_path).st_size == 0

    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists or file_is_empty:
            writer.writerow(header_columns)

        writer.writerows(data_rows)


def report_memory(device):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
    reserved_memory = torch.cuda.memory_reserved(device) / (1024**3)
    print(
        f" {device} | Allocated Memory: {allocated_memory:.2f} GB | Reserved Memory: {reserved_memory:.2f} GB"
    )

def load_config(config_path:str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config