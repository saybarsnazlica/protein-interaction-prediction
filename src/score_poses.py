#!/usr/bin/env python
import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import Dropout, Flatten, Linear, Module, ReLU, Sequential, Sigmoid

from utils import prepare_features_single_query, read_queries

PROJECT_DIR = Path().cwd().resolve()
SVD_MODEL_PATH = PROJECT_DIR / "models" / "svd_model_500"
REGRESSOR_PATH = PROJECT_DIR / "models" / "dockq_regressor.json"
ML_MODEL_PATH = PROJECT_DIR / "models" / "pytorch" / "model_20220505_183633_4.pth"
NORMALIZER_PATH = (
    PROJECT_DIR / "models" / "pytorch" / "normalizer161_20220505_114320.pkl"
)

TRAIN_INPUTS_PATH = PROJECT_DIR / "work" / "training_data"
DATASET_IDS = PROJECT_DIR / "work" / "input_lists" / "train_test_161"


class Net(Module):
    def __init__(self, n_inputs):
        super(Net, self).__init__()
        self.layers = Sequential(
            Flatten(),
            Linear(n_inputs, 64),
            Dropout(),
            ReLU(),
            Linear(64, 32),
            Dropout(),
            ReLU(),
            Linear(32, 1),
            Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


def calculate_a_score(reduced_mat):
    pytorch_model = Net(500)
    checkpoint = torch.load(ML_MODEL_PATH)
    pytorch_model.load_state_dict(checkpoint)
    pytorch_model.eval()
    norm = pickle.load(open(NORMALIZER_PATH, "rb"))
    reduced_mat = reduced_mat.reshape(1, -1)
    reduced_mat = norm.transform(reduced_mat)
    reduced_mat = torch.from_numpy(reduced_mat.astype(np.float32))
    a_score = pytorch_model(reduced_mat)

    return a_score.item()


def make_output_dir(output_dir: str):
    output_directory = Path(output_dir).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)


def extract_tar_file(query_path, query_idx):
    out_dir = query_path / f"{query_idx}_tmp"
    make_output_dir(out_dir)

    with tarfile.open(next(query_path.glob("*.tar.xz"))) as tgz:
        tgz.extractall(out_dir)


def read_ft(ft_file_path):
    ft_data = []
    with open(ft_file_path) as handle:
        for line in handle:
            ft_data.append(line)

    return ft_data


def add_a_score_to_ft(query_tmp_path, ft_file_data, a_scores):
    data_to_write = []
    for line in ft_file_data:
        line = line.strip()
        if line.startswith("lig_center"):
            data_to_write.append(line)
        else:
            data_to_write.append(line + f" {next(a_scores)}")

    with open(query_tmp_path / "ft.000.00_with_a_score", "w") as handle:
        for data in data_to_write:
            handle.write(data + "\n")


def generate_a_scores(query):
    query_features = prepare_features_single_query(
        query, TRAIN_INPUTS_PATH, SVD_MODEL_PATH
    )

    for i in range(len(query_features)):
        a_score = calculate_a_score(query_features[i][:])
        yield a_score


def main():
    queries = read_queries(DATASET_IDS)

    for query in queries:
        work_dir = TRAIN_INPUTS_PATH / query / f"{query}_tmp"
        a_score_gen = generate_a_scores(query)
        extract_tar_file(TRAIN_INPUTS_PATH / query, query)
        ft_info = read_ft(work_dir / "ft.000.00")
        add_a_score_to_ft(work_dir, ft_info, a_score_gen)


if __name__ == "__main__":
    main()
