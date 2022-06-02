#!/usr/bin/env python
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from datetime import datetime
from utils import read_poses, read_queries


PROJECT_DIR = Path().cwd().resolve()
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUTS_DIR = PROJECT_DIR / "work" / "training_data"
TRAIN_SET_FILE = PROJECT_DIR / "work" / "input_lists" / "train_test_161"
MODELS_DIR = PROJECT_DIR / "models"

N_FEAT = 500


def train():
    vector_data = []
    train = list(read_queries(TRAIN_SET_FILE))[:100]

    for query in train:
        pose_input_path = OUTPUTS_DIR / query / "poses.tar.gz"

        try:
            feat_vector, _ = read_poses(pose_input_path, 10_001)
        except Exception as err:
            print(f"Error in {query}: {err}")
        else:
            if feat_vector.shape[1] == 14028:
                vector_data.append(feat_vector)

        features_matrix = np.concatenate(vector_data)

        print(f"N_FEAT: {N_FEAT}\tFeat Dim: {features_matrix.shape}")

        svd = TruncatedSVD(n_components=N_FEAT)
        svd.fit(features_matrix)

    print(f"EVS: {svd.explained_variance_ratio_.sum()}")

    with open(MODELS_DIR / f"svd_model_{N_FEAT}_{TIMESTAMP}", "wb") as svd_model:
        pickle.dump(svd, svd_model)


if __name__ == "__main__":
    train()
