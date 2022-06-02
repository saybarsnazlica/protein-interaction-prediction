import codecs
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
)


def read_queries(query_ids_file: str) -> str:
    with open(query_ids_file) as handle:
        for idx in handle:
            yield idx.strip()


def read_poses(input_file: str, n_pose=70_001):
    targets, feats = [], []
    reader = codecs.getreader("utf-8")
    archive = tarfile.open(input_file, "r:gz")
    f = reader(archive.extractfile(archive.getmembers()[0]))
    content = f.readlines()
    pose_count = 0
    for line in content:
        if line.startswith("POSE"):
            (
                header,
                index,
                ipose,
                irmsd,
                lig_rmsd,
                rec_prec,
                lig_prec,
                epiper,
                *feat_vec,
            ) = line.strip().split()

            mean_prec = np.mean((float(rec_prec) + float(lig_prec)))
            targets.append(np.array([float(irmsd), float(lig_rmsd), mean_prec]))
            pose_count += 1
            if pose_count > n_pose:
                break

            feats.append(np.array(feat_vec, float))
    archive.close()

    return np.stack(feats), np.stack(targets)


def predict_target(target_feat, regressor_path: str):
    regressor = xgb.XGBRegressor()
    regressor.load_model(regressor_path)
    target_scores = regressor.predict(target_feat)
    return target_scores


def binarize(target_scores):
    return np.where(target_scores >= 0.23, 1, 0)


def reduce_dimensions(features, svd_model_path: str):
    with open(svd_model_path, "rb") as model:
        svd = pickle.load(model)
    return svd.transform(features)


def prep_data(input_files: list, regressor_path: str, svd_model_path: str):
    feat_data = []
    target_data = []

    for input_file in input_files:
        try:
            features, target_feat = read_poses(input_file)
        except Exception as err:
            print(f"Error: {err} in preparing {input_file}")
        else:
            target_scores = predict_target(target_feat, regressor_path)
            binary_target = binarize(target_scores)
            # features = reduce_dimensions(features, svd_model_path)
            target_data.append(binary_target)
            feat_data.append(features)

    all_features = np.concatenate(feat_data)
    all_targets = np.concatenate(target_data)
    all_features = all_features.astype(np.float32)
    all_targets = all_targets.astype(np.float32)

    return all_features, all_targets


def prepare_features_single_query(
    query, training_inputs_path: str, svd_model_path: str
):
    input_file = Path(training_inputs_path) / query / "poses.tar.gz"
    features, _ = read_poses(input_file)
    reduced_features = reduce_dimensions(features, svd_model_path)

    return reduced_features.astype(np.float32)


def plot_roc_curve(actuals, predictions, path):
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    fpr, tpr, _ = roc_curve(actuals, predictions)
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(path)
    plt.show()


def plot_pr_curve(actuals, predictions, path):
    plt.figure()
    precision, recall, _ = precision_recall_curve(actuals, predictions)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(path)
    plt.show()
