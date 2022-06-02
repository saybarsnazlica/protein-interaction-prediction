#!/usr/bin/env python
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from torch import vstack
from torch.nn import (
    BCELoss,
    Dropout,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
)
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from utils import plot_pr_curve, plot_roc_curve, prep_data, read_queries

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PROJECT_DIR = Path().cwd().resolve()
MODEL_DIR_PATH = PROJECT_DIR / "models" / "pytorch"
TRAIN_INPUTS_ORIGINAL = PROJECT_DIR / "work" / "training_data"
TRAIN_INPUTS_RANDROT = PROJECT_DIR / "work" / "training_data_randrot"
SVD_MODEL_PATH = PROJECT_DIR / "models" / "svd_model_500_20220515_000045"
REGRESSOR_PATH = PROJECT_DIR / "models" / "dockq_regressor.json"

DATASET_IDS_ORIG = PROJECT_DIR / "work" / "input_lists" / "train_orig"
DATASET_IDS_ROT = PROJECT_DIR / "work" / "input_lists" / "train_rot"

TRAIN_EVALS_PATH = PROJECT_DIR / "reports" / f"train_{TIMESTAMP}.csv"
NORMALIZER_PATH = MODEL_DIR_PATH / f"normalizer_{TIMESTAMP}.pkl"
ROC_CURVE_FIG = (
    PROJECT_DIR / "reports" / "figures" / f"roc_curve_{TIMESTAMP}.png"
)
PR_CURVE_FIG = PROJECT_DIR / "reports" / "figures" / f"pr_curve_{TIMESTAMP}.png"


@dataclass
class TrainParameter:
    epoch: int
    inputs_dim: int
    learning_rate: float


class PoseDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return [self.x[idx, :], self.y[idx]]


class Net(Module):
    def __init__(self, n_inputs):
        super(Net, self).__init__()
        self.layers = Sequential(
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


def create_sampler(labels):
    class_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1.0 / class_count
    sample_weight = np.array([weight[int(t)] for t in labels])
    sample_weight = torch.from_numpy(sample_weight)
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight))

    return sampler


def prepare_dataset(queries, is_train_set=False, normalizer=None):
    query_paths = []

    for query in queries:
        if query.endswith("_orig"):
            query_file = (
                Path(TRAIN_INPUTS_ORIGINAL) / query.strip("_orig") / "poses.tar.gz"
            )
        elif query.endswith("_rot"):
            query_file = (
                Path(TRAIN_INPUTS_RANDROT) / query.strip("_rot") / "poses.tar.gz"
            )
        else:
            raise Exception("Error: Invalid query")
        query_paths.append(query_file)

    feat, label = prep_data(query_paths, REGRESSOR_PATH, SVD_MODEL_PATH)

    if is_train_set:
        normalizer = Normalizer()
        feat = normalizer.fit_transform(feat)
        sampler = create_sampler(label)
    else:
        feat = normalizer.transform(feat)

    feat = torch.from_numpy(feat)
    label = torch.from_numpy(label)
    dataset = PoseDataset(feat, label)

    if is_train_set:
        dl = DataLoader(dataset, batch_size=64, sampler=sampler)
    else:
        dl = DataLoader(dataset, batch_size=64)

    return normalizer, dl


def calc_metrics_one_epoch(tag, epoch_predictions, epoch_actuals):
    binary_pred = [i.round() for i in epoch_predictions]
    y_pred, pred, act = (
        vstack(binary_pred),
        vstack(epoch_predictions),
        vstack(epoch_actuals),
    )
    acc = accuracy_score(act, y_pred)
    roc_auc = roc_auc_score(act, pred)
    precision, recall, _ = precision_recall_curve(act, pred)
    pr_auc = auc(recall, precision)
    metrics = {
        f"{tag}_accuracy": acc,
        f"{tag}_roc_auc": roc_auc,
        f"{tag}_pr_auc": pr_auc,
    }

    return metrics


def train_for_one_epoch(
    epoch_index, tb_writer, train_data_loader, loss_fn, optimizer, model
):
    running_loss = 0.0
    last_loss = 0.0
    pred_train_one_epoch, actual_train_one_epoch = [], []

    for i, (inputs, targets) in enumerate(train_data_loader):
        optimizer.zero_grad()
        y_hat = model(inputs)
        targets = targets.unsqueeze(-1)
        loss = loss_fn(y_hat, targets)
        loss.backward()
        optimizer.step()

        actual = targets
        actual = actual.reshape((len(actual), 1))
        pred_train_one_epoch.append(y_hat.detach())
        actual_train_one_epoch.append(actual)

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f"    batch {i + 1} loss: {last_loss}")
            tb_x = epoch_index * len(train_data_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    metrics_one_epoch = calc_metrics_one_epoch(
        "train", pred_train_one_epoch, actual_train_one_epoch
    )

    return last_loss, metrics_one_epoch


def train_model(train_dl, val_dl, train_param: TrainParameter):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/pose_trainer_{timestamp}")
    metrics_data = []

    model = Net(train_param.inputs_dim)
    loss_fn = BCELoss()
    optimizer = Adam(model.parameters(), lr=train_param.learning_rate)

    best_val_loss = 1_000_000.0
    epoch_number = 0
    for _ in range(train_param.epoch):
        pred_val, actual_val = [], []

        model.train()
        avg_loss, train_metrics = train_for_one_epoch(
            epoch_number, writer, train_dl, loss_fn, optimizer, model
        )

        model.train(False)
        running_val_loss = 0.0
        for i, (vinputs, vtargets) in enumerate(val_dl):
            voutputs = model(vinputs)
            vtargets = vtargets.unsqueeze(-1)
            vloss = loss_fn(voutputs, vtargets)
            running_val_loss += vloss

            vactual = vtargets
            vactual = vactual.reshape((len(vactual), 1))
            pred_val.append(voutputs.detach())
            actual_val.append(vactual)

        avg_val_loss = running_val_loss / (i + 1)
        print(f"Train loss: {avg_loss} Validation loss: {avg_val_loss}")

        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_val_loss},
            epoch_number + 1,
        )
        writer.flush()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = MODEL_DIR_PATH / f"model_{timestamp}_{epoch_number}.pth"
            torch.save(model.state_dict(), model_path)

        val_metrics = calc_metrics_one_epoch("val", pred_val, actual_val)
        train_val_metrics = {**train_metrics, **val_metrics}
        metrics_data.append(train_val_metrics)

        epoch_number += 1

    return pd.DataFrame(metrics_data), model_path


def test_model(test_dl, loaded_model):
    with torch.no_grad():
        predictions, actuals = [], []
        for _, (inputs, targets) in enumerate(test_dl):
            y_hat = loaded_model(inputs)
            actual = targets
            actual = actual.reshape((len(actual), 1))
            predictions.append(y_hat.detach())
            actuals.append(actual)

        test_metrics = calc_metrics_one_epoch("test", predictions, actuals)
        plot_roc_curve(vstack(actuals), vstack(predictions), ROC_CURVE_FIG)
        plot_pr_curve(vstack(actuals), vstack(predictions), PR_CURVE_FIG)

    return test_metrics


def train_test_val_split():
    orig_ids = list(read_queries(DATASET_IDS_ORIG))
    rot_ids = list(read_queries(DATASET_IDS_ROT))

    left_over_rot, test_ids = train_test_split(rot_ids, test_size=0.5, random_state=42)
    train_val_ids = orig_ids + left_over_rot

    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=0.10, random_state=42
    )

    return train_ids[:5], val_ids[:1], test_ids[:1]


def main():
    train_ids, val_ids, test_ids = train_test_val_split()

    norm, train_dl = prepare_dataset(train_ids, is_train_set=True)
    _, val_dl = prepare_dataset(val_ids, normalizer=norm)

    pickle.dump(norm, open(NORMALIZER_PATH, "wb"))
    norm_test = pickle.load(open(NORMALIZER_PATH, "rb"))
    _, test_dl = prepare_dataset(test_ids, normalizer=norm_test)

    print(
        f"Train set: {len(train_dl.dataset)}",
        f"Val set: {len(val_dl.dataset)}",
        f"Test set: {len(test_dl.dataset)}",
    )

    param = TrainParameter(
        epoch=10,
        inputs_dim=14_028,
        learning_rate=0.001,
    )

    # Begin Training
    train_metrics, model_path = train_model(train_dl, val_dl, param)
    train_metrics.to_csv(TRAIN_EVALS_PATH)

    # Begin Testing
    saved_model = Net(param.inputs_dim)
    saved_model.load_state_dict(torch.load(model_path))

    test_metrics = test_model(test_dl, saved_model)

    print(
        f'Accuracy: {test_metrics["test_accuracy"]:.3f}',
        f'ROC-AUC: {test_metrics["test_roc_auc"]:.3f}',
        f'PR-AUC: {test_metrics["test_pr_auc"]:.3f}',
    )


if __name__ == "__main__":
    main()
