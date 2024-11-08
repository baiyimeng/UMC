import os
import torch
import pandas as pd
import numpy as np
import random

import sys

global root
root = os.path.abspath(".")
sys.path.append(root)

from models.inputs import SparseFeat
import torch.utils.data as Data
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metric import *
import torch.nn.functional as F
from utils.metric import *
from sklearn.model_selection import train_test_split
from calib.DeepEnsemShapeCalib import DESC
from calib.MonotonicNN import UMC, UMNN
from calib.SelfBoostCalibRank import SBCR
from calib.NeuralCalib import NeuralCalib


def setup_seed(seed):
    import torch  # # Warning: Do not remove, as it will cause an error later!!!

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config(object):
    def __init__(self):

        self.data_root = "/data/baiyimeng/dataset"
        self.data_name = "avazu"
        self.model_name = "deepfm"
        self.batch_size = 1024 * 16
        self.dropout = 0.0
        self.init_std = 1e-4
        self.lr = 1e-3
        self.l2_reg = 0.0
        self.embedding_dim = 16
        self.seed = 1024
        self.filepath = "/data/baiyimeng/ckpt"

        self.method = None
        self.field_index = 2

        self.lr_calib = 1e-3
        self.epochs_calib = 20
        self.batch_size_calib = 1024 * 16


def get_data(data_path=None):
    if not data_path:
        raise ValueError("data_path must be provided")

    path = os.path.join(data_path, "data.pkl")
    data = pd.read_pickle(filepath_or_buffer=path)

    feature_names = list(data.columns)[:-1]  # only sparse features
    label_names = list(data.columns)[-1]
    print("Feature names:", feature_names)
    print("Label names:", label_names)

    return data, feature_names, label_names


def trial(config_update):
    config = Config()
    if config_update is not None:
        for name, value in config_update.items():
            setattr(config, name, value)

    experiment_name = ""
    experiment_info = [
        "data_name",
        "model_name",
        "batch_size",
        "dropout",
        "init_std",
        "lr",
        "l2_reg",
        "seed",
    ]

    for name in experiment_info:
        value = getattr(config, name)
        experiment_name += name + "=" + str(value) + "_"

    setup_seed(1024)

    data, feature_names, label_names = get_data(
        data_path=os.path.join(config.data_root, config.data_name)
    )
    train, valid_test = train_test_split(
        data, test_size=0.4, random_state=1024, shuffle=False
    )
    valid, test = train_test_split(
        valid_test, test_size=0.5, random_state=1024, shuffle=False
    )
    feature_columns = [
        SparseFeat(
            feat,
            vocabulary_size=int(data[feat].max()) + 1,
            embedding_dim=config.embedding_dim,
        )
        for feat in feature_names
    ]

    valid_x = {name: np.array(valid[name]) for name in feature_names}
    valid_y = np.transpose([np.array(valid[label_names])])
    test_x = {name: np.array(test[name]) for name in feature_names}
    test_y = np.transpose([np.array(test[label_names])])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = config.filepath + "/" + experiment_name + ".pth"
    model = torch.load(path)
    model = model.to(device)
    # fix all parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # valid and test dataloader
    if isinstance(valid_x, dict):
        valid_x = [valid_x[feature] for feature in model.feature_index]
    for i in range(len(valid_x)):
        if len(valid_x[i].shape) == 1:
            valid_x[i] = np.expand_dims(valid_x[i], axis=1)
    valid_tensor_data = Data.TensorDataset(
        torch.from_numpy(np.concatenate(valid_x, axis=-1)), torch.from_numpy(valid_y)
    )
    valid_loader = DataLoader(
        dataset=valid_tensor_data,
        shuffle=True,
        batch_size=config.batch_size_calib,
        pin_memory=False,
        generator=torch.Generator().manual_seed(1024),
    )

    if isinstance(test_x, dict):
        test_x = [test_x[feature] for feature in model.feature_index]
    for i in range(len(test_x)):
        if len(test_x[i].shape) == 1:
            test_x[i] = np.expand_dims(test_x[i], axis=1)
    test_tensor_data = Data.TensorDataset(
        torch.from_numpy(np.concatenate(test_x, axis=-1)), torch.from_numpy(test_y)
    )
    test_loader = DataLoader(
        dataset=test_tensor_data,
        shuffle=False,
        batch_size=config.batch_size_calib,
        pin_memory=False,
    )

    # evaluate function
    def evaluate(test_y_pred_calib, test_y, index):
        test_auc = get_auc(test_y, test_y_pred_calib)
        test_gauc = get_gauc(test_y, test_y_pred_calib, np.squeeze(test_x[index]))
        test_logloss = get_logloss(test_y, test_y_pred_calib)
        test_pcoc = get_pcoc(test_y, test_y_pred_calib)
        test_ece = get_ece(test_y, test_y_pred_calib, 100)
        test_fece = get_fece(test_y, test_y_pred_calib, np.squeeze(test_x[index]), 1)
        fece_list = get_mfece(test_y, test_y_pred_calib, test_x, 1)
        test_mfece = np.mean(fece_list)
        test_rce = get_rce(test_y, test_y_pred_calib, 100)
        test_frce = get_frce(test_y, test_y_pred_calib, np.squeeze(test_x[index]), 1)
        rce_list = get_mfrce(test_y, test_y_pred_calib, test_x, 1)
        test_mfrce = np.mean(rce_list)

        log = f"test_auc = {test_auc:.6f}, test_gauc = {test_gauc:.6f}, test_logloss = {test_logloss:.6f}, test_pcoc = {test_pcoc:.6f}, test_ece = {test_ece:.6f}, test_fece = {test_fece:.6f}, test_mfece = {test_mfece:.6f}, test_rce = {test_rce:.6f}, test_frce = {test_frce:.6f}, test_mfrce = {test_mfrce:.6f}"
        # fece_log = f"multi-field fece list: {fece_list}"
        # rce_log = f"multi-field rce list: {rce_list}"
        print(log)
        # print(fece_log)
        # print(rce_log)

    # different methods, based on binning or distribution
    if config.method == "neu":
        model_calib = NeuralCalib(
            [200, 200], feature_columns, model.feature_index, device, 100
        )
    elif config.method == "desc":
        model_calib = DESC(
            [200, 200], feature_columns, model.feature_index, device, 100
        )
    elif config.method == "sbcr":
        model_calib = SBCR(
            [256, 128, 128], feature_columns, model.feature_index, device, 100
        )
    elif config.method == "umnn":
        model_calib = UMNN([50, 50], device, 50)
    elif config.method == "umc_wor":
        model_calib = UMC(
            [50, 50], feature_columns, model.feature_index, device, 50, False
        )
    elif config.method == "umc":
        model_calib = UMC(
            [50, 50], feature_columns, model.feature_index, device, 50, True
        )
    else:
        return NotImplementedError("Not implement")

    optim = torch.optim.Adam(model_calib.parameters(), lr=config.lr_calib)

    K = 10
    beta = 0.99
    acc_cache = torch.zeros(K).to(device)
    con_cache = torch.zeros(K).to(device)
    num_cache = torch.zeros(K).to(device)
    lam = 1e-3

    for epoch in range(config.epochs_calib):
        for step, (x_valid, y_valid) in tqdm(enumerate(valid_loader), disable=1):
            x = x_valid.to(device).float()
            y = y_valid.to(device).float()
            optim.zero_grad()
            y_pred = model(x)
            logit_calib = (
                model_calib(x, torch.logit(y_pred), model.embedding_dict)
                if config.method != "umnn"
                else model_calib(torch.logit(y_pred))
            )
            y_pred_calib = torch.sigmoid(logit_calib)
            aux_loss = model_calib.compute_aux_loss() if config.method == "neu" else 0.0

            bin_boundaries = torch.linspace(0, 1, K + 1).to(device)
            bin_indices = torch.bucketize(y_pred_calib, bin_boundaries[1:-1])
            loss_calib = 0.0
            for bin_idx in range(K):
                mask = bin_indices == bin_idx
                bin_samples = torch.sum(mask)
                if bin_samples == 0:
                    continue
                bin_accuracy = torch.mean(y[mask])
                bin_confidence = torch.mean(y_pred_calib[mask])
                acc_update = bin_accuracy * (1 - beta) + acc_cache[bin_idx] * beta
                con_update = bin_confidence * (1 - beta) + con_cache[bin_idx] * beta
                num_update = bin_samples * (1 - beta) + num_cache[bin_idx] * beta
                loss_calib += ((acc_update - con_update) ** 2) * num_update
                acc_cache[bin_idx], con_cache[bin_idx], num_cache[bin_idx] = (
                    acc_update.detach(),
                    con_update.detach(),
                    num_update.detach(),
                )

            loss = (
                F.binary_cross_entropy_with_logits(logit_calib, y)
                + aux_loss
                + loss_calib * lam
            )
            loss.backward()
            optim.step()

        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        with torch.no_grad():
            for _, (x_test, y_test) in tqdm(enumerate(test_loader), disable=1):
                x = x_test.to(device).float()
                y = y_test.to(device).float()
                test_y.append(y.cpu().data.numpy())
                y_pred = model(x)
                test_y_pred.append(y_pred.cpu().data.numpy())
                logit_calib = (
                    model_calib(x, torch.logit(y_pred), model.embedding_dict)
                    if config.method != "umnn"
                    else model_calib(torch.logit(y_pred))
                )
                y_pred_calib = F.sigmoid(logit_calib)
                test_y_pred_calib.append(y_pred_calib.cpu().data.numpy())
        test_y = np.concatenate(test_y).astype("float32").flatten()
        test_y_pred = np.concatenate(test_y_pred).astype("float32").flatten()
        test_y_pred_calib = (
            np.concatenate(test_y_pred_calib).astype("float32").flatten()
        )

        # evaluate(test_y_pred, test_y, config.field_index)
        evaluate(test_y_pred_calib, test_y, config.field_index)


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    config_update = {
        "data_name": "aliccp",
        "model_name": "deepfm",
        "batch_size": 1024 * 16,
        "dropout": 0.0,
        "init_std": 1e-4,
        "lr": 5e-4,
        "l2_reg": 1e-6,
        "method": "umc",
        "field_index": 0,
        "lr_calib": 1e-3,
        "epochs_calib": 20,
        "batch_size_calib": 1024 * 16,
    }
    trial(config_update=config_update)


## Best hyper-parameters
# K = 10
# beta = 0.95
# lam = 1e-2
