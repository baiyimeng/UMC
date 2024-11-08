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
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import LinearConstraint, Bounds
from clogistic import LogisticRegression


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
        dataset=valid_tensor_data, shuffle=False, batch_size=1024 * 16, pin_memory=False
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
        dataset=test_tensor_data, shuffle=False, batch_size=1024 * 16, pin_memory=False
    )

    # get valid y numpy list
    valid_y = []
    valid_y_pred = []
    valid_logit_pred = []
    for step, (x_valid, y_valid) in tqdm(enumerate(valid_loader)):
        x = x_valid.to(device).float()
        y = y_valid.to(device).float()
        valid_y.append(y.cpu().data.numpy())
        y_pred = model(x)
        valid_y_pred.append(y_pred.cpu().data.numpy())
        logit_pred = torch.logit(model(x)).cpu().data.numpy()
        valid_logit_pred.append(logit_pred)
    valid_y = np.concatenate(valid_y).astype("float32").flatten()
    valid_y_pred = np.concatenate(valid_y_pred).astype("float32").flatten()
    valid_logit_pred = np.concatenate(valid_logit_pred).astype("float32").flatten()

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
    if config.method == "hb":
        num_bins = 10
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        confidences = valid_y_pred
        indices = np.digitize(confidences, bins, right=True)
        bin_accuracies = np.zeros(num_bins, dtype=np.float32)
        for b in range(num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies[b] = np.mean(valid_y.flatten()[selected])

        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        with torch.no_grad():
            for step, (x_test, y_test) in tqdm(enumerate(test_loader)):
                x = x_test.to(device).float()
                y = y_test.to(device).float()
                test_y.append(y.cpu().data.numpy())
                y_pred = model(x)
                test_y_pred.append(y_pred.cpu().data.numpy())
                indices = np.digitize(y_pred.cpu().data.numpy(), bins, right=True)
                y_pred_calib = bin_accuracies[indices - 1]
                test_y_pred_calib.append(y_pred_calib)

    elif config.method == "ir":
        model_calib = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        model_calib.fit(valid_y_pred, valid_y)

        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        with torch.no_grad():
            for step, (x_test, y_test) in tqdm(enumerate(test_loader)):
                x = x_test.to(device).float()
                y = y_test.to(device).float()
                test_y.append(y.cpu().data.numpy())
                y_pred = model(x)
                test_y_pred.append(y_pred.cpu().data.numpy())
                y_pred_calib = model_calib.predict(y_pred.cpu().data.numpy())
                test_y_pred_calib.append(y_pred_calib)

    elif config.method == "sir":
        num_bins = 10
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        confidences = valid_y_pred.flatten()
        indices = np.digitize(confidences, bins, right=True)
        bin_accuracies = np.zeros(num_bins, dtype=np.float32)
        bin_count = np.zeros(num_bins, dtype=np.float32)
        bin_min = np.zeros(num_bins, dtype=np.float32)
        bin_max = np.zeros(num_bins, dtype=np.float32)
        for b in range(num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies[b] = np.mean(valid_y.flatten()[selected])
                bin_count[b] = len(selected)
                bin_min[b] = np.min(confidences.flatten()[selected])
                bin_max[b] = np.max(confidences.flatten()[selected])
        bin_mid = (bin_min + bin_max) / 2

        i = 0
        while i < len(bin_mid) - 1:
            if bin_accuracies[i] > bin_accuracies[i + 1]:
                bin_mid[i] = (
                    bin_count[i] * bin_mid[i] + bin_count[i + 1] * bin_mid[i + 1]
                ) / (bin_count[i] + bin_count[i + 1])
                bin_accuracies[i] = (
                    bin_count[i] * bin_accuracies[i]
                    + bin_count[i + 1] * bin_accuracies[i + 1]
                ) / (bin_count[i] + bin_count[i + 1])
                bin_mid = np.delete(bin_mid, i + 1, 0)
                bin_accuracies = np.delete(bin_accuracies, i + 1, 0)
                bin_count = np.delete(bin_count, i + 1, 0)
            else:
                i = i + 1

        bin_mid = np.insert(np.append(bin_mid, 1.0), 0, 0.0)
        bin_accuracies = np.insert(np.append(bin_accuracies, 1.0), 0, 0.0)
        weight = np.diff(bin_accuracies) / (np.diff(bin_mid) + 1e-4)
        bias = bin_accuracies[1:] - weight * bin_mid[1:]

        assert np.all(np.diff(bin_accuracies) >= 0)

        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        with torch.no_grad():
            for step, (x_test, y_test) in tqdm(enumerate(test_loader)):
                x = x_test.to(device).float()
                y = y_test.to(device).float()
                test_y.append(y.cpu().data.numpy())
                y_pred = model(x)
                test_y_pred.append(y_pred.cpu().data.numpy())
                indices = np.digitize(y_pred.cpu().data.numpy(), bin_mid, right=True)
                y_pred_calib = (
                    weight[indices - 1] * y_pred.cpu().data.numpy() + bias[indices - 1]
                )
                test_y_pred_calib.append(y_pred_calib)

    elif config.method == "platt":
        valid_logit_pred = valid_logit_pred.reshape(-1, 1)
        X = valid_logit_pred
        lb = np.array([0])
        ub = np.array([np.inf])
        A = np.zeros((1, 2))
        A[0, 0] = np.array(1)
        cons = LinearConstraint(A, lb, ub)
        lb = np.array([-np.inf, -np.inf])
        ub = np.array([np.inf, np.inf])
        bounds = Bounds(lb, ub)
        model_calib = LogisticRegression(solver="ecos")
        model_calib.fit(X, valid_y, bounds=bounds, constraints=cons)
        a = torch.tensor(model_calib.coef_[0][0]).to(device)
        b = torch.tensor(model_calib.intercept_).to(device)

        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        with torch.no_grad():
            for step, (x_test, y_test) in tqdm(enumerate(test_loader)):
                x = x_test.to(device).float()
                y = y_test.to(device).float()
                test_y.append(y.cpu().data.numpy())
                y_pred = model(x)
                test_y_pred.append(y_pred.cpu().data.numpy())
                logit_calib = a * torch.logit(y_pred) + b
                y_pred_calib = F.sigmoid(logit_calib).cpu().data.numpy()
                test_y_pred_calib.append(y_pred_calib)

    elif config.method == "gauss":
        valid_logit_pred = valid_logit_pred.reshape(-1, 1)
        smin, smax = -13.0, 13.0
        X = np.concatenate((np.power(valid_logit_pred, 2), valid_logit_pred), axis=1)
        lb = np.array([0, 0])
        ub = np.array([np.inf, np.inf])
        A = np.zeros((2, X.shape[1] + 1))
        A[0, :2] = np.array([2 * smin, 1.0])
        A[1, :2] = np.array([2 * smax, 1.0])
        cons = LinearConstraint(A, lb, ub)
        lb = np.r_[-np.inf, -np.inf, -np.inf]
        ub = np.r_[np.inf, np.inf, np.inf]
        bounds = Bounds(lb, ub)
        model_calib = LogisticRegression(solver="ecos")
        model_calib.fit(X, valid_y, bounds=bounds, constraints=cons)
        a = torch.tensor(model_calib.coef_[0][0]).to(device)
        b = torch.tensor(model_calib.coef_[0][1]).to(device)
        c = torch.tensor(model_calib.intercept_).to(device)

        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        with torch.no_grad():
            for step, (x_test, y_test) in tqdm(enumerate(test_loader)):
                x = x_test.to(device).float()
                y = y_test.to(device).float()
                test_y.append(y.cpu().data.numpy())
                y_pred = model(x)
                test_y_pred.append(y_pred.cpu().data.numpy())
                logit_calib = (
                    a * torch.pow(torch.logit(y_pred), 2) + b * torch.logit(y_pred) + c
                )
                y_pred_calib = F.sigmoid(logit_calib).cpu().data.numpy()
                test_y_pred_calib.append(y_pred_calib)

    elif config.method == "gamma":
        valid_logit_pred = valid_logit_pred.reshape(-1, 1)
        shift = 13.0
        smin, smax = 1e-4, 2 * shift
        X = np.concatenate(
            (np.log(valid_logit_pred + shift), valid_logit_pred + shift), axis=1
        )
        lb = np.array([0, 0])
        ub = np.array([np.inf, np.inf])
        A = np.zeros((2, X.shape[1] + 1))
        A[0, :2] = np.array([2 * smin, 1.0])
        A[1, :2] = np.array([2 * smax, 1.0])
        cons = LinearConstraint(A, lb, ub)
        lb = np.r_[-np.inf, -np.inf, -np.inf]
        ub = np.r_[np.inf, np.inf, np.inf]
        bounds = Bounds(lb, ub)
        model_calib = LogisticRegression(solver="ecos")
        model_calib.fit(X, valid_y, bounds=bounds, constraints=cons)
        a = torch.tensor(model_calib.coef_[0][0]).to(device)
        b = torch.tensor(model_calib.coef_[0][1]).to(device)
        c = torch.tensor(model_calib.intercept_).to(device)

        test_y = []
        test_y_pred = []
        test_y_pred_calib = []
        with torch.no_grad():
            for step, (x_test, y_test) in tqdm(enumerate(test_loader)):
                x = x_test.to(device).float()
                y = y_test.to(device).float()
                test_y.append(y.cpu().data.numpy())
                y_pred = model(x)
                test_y_pred.append(y_pred.cpu().data.numpy())
                logit_calib = (
                    a * torch.log(torch.logit(y_pred) + shift)
                    + b * (torch.logit(y_pred) + shift)
                    + c
                )
                y_pred_calib = F.sigmoid(logit_calib).cpu().data.numpy()
                test_y_pred_calib.append(y_pred_calib)

    else:
        return NotImplementedError("Not implement")

    test_y = np.concatenate(test_y).astype("float32").flatten()
    test_y_pred = np.concatenate(test_y_pred).astype("float32").flatten()
    test_y_pred_calib = np.concatenate(test_y_pred_calib).astype("float32").flatten()
    evaluate(test_y_pred, test_y, config.field_index)
    evaluate(test_y_pred_calib, test_y, config.field_index)


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    config_update = {
        "data_name": "avazu",
        "model_name": "deepfm",
        "batch_size": 1024 * 16,
        "dropout": 0.0,
        "init_std": 1e-4,
        "lr": 1e-3,
        "l2_reg": 1e-6,
        "method": "sir",
        "field_index": 2,
    }
    trial(config_update=config_update)
