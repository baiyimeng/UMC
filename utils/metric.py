import pandas as pd
import numpy as np
from sklearn.metrics import *


def get_gauc(y_true, y_pred, field_id):
    data = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "field_id": field_id})
    grouped_data = data.groupby("field_id")

    group_aucs = []
    group_sizes = []

    for field_id, group in grouped_data:
        group_y_true = group["y_true"].values
        group_y_pred = group["y_pred"].values
        if 0 < np.mean(group_y_true) < 1:
            auc = roc_auc_score(group_y_true, group_y_pred)
            group_aucs.append(auc)
            group_sizes.append(len(group))

    group_aucs = np.array(group_aucs)
    group_sizes = np.array(group_sizes)

    gauc = np.average(group_aucs, weights=group_sizes)
    return gauc


def get_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc


def get_logloss(y_true, y_pred):
    logloss = log_loss(y_true, y_pred)
    return logloss


def get_pcoc(y_true, y_pred):
    pcoc = np.sum(y_pred) / np.sum(y_true)
    return pcoc


def get_ece(y_true, y_pred, M=100):
    data = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "bin_id": (y_pred * M).astype("int32")}
    )
    q_curve = data.groupby("bin_id").agg(
        {
            "y_true": ["mean", "count"],
            "y_pred": ["mean"],
        }
    )

    ece = np.sum(
        np.abs(q_curve["y_true"]["mean"] - q_curve["y_pred"]["mean"])
        * q_curve["y_true"]["count"]
    )
    n = np.sum(q_curve["y_true"]["count"])
    return ece / n


def get_rce(y_true, y_pred, M=100):
    data = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "bin_id": (y_pred * M).astype("int32")}
    )
    q_curve = data.groupby("bin_id").agg(
        {
            "y_true": ["mean", "count"],
            "y_pred": ["mean"],
        }
    )
    # remove zeros
    q_curve = q_curve[q_curve["y_true"]["mean"] > 0]
    rce = np.sum(
        np.abs(q_curve["y_true"]["mean"] - q_curve["y_pred"]["mean"])
        * q_curve["y_true"]["count"]
        / (q_curve["y_true"]["mean"])
    )

    n = np.sum(q_curve["y_true"]["count"])
    return rce / n


def get_fece(y_true, y_pred, field_id, M=1):
    data = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "bin_id": (y_pred * M).astype("int32"),
            "field_id": field_id,
        }
    )
    q_curve = data.groupby(["field_id", "bin_id"], as_index=True).agg(
        {
            "y_true": ["mean", "count"],
            "y_pred": ["mean"],
        }
    )
    fece = np.sum(
        np.abs(q_curve["y_true"]["mean"] - q_curve["y_pred"]["mean"])
        * q_curve["y_true"]["count"]
    )
    n = np.sum(q_curve["y_true"]["count"])
    return fece / n


def get_frce(y_true, y_pred, field_id, M=1):
    data = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "bin_id": (y_pred * M).astype("int32"),
            "field_id": field_id,
        }
    )
    q_curve = data.groupby(["field_id", "bin_id"], as_index=True).agg(
        {
            "y_true": ["mean", "count"],
            "y_pred": ["mean"],
        }
    )
    # remove zeros
    q_curve = q_curve[q_curve["y_true"]["mean"] > 0]
    frce = np.sum(
        np.abs(q_curve["y_true"]["mean"] - q_curve["y_pred"]["mean"])
        * q_curve["y_true"]["count"]
        / (q_curve["y_true"]["mean"])
    )

    n = np.sum(q_curve["y_true"]["count"])
    return frce / n


def get_mfece(y_true, y_pred, x, M=1):
    result = []
    for field_id in x:
        field_id = np.squeeze(field_id)
        result.append(get_fece(y_true, y_pred, field_id, M))
    return np.array(result)


def get_mfrce(y_true, y_pred, x, M=1):
    result = []
    for field_id in x:
        field_id = np.squeeze(field_id)
        result.append(get_frce(y_true, y_pred, field_id, M))
    return np.array(result)
