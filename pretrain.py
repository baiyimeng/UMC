import os
import pandas as pd
import torch
import numpy as np
import random
import torch.backends

import sys

global root
root = os.path.abspath(".")
sys.path.append(root)

from models.callbacks import EarlyStopping, ModelCheckpoint
from models.inputs import SparseFeat
from models.deepfm import DeepFM
from sklearn.metrics import *
from ray import tune
from sklearn.model_selection import train_test_split
from utils.metric import *


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
        # data
        self.data_root = "/data/baiyimeng/dataset"
        self.data_name = 'avazu'
        self.field_index = 2

        # model
        self.model_name = "deepfm"
        self.l2_reg = 0.0
        self.init_std = 1e-4
        self.hidden_units = [512, 256, 128, 64]
        self.dropout = 0.0
        self.embedding_dim = 16

        # train/test
        self.seed = 1024
        self.batch_size = 1024 * 16
        self.epochs = 1
        self.optim = "adam"
        self.lr = 1e-3
        self.loss = "bce"
        self.metrics = ["auc"]
        self.verbose = 2  #  0 = silent, 1 = progress bar, 2 = one line per epoch.
        self.use_tune = True  # use ray.tune or not]

        # earlystopping
        self.monitor = "val_auc"
        self.min_delta = 1e-5
        self.patience = 5
        self.mode = "max"
        self.restore_best_weights = True

        # modelcheckpoint
        self.filepath = "/data/baiyimeng/ckpt"
        self.save_best_only = True
        self.save_weights_only = False
        self.save_freq = "epoch"
        self.is_save = True

        # history
        self.history_path = os.path.join(root, "history")


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

    setup_seed(config.seed)

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

    train_x = {name: np.array(train[name]) for name in feature_names}
    train_y = np.transpose([np.array(train[label_names])])
    valid_x = {name: np.array(valid[name]) for name in feature_names}
    valid_y = np.transpose([np.array(valid[label_names])])
    test_x = {name: np.array(test[name]) for name in feature_names}
    test_y = np.transpose([np.array(test[label_names])])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.model_name == "deepfm":
        model = DeepFM(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            use_fm=True,
            dnn_hidden_units=config.hidden_units,
            l2_reg_linear=config.l2_reg,
            l2_reg_embedding=config.l2_reg,
            l2_reg_dnn=config.l2_reg,
            init_std=config.init_std,
            dnn_dropout=config.dropout,
            dnn_activation="relu",
            dnn_use_bn=False,
            task="binary",
            device=device,
        )
    else:
        return NotImplementedError

    model.compile(
        optimizer=config.optim,
        lr=config.lr,
        loss=config.loss,
        metrics=config.metrics,
        use_tune=config.use_tune,
    )

    early_stopping = EarlyStopping(
        monitor=config.monitor,
        min_delta=config.min_delta,
        verbose=config.verbose,
        patience=config.patience,
        mode=config.mode,
        restore_best_weights=config.restore_best_weights,
    )

    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(config.filepath, experiment_name),
        monitor=config.monitor,
        verbose=config.verbose,
        save_best_only=config.save_best_only,
        save_weights_only=config.save_weights_only,
        mode=config.mode,
        save_freq=config.save_freq,
        is_save=config.is_save,
    )

    ## shuffle == False is better
    history = model.fit(
        x=train_x,
        y=train_y,
        batch_size=config.batch_size,
        epochs=config.epochs,
        verbose=config.verbose,
        initial_epoch=0,
        validation_split=0.0,
        shuffle=False,
        callbacks=[early_stopping, model_checkpoint],
        validation_data=[valid_x, valid_y],
        test_data=[test_x, test_y],
    )

    # # valid and test dataloader
    # if isinstance(valid_x, dict):
    #     valid_x = [valid_x[feature] for feature in model.feature_index]
    # for i in range(len(valid_x)):
    #     if len(valid_x[i].shape) == 1:
    #         valid_x[i] = np.expand_dims(valid_x[i], axis=1)
    # valid_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(valid_x, axis=-1)), torch.from_numpy(valid_y))
    # valid_loader = DataLoader(dataset=valid_tensor_data, shuffle=False, batch_size=1024*16, pin_memory=False)

    # if isinstance(test_x, dict):
    #     test_x = [test_x[feature] for feature in model.feature_index]
    # for i in range(len(test_x)):
    #     if len(test_x[i].shape) == 1:
    #         test_x[i] = np.expand_dims(test_x[i], axis=1)
    # test_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(test_x, axis=-1)), torch.from_numpy(test_y))
    # test_loader = DataLoader(dataset=test_tensor_data, shuffle=False, batch_size=1024*16, pin_memory=False)

    # # evaluate function
    # def evaluate(test_y_pred_calib, test_y, index):
    #     test_auc = get_auc(test_y, test_y_pred_calib)
    #     test_gauc = get_gauc(test_y, test_y_pred_calib, np.squeeze(test_x[index]))
    #     test_logloss = get_logloss(test_y, test_y_pred_calib)
    #     test_pcoc = get_pcoc(test_y, test_y_pred_calib)
    #     test_ece = get_ece(test_y, test_y_pred_calib, 100)
    #     test_fece = get_fece(test_y, test_y_pred_calib, np.squeeze(test_x[index]), 1)
    #     fece_list = get_mfece(test_y, test_y_pred_calib, test_x, 1)
    #     test_mfece = np.mean(fece_list)
    #     test_rce = get_rce(test_y, test_y_pred_calib, 100)
    #     test_frce = get_frce(test_y, test_y_pred_calib, np.squeeze(test_x[index]), 1)
    #     rce_list = get_mfrce(test_y, test_y_pred_calib, test_x, 1)
    #     test_mfrce = np.mean(rce_list)

    #     log = f"test_auc = {test_auc:.6f}, test_gauc = {test_gauc:.6f}, test_logloss = {test_logloss:.6f}, test_pcoc = {test_pcoc:.6f}, test_ece = {test_ece:.6f}, test_fece = {test_fece:.6f}, test_mfece = {test_mfece:.6f}, test_rce = {test_rce:.6f}, test_frce = {test_frce:.6f}, test_mfrce = {test_mfrce:.6f}"
    #     fece_log = f"multi-field fece list: {fece_list}"
    #     rce_log = f"multi-field rce list: {rce_list}"
    #     print(log)
    #     print(fece_log)
    #     print(rce_log)

    # test_y = []
    # test_y_pred = []
    # with torch.no_grad():
    #     for step, (x_test, y_test) in tqdm(enumerate(test_loader)):
    #         x = x_test.to(device).float()
    #         y = y_test.to(device).float()
    #         test_y.append(y.cpu().data.numpy())
    #         y_pred = model(x)
    #         test_y_pred.append(y_pred.cpu().data.numpy())
    # test_y = np.concatenate(test_y).astype("float32").flatten()
    # test_y_pred = np.concatenate(test_y_pred).astype("float32").flatten()

    # evaluate(test_y_pred, test_y, config.field_index)


if __name__ == "__main__":

    ##########################################
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
    use_tune = 1  # use ray.tune or not
    ##########################################

    if use_tune:
        config_update = {
            "lr": tune.grid_search([1e-4, 5e-4, 1e-3]),  # 1e-4, 5e-4, 1e-3
            "l2_reg": tune.grid_search([0.0, 1e-6, 1e-5, 1e-4]),  # 0., 1e-6, 1e-5, 1e-4
        }

        analysis = tune.run(
            run_or_experiment=trial,
            config=config_update,
            resources_per_trial={"cpu": 1, "gpu": 1},
            local_dir=os.path.join(root, "ray"),
            name="",
            resume="AUTO",
        )

        metric = "val_auc"

        best_trial = analysis.get_best_trial(  # best trial is reported after stopping, so 'last' is 'best'
            metric=metric,
            mode="max",
            scope="last",
        )
        print("Best config:", best_trial.config)
        print("Best result:", best_trial.last_result)

    else:
        config_update = {
            # 'data_name': 'avazu',
            # 'model_name': 'deepfm',
            # 'batch_size': 1024 * 16,
            # 'dropout': 0.,
            # 'init_std': 1e-4,
            # 'lr': 1e-3,
            # 'l2_reg': 1e-6,
            # 'seed': 1024,
            # 'field_index': 2,
            # 'use_tune': False,
            # 'is_save': True,
            # 'verbose': 1,
            # 'data_name': 'aliccp',
            # 'model_name': 'deepfm',
            # 'batch_size': 1024 * 16,
            # 'dropout': 0.,
            # 'init_std': 1e-4,
            # 'lr': 5e-4,
            # 'l2_reg': 1e-6,
            # 'seed': 1024,
            # 'field_index': 2,
            # 'use_tune': False,
            # 'is_save': True,
            # 'verbose': 1,
        }
        trial(config_update=config_update)
