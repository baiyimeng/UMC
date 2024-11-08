import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from models.inputs import (
    build_input_features,
    SparseFeat,
    DenseFeat,
    VarLenSparseFeat,
    get_varlen_pooling_list,
    create_embedding_matrix,
    varlen_embedding_lookup,
)
from models.layers.core import PredictionLayer
from models.layers.utils import slice_arrays
from models.callbacks import CallbackList, History

from ray import train


class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device="cpu"):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
            if len(feature_columns)
            else []
        )
        self.dense_feature_columns = (
            list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
            if len(feature_columns)
            else []
        )

        self.varlen_sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
            if len(feature_columns)
            else []
        )

        self.embedding_dict = create_embedding_matrix(
            feature_columns, init_std, linear=True, sparse=False, device=device
        )

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(
                torch.Tensor(
                    sum(fc.dimension for fc in self.dense_feature_columns), 1
                ).to(device)
            )
            nn.init.xavier_normal_(self.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, X, sparse_feat_refine_weight=None):

        sparse_embedding_list = [
            self.embedding_dict[feat.embedding_name](
                X[
                    :,
                    self.feature_index[feat.name][0] : self.feature_index[feat.name][1],
                ].long()
            )
            for feat in self.sparse_feature_columns
        ]

        dense_value_list = [
            X[:, self.feature_index[feat.name][0] : self.feature_index[feat.name][1]]
            for feat in self.dense_feature_columns
        ]

        sequence_embed_dict = varlen_embedding_lookup(
            X,
            self.embedding_dict,
            self.feature_index,
            self.varlen_sparse_feature_columns,
        )
        varlen_embedding_list = get_varlen_pooling_list(
            sequence_embed_dict,
            X,
            self.feature_index,
            self.varlen_sparse_feature_columns,
            self.device,
        )

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                # w_{x,i}=m_{x,i} * w_i (in IFM and DIFM)
                sparse_embedding_cat = (
                    sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
                )
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit


class BaseModel(nn.Module):
    def __init__(
        self,
        linear_feature_columns,
        dnn_feature_columns,
        l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5,
        init_std=0.0001,
        task="binary",
        device="cpu",
        gpus=None,
    ):

        super(BaseModel, self).__init__()
        self.dnn_feature_columns = dnn_feature_columns

        self.device = device
        self.reg_loss = torch.tensor(0.0, device=device)

        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError("`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns
        )

        self.embedding_dict = create_embedding_matrix(
            dnn_feature_columns, init_std, sparse=False, device=device
        )

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device
        )

        self.regularization_weight = []

        self.add_regularization_weight(
            self.embedding_dict.parameters(), l2=l2_reg_embedding
        )
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(
            task,
        )
        self.to(device)

        assert l2_reg_linear == l2_reg_embedding
        self.l2_reg = l2_reg_linear

        # parameters for callback
        self.history = History()

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose=1,
        initial_epoch=0,
        validation_split=0.0,
        shuffle=True,
        callbacks=None,
        validation_data=None,
        test_data=None,
    ):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """

        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        do_validation = False
        do_test = False

        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = (
                    validation_data  # pylint: disable=unpacking-non-sequence
                )
            else:
                raise ValueError(
                    "When passing a `validation_data` argument, "
                    "it must contain either 2 items (x_val, y_val), "
                    "or 3 items (x_val, y_val, val_sample_weights), "
                    "or alternatively it could be a dataset or a "
                    "dataset or a dataset iterator. "
                    "However we received `validation_data=%s`" % validation_data
                )
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0.0 < validation_split < 1.0:
            do_validation = True
            if hasattr(x[0], "shape"):
                split_at = int(x[0].shape[0] * (1.0 - validation_split))
            else:
                split_at = int(len(x[0]) * (1.0 - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at), slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at), slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []

        if test_data:
            do_test = True
            if len(test_data) == 2:
                test_x, test_y = test_data
                test_sample_weight = None
            elif len(test_data) == 3:
                test_x, test_y, test_sample_weight = (
                    test_data  # pylint: disable=unpacking-non-sequence
                )
            else:
                raise ValueError(
                    "When passing a `test_data` argument, "
                    "it must contain either 2 items (x_test, y_test), "
                    "or 3 items (x_test, y_test, test_sample_weights), "
                    "or alternatively it could be a dataset or a "
                    "dataset or a dataset iterator. "
                    "However we received `test_data=%s`" % test_data
                )
            if isinstance(test_x, dict):
                test_x = [test_x[feature] for feature in self.feature_index]

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)), torch.from_numpy(y)
        )
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print("parallel running on these gpus:", self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print("single running on gpu:", self.device)

        ### set shuffle, different for epochs, but the same in each epoch, no matter what the model is
        ### generator
        train_loader = DataLoader(
            dataset=train_tensor_data,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4,
            generator=torch.Generator().manual_seed(1024),
        )  # num_worker

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        if not hasattr(callbacks, "model"):
            callbacks.__setattr__("model", self)
        callbacks.model.stop_training = False

        print(
            "Train on {0} samples, validate on {1} samples, test on {2} samples, {3} steps per epoch".format(
                len(train_tensor_data), len(val_y), len(test_y), steps_per_epoch
            )
        )

        # optimization
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0

            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for step, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        optim.zero_grad()
                        y_pred = model(x)

                        loss = loss_func(input=y_pred[:, 0], target=y[:, 0])
                        reg_loss = (
                            self.get_regularization_loss() if self.l2_reg > 0 else 0.0
                        )
                        total_loss = loss + reg_loss
                        loss_epoch += loss.detach().cpu().numpy() * len(y)
                        total_loss_epoch += total_loss.detach().cpu().numpy() * len(y)

                        total_loss.backward()
                        optim.step()

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = loss_epoch / sample_num
            epoch_logs["total_loss"] = total_loss_epoch / sample_num

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            if do_test:
                eval_result = self.evaluate(test_x, test_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["test_" + name] = result

            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print("Epoch {0}/{1}".format(epoch + 1, epochs))

                eval_str = "{0}s".format(epoch_time)

                eval_str += (
                    " - " + "total_loss" + ": {0: .4f}".format(epoch_logs["total_loss"])
                )
                eval_str += " - " + "loss" + ": {0}".format(epoch_logs["loss"])

                if do_validation:
                    for name in self.metrics:
                        eval_str += (
                            " - "
                            + "val_"
                            + name
                            + ": {0: .4f}".format(epoch_logs["val_" + name])
                        )

                if do_test:
                    for name in self.metrics:
                        eval_str += (
                            " - "
                            + "test_"
                            + name
                            + ": {0: .4f}".format(epoch_logs["test_" + name])
                        )

                print(eval_str)

            # ray.tune
            if self.use_tune:
                epoch_logs_eval = {name: value for name, value in epoch_logs.items()}
                train.report(epoch_logs_eval)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        if do_validation:
            eval_result = self.evaluate(val_x, val_y, batch_size)
            for name, result in eval_result.items():
                epoch_logs["val_" + name] = result

        if do_test:
            eval_result = self.evaluate(test_x, test_y, batch_size)
            for name, result in eval_result.items():
                epoch_logs["test_" + name] = result

        # ray.tune for last=best -- val and test
        # note that train is last but not best
        if self.use_tune:
            epoch_logs_eval = {name: value for name, value in epoch_logs.items()}
            train.report(epoch_logs_eval)
        else:
            eval_str = "Best by validation:"

            if do_validation:
                for name in self.metrics:
                    eval_str += (
                        " - "
                        + "val_"
                        + name
                        + ": {0: .4f}".format(epoch_logs["val_" + name])
                    )

            if do_test:
                for name in self.metrics:
                    eval_str += (
                        " - "
                        + "test_"
                        + name
                        + ": {0: .4f}".format(epoch_logs["test_" + name])
                    )
            print(eval_str)

        return self.history

    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y[:, 0], pred_ans[:, 0])

        return eval_result

    def predict(self, x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data,
            shuffle=False,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4,
        )

        pred_ans = []

        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                y_pred = model(x).cpu().data.numpy()

                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float32")

    def input_from_feature_columns(
        self, X, feature_columns, embedding_dict, support_dense=True
    ):

        sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
            if len(feature_columns)
            else []
        )
        dense_feature_columns = (
            list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
            if len(feature_columns)
            else []
        )

        varlen_sparse_feature_columns = (
            list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
            if feature_columns
            else []
        )

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError("DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [
            embedding_dict[feat.embedding_name](
                X[
                    :,
                    self.feature_index[feat.name][0] : self.feature_index[feat.name][1],
                ].long()
            )
            for feat in sparse_feature_columns
        ]

        sequence_embed_dict = varlen_embedding_lookup(
            X, embedding_dict, self.feature_index, varlen_sparse_feature_columns
        )
        varlen_sparse_embedding_list = get_varlen_pooling_list(
            sequence_embed_dict,
            X,
            self.feature_index,
            varlen_sparse_feature_columns,
            self.device,
        )

        dense_value_list = [
            X[:, self.feature_index[feat.name][0] : self.feature_index[feat.name][1]]
            for feat in dense_feature_columns
        ]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(
        self,
        feature_columns,
        include_sparse=True,
        include_dense=True,
        feature_group=False,
    ):
        sparse_feature_columns = (
            list(
                filter(
                    lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)),
                    feature_columns,
                )
            )
            if len(feature_columns)
            else []
        )
        dense_feature_columns = (
            list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
            if len(feature_columns)
            else []
        )

        dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(
                feat.embedding_dim for feat in sparse_feature_columns
            )
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(
        self,
    ):
        # total_reg_loss = torch.tensor(0., device=self.device)
        total_reg_loss = 0.0
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def compile(
        self,
        optimizer,
        lr,
        loss=None,
        metrics=None,
        use_tune=False,
    ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer, lr)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)
        self.use_tune = use_tune

    def _get_optim(self, optimizer, lr):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters(), lr=lr)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters(), lr=lr)  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters(), lr=lr)
            else:
                raise NotImplementedError
        elif not lr:
            raise ValueError("learning rate must be provided")
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            loss_func = self._get_loss_func_single(loss)
        elif isinstance(loss, list):
            loss_func = [
                self._get_loss_func_single(loss_single) for loss_single in loss
            ]
        else:
            loss_func = loss
        return loss_func

    def _get_loss_func_single(self, loss):
        if loss == "bce":
            loss_func = F.binary_cross_entropy
        elif loss == "mse":
            loss_func = F.mse_loss
        elif loss == "mae":
            loss_func = F.l1_loss
        else:
            raise NotImplementedError
        return loss_func

    def _log_loss(
        self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None
    ):
        # change eps to improve calculation accuracy
        return log_loss(y_true, y_pred, eps, normalize, sample_weight, labels)

    @staticmethod
    def _accuracy_score(y_true, y_pred):
        return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))

    @staticmethod
    def _roc_auc_score_group(y_true, y_pred, user_id):
        data = pd.DataFrame({"user_id": user_id, "y_true": y_true, "y_pred": y_pred})
        grouped_data = data.groupby("user_id")

        group_aucs = []
        total = 0
        for user_id, group in grouped_data:
            group_y_true = group["y_true"].values
            group_y_pred = group["y_pred"].values
            if sum(group_y_true) != 0 and sum(group_y_true) != len(group):
                auc = roc_auc_score(group_y_true, group_y_pred)
                group_aucs.append(auc * len(group))  # weight
                total += len(group)

        gauc = np.sum(group_aucs) / total  # average
        return gauc

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "bce" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = self._accuracy_score
                self.metrics_names.append(metric)
        return metrics_

    def _in_multi_worker_mode(self):
        return None

    @property
    def embedding_size(
        self,
    ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = (
            list(
                filter(
                    lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)),
                    feature_columns,
                )
            )
            if len(feature_columns)
            else []
        )
        embedding_size_set = set(
            [feat.embedding_dim for feat in sparse_feature_columns]
        )
        if len(embedding_size_set) > 1:
            raise ValueError(
                "embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!"
            )
        return list(embedding_size_set)[0]
