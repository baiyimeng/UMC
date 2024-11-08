import torch
import numpy as np
import copy
import os


class CallbackList(object):
    def __init__(self, callbacks=None):
        self.callbacks = callbacks

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch, logs):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


class Callback(object):
    def __init__(
        self,
    ):
        self.model = None

    def set_model(self, model):
        self.model = model


class EarlyStopping(Callback):

    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    ):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ["auto", "min", "max"]:
            print("EarlyStopping mode %s is unknown, " "fallback to auto mode.", mode)
            mode = "auto"  ### 这一行之前没缩进导致bug

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if "acc" in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, logs):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            # self.best_weights = self.model.get_weights()
            self.best_weights = copy.deepcopy(
                self.model.state_dict()
            )  #  去掉copy会报错

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            if self.restore_best_weights:
                # self.best_weights = self.model.get_weights()
                self.best_weights = copy.deepcopy(
                    self.model.state_dict()
                )  # 去掉copy会报错
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    print("Restoring model weights from the end of the best epoch.")
                # self.model.set_weights(self.best_weights)
                self.model.load_state_dict(self.best_weights)

    def on_train_end(self):
        pass

    def get_monitor_value(self, logs):
        logs = logs or {}
        if isinstance(self.monitor, list):
            # 如果self.monitor是列表，则计算多个指标的平均值
            monitor_values = [logs.get(metric) for metric in self.monitor]
            monitor_value = sum(monitor_values) / len(monitor_values)
        else:
            # 如果self.monitor是字符串，则获取单个指标的值
            monitor_value = logs.get(self.monitor)

        if monitor_value is None:
            print(
                "Early stopping metric is not available. Available metrics are: %s",
                ",".join(list(logs.keys())),
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)


class ModelCheckpoint(Callback):
    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        is_save=False,
    ):
        super(ModelCheckpoint, self).__init__()

        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        self.is_save = is_save
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.epochs_since_last_save = 0
        self.period = 1

        if mode not in ["auto", "min", "max"]:
            print("ModelCheckpoint mode %s is unknown, " "fallback to auto mode.", mode)
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        if self.save_freq != "epoch":
            raise ValueError("Only support save_freq=epoch")

    def on_train_begin(self):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            # filepath = self.filepath.format(epoch=epoch + 1, **logs)
            filepath = self.filepath + ".pth"
            if self.save_best_only:
                if isinstance(self.monitor, list):
                    # 如果self.monitor是列表，则计算多个指标的平均值
                    monitor_values = [logs.get(metric) for metric in self.monitor]
                    current = sum(monitor_values) / len(monitor_values)
                else:
                    # 如果self.monitor是字符串，则获取单个指标的值
                    current = logs.get(self.monitor)
                if current is None:
                    print(
                        "Can save best model only with %s available, skipping."
                        % self.monitor
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                "Epoch %05d: %s improved from %0.5f to %0.5f,"
                                " saving model to %s"
                                % (
                                    epoch + 1,
                                    self.monitor,
                                    self.best,
                                    current,
                                    filepath,
                                )
                            )
                        self.best = current
                        if self.is_save:
                            if self.save_weights_only:
                                torch.save(self.model.state_dict(), filepath)
                            else:
                                torch.save(self.model, filepath)
                    else:
                        if self.verbose > 0:
                            print(
                                "Epoch %05d: %s did not improve from %0.5f"
                                % (epoch + 1, self.monitor, self.best)
                            )
            else:
                if self.verbose > 0:
                    print("Epoch %05d: saving model to %s" % (epoch + 1, filepath))
                if self.is_save:
                    if self.save_weights_only:
                        torch.save(self.model.state_dict(), filepath)
                    else:
                        torch.save(self.model, filepath)

    def on_train_end(self):
        pass


class History(Callback):
    def __init__(self):
        super(History, self).__init__()
        self.history = {}

    def on_train_begin(self):
        self.epoch = []

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # Set the history attribute on the model after the epoch ends. This will
        # make sure that the state which is set is the latest one.
        self.model.history = self

    def on_train_end(self):
        pass
