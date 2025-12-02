#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 22:50
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   rnn_lstm_classification.py
# @Desc     :   

from PySide6.QtCore import QObject, Signal
from datetime import datetime
from json import dumps
from numpy import ndarray
from sklearn import metrics
from torch import nn, optim, no_grad, save, device
from tqdm import tqdm

from src.dataloaders import TorchDataLoader
from src.utils.logger import record_log
from src.utils.PT import get_device


class TorchTrainer4Seq2Classification(QObject):
    """ Trainer class for managing training process """
    losses: Signal = Signal(int, float, float, float)

    def __init__(self, model: nn.Module, optimiser, criterion, scheduler=None, accelerator: str = "auto") -> None:
        super().__init__()
        self._accelerator = get_device(accelerator)
        self._model = model.to(device(self._accelerator))
        self._optimiser = optimiser
        self._criterion = criterion
        self._scheduler = scheduler

    def _epoch_train(self, dataloader: TorchDataLoader) -> float:
        """ Train the model for one epoch
        :param dataloader: DataLoader for training data
        :return: average training loss for the epoch
        """
        # Set model to training mode
        self._model.train()

        _loss: float = 0.0
        _total: float = 0.0
        for features, labels in dataloader:
            features, labels = features.to(device(self._accelerator)), labels.to(device(self._accelerator))

            if labels.dim() > 1:
                labels = labels.squeeze()

            self._optimiser.zero_grad()
            outputs = self._model(features)
            # print(outputs.shape, labels.shape)

            loss = self._criterion(outputs, labels)
            nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            loss.backward()
            self._optimiser.step()

            _loss += loss.item() * features.size(0)
            _total += features.size(0)

        return _loss / _total

    def _epoch_valid(self, dataloader: TorchDataLoader) -> tuple[float, dict[str, float]]:
        """ Validate the model for one epoch
        :param dataloader: DataLoader for validation data
        :return: average validation loss for the epoch
        """
        # Set model to evaluation mode
        self._model.eval()

        _loss: float = 0.0
        _correct: float = 0.0
        _total: float = 0.0

        _outs: list[ndarray] = []
        _targets: list[ndarray] = []
        with no_grad():
            for features, labels in dataloader:
                features, labels = features.to(device(self._accelerator)), labels.to(device(self._accelerator))
                outputs = self._model(features)
                # print(outputs.shape, labels.shape)

                if labels.dim() > 1:
                    labels = labels.squeeze()
                # print(outputs.shape, labels.shape)

                loss = self._criterion(outputs, labels)
                _loss += loss.item() * features.size(0)
                _total += labels.numel()

                _outs.extend(outputs.argmax(dim=1).cpu().numpy())
                _targets.extend(labels.cpu().numpy())

        _metrics: dict[str, float] = {
            "accuracy": round(float(metrics.accuracy_score(_targets, _outs)), 4),
            "precision": round(float(metrics.precision_score(_targets, _outs, average="weighted", zero_division=0)), 4),
            "recall": round(float(metrics.recall_score(_targets, _outs, average="weighted", zero_division=0)), 4),
            "f1_score": round(float(metrics.f1_score(_targets, _outs, average="weighted", zero_division=0)), 4),
        }

        cm = metrics.confusion_matrix(_targets, _outs)
        if cm.shape == (2, 2):
            # Binary classification
            TN, FP, FN, TP = cm.ravel()
            _metrics.update({"TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN)})
        else:
            # Multi-class classification
            num_classes = cm.shape[0]
            for i in range(num_classes):
                TP = int(cm[i, i])
                FP = int(cm[:, i].sum() - TP)
                FN = int(cm[i, :].sum() - TP)
                TN = int(cm.sum() - (TP + FP + FN))
                _metrics.update({
                    f"class_{i}_TP": TP,
                    f"class_{i}_FP": FP,
                    f"class_{i}_FN": FN,
                    f"class_{i}_TN": TN
                })

        return _loss / _total, _metrics

    def fit(self,
            train_loader: TorchDataLoader, valid_loader: TorchDataLoader,
            epochs: int, model_save_path: str | None = None
            ) -> None:
        """ Fit the model to the training data
        :param train_loader: DataLoader for training data
        :param valid_loader: DataLoader for validation data
        :param epochs: number of training epochs
        :param model_save_path: path to save the best model parameters
        :return: None
        """
        # Initialize logger
        timer = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logger = record_log(f"train@{timer}-THULAC")

        _best_valid_loss = float("inf")
        _patience = 5
        _patience_counter = 0
        _min_delta = 5e-4
        for epoch in range(epochs):
            train_loss = self._epoch_train(train_loader)
            valid_loss, _metrics = self._epoch_valid(valid_loader)

            # Emit training and validation progress signal
            self.losses.emit(epoch + 1, train_loss, valid_loss, _metrics["accuracy"])

            # Log epoch results
            logger.info(dumps({
                "epochs": epochs,
                "epoch": epoch + 1,
                "alpha": self._optimiser.param_groups[0]["lr"],
                "train_loss": round(float(train_loss), 4),
                "valid_loss": round(float(valid_loss), 4),
                **_metrics,
            }))

            # Save the model if it has the best validation loss so far
            if valid_loss < _best_valid_loss - _min_delta:
                _patience_counter = 0
                _best_valid_loss = valid_loss
                save(self._model.state_dict(), model_save_path)
                print(f"Model's parameters saved to {model_save_path}")
            else:
                _patience_counter += 1
                print(f"Validation loss [{_patience_counter}/{_patience}]did not improve.")
                if _patience_counter >= _patience:
                    print(f"Early stopping triggered at the {epoch} epoch and the loss is {_best_valid_loss:4f}.")
                    break

            if self._scheduler is not None:
                if isinstance(self._scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self._scheduler.step(valid_loss)
                else:
                    self._scheduler.step()

        if _patience_counter < _patience:
            print(f"Training completed after {epochs} epochs.")


if __name__ == "__main__":
    pass
