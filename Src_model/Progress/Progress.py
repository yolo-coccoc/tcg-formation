import os

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score as ACC, f1_score as F1S, precision_score as PRS, recall_score as RCL
from torch.nn import CrossEntropyLoss, Softmax
from torch.optim import AdamW


class ClassificationModule(L.LightningModule):
    """Lightning module for classification with cross-entropy loss and metrics logging."""

    def __init__(self,
                 model,
                 class_weight=None,
                 export_result=None,
                 out_dir=None,
                 optimizer_kwargs = {"lr": 1e-4, "weight_decay": 1e-2},):
        
        super().__init__()
        self.model = model
        if class_weight is not None:
            self.loss_func = CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float))
        else:
            self.loss_func = CrossEntropyLoss()

        self.out_dir = out_dir
        self.export_result = export_result
        self.optimizer_kwargs = optimizer_kwargs

        # Initialize prediction and true label lists for each phase
        self.pred_lists = {"train": [], "val": [], "test": []}
        self.true_lists = {"train": [], "val": [], "test": []}

    def _compute_metrics(self, true: np.ndarray, pred: np.ndarray) -> dict:
        """Compute precision, recall, and F1 score."""
        return {
            "prs": PRS(true, pred, average="macro"),
            "rcl": RCL(true, pred, average="macro"),
            "f1s": F1S(true, pred, average="macro"),
        }

    def _log_step_metrics(self, loss: torch.Tensor, true: np.ndarray, pred: np.ndarray, prefix: str):
        """Log metrics for a single step."""
        metrics = self._compute_metrics(true, pred)
        self.log(f"{prefix}_step_loss", loss, prog_bar=True, on_epoch=True)
        for key, value in metrics.items():
            self.log(f"{prefix}_step_{key}", value, on_epoch=True)

    def _log_epoch_metrics(self, phase: str):
        """Log metrics for an entire epoch."""
        pred = np.array(self.pred_lists[phase], dtype=int)
        true = np.array(self.true_lists[phase], dtype=int)
        metrics = self._compute_metrics(true, pred)

        for key, value in metrics.items():
            self.log(f"{phase}_epoch_{key}", value, prog_bar=True, on_epoch=True)

        # Reset lists
        self.pred_lists[phase] = []
        self.true_lists[phase] = []

    def training_step(self, batch, batch_idx):
        inputs, true = batch
        pred_logits = self.model(inputs)
        loss = self.loss_func(pred_logits, true)
        pred = torch.argmax(pred_logits, dim=1).cpu().detach().numpy()
        true = true.cpu().detach().numpy()

        self._log_step_metrics(loss, true, pred, "train")
        self.pred_lists["train"].extend(pred)
        self.true_lists["train"].extend(true)

        return loss

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def validation_step(self, batch, batch_idx):
        inputs, true = batch
        pred_logits = self.model(inputs)
        loss = self.loss_func(pred_logits, true)
        pred = torch.argmax(pred_logits, dim=1).cpu().detach().numpy()
        true = true.cpu().detach().numpy()

        self._log_step_metrics(loss, true, pred, "val")
        self.pred_lists["val"].extend(pred)
        self.true_lists["val"].extend(true)

        return loss

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def test_step(self, batch, batch_idx):
        inputs, true = batch
        pred_logits = self.model(inputs)
        pred = torch.argmax(pred_logits, dim=1).cpu().detach().numpy()
        true = true.cpu().detach().numpy()

        self.pred_lists["test"].extend(pred)
        self.true_lists["test"].extend(true)

    def on_test_epoch_end(self):
        pred = np.array(self.pred_lists["test"], dtype=int)
        true = np.array(self.true_lists["test"], dtype=int)

        scoreboard = pd.DataFrame()
        for metric in [ACC, PRS, RCL, F1S]:
            scoreboard.loc["1", metric.__name__] = metric(true, pred)
        scoreboard["Support"] = 0
        scoreboard.loc["1", "Support"] = (true == 1).sum()

        if self.export_result:
            scoreboard.to_excel(os.path.join(self.out_dir, f"{self.export_result}.xlsx"))
        else:
            print(scoreboard)

        self.pred_lists["test"] = []
        self.true_lists["test"] = []

    def predict_step(self, batch, batch_idx):
        inputs, _ = batch
        pred_logits = self.model(inputs)
        return Softmax(dim=1)(pred_logits)[:, 1]

    def configure_optimizers(self):
        return AdamW(self.parameters(), **self.optimizer_kwargs)