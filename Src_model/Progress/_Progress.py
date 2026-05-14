import os

import torch

import lightning as L

from torch.nn import CrossEntropyLoss, Softmax
from torch.optim import AdamW

from Utils.Metrics import *


class CrossEntropyLoss_base(L.LightningModule):
    def __init__(self, model, class_weight = None, export_result = None, out_dir = None,
                 weight_decay = 1e-2, learning_rate = 1e-4):
    
        super().__init__()
        self.model = model
        
        if class_weight is not None:
            self.loss_func = CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float, device=self.device))
        else:
            self.loss_func = CrossEntropyLoss()
        
        self.out_dir = out_dir
        self.export_result = export_result
        
        # training argument
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        
        # Inner variables
        self.train_pred_list = []
        self.train_true_list = []
        
        self.val_pred_list = []
        self.val_true_list = []
        
        self.test_pred_list = []
        self.test_true_list = []

    def training_step(self, batch, batch_idx):
        inputs, true = batch
        pred = self.model(inputs)
        loss = self.loss_func(pred ,true)
        pred = torch.argmax(pred, dim=1)
        
        pred = pred.cpu().detach().numpy()
        true = true.cpu().detach().numpy()
        
        self.log("train_step_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_step_prs", PRS(true, pred, label=1), on_epoch=True)
        self.log("train_step_rcl", RCL(true, pred, label=1), on_epoch=True)
        self.log("train_step_f1s", F1S(true, pred, label=1), on_epoch=True)
        
        self.train_pred_list.extend(pred)
        self.train_true_list.extend(true)

        return loss
    
    def on_train_epoch_end(self):
        pred = np.array(self.train_pred_list, dtype=int)
        true = np.array(self.train_true_list, dtype=int)
        
        self.log("train_epoch_prs", PRS(true, pred, label=1), prog_bar=True, on_epoch=True)
        self.log("train_epoch_rcl", RCL(true, pred, label=1), prog_bar=True, on_epoch=True)
        self.log("train_epoch_f1s", F1S(true, pred, label=1), prog_bar=True, on_epoch=True)
        
        self.train_pred_list = []
        self.train_true_list = []

    def validation_step(self, batch, batch_idx):
        inputs, true = batch
        pred = self.model(inputs)
        loss = self.loss_func(pred ,true)
        pred = torch.argmax(pred, dim=1)
        
        pred = pred.cpu().detach().numpy()
        true = true.cpu().detach().numpy()
        
        self.log("val_step_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_step_prs", PRS(true, pred, label=1), on_epoch=True)
        self.log("val_step_rcl", RCL(true, pred, label=1), on_epoch=True)
        self.log("val_step_f1s", F1S(true, pred, label=1), on_epoch=True)
        
        self.val_pred_list.extend(pred)
        self.val_true_list.extend(true)

        return loss
    
    def on_validation_epoch_end(self):
        pred = np.array(self.val_pred_list, dtype=int)
        true = np.array(self.val_true_list, dtype=int)
        
        self.log("val_epoch_prs", PRS(true, pred, label=1), prog_bar=True, on_epoch=True)
        self.log("val_epoch_rcl", RCL(true, pred, label=1), prog_bar=True, on_epoch=True)
        self.log("val_epoch_f1s", F1S(true, pred, label=1), prog_bar=True, on_epoch=True)
        
        self.val_pred_list = []
        self.val_true_list = []
    
    def test_step(self, batch, batch_idx):
        inputs, true = batch
        pred = self.model(inputs)
        pred = torch.argmax(pred, dim=1)
        
        pred = pred.cpu().detach().numpy()
        true = true.cpu().detach().numpy()

        self.test_pred_list.extend(pred)
        self.test_true_list.extend(true)

    def on_test_epoch_end(self):
        pred = np.array(self.test_pred_list, dtype=int)
        true = np.array(self.test_true_list, dtype=int)

        scoreboard = pd.DataFrame()
    
        for metric in [ACC, PRS, RCL, F1S]:
            scoreboard.loc['All', metric.__name__] = metric(true, pred)
        for metric in [PRS, RCL, F1S]:
            scoreboard.loc['All', metric.__name__] = metric(true, pred)
            for _class in np.unique(true):
                scoreboard.loc[str(_class), metric.__name__] = metric(true, pred, label=_class)
        scoreboard['Support'] = 0
        scoreboard.loc['All', 'Support'] = len(true)
        for _class in np.unique(true):
            scoreboard.loc[str(_class), 'Support'] = len(true[true == _class])
            
        if self.export_result is not None:
            scoreboard.to_excel(os.path.join(self.out_dir, f'{self.export_result}.xlsx'))
        
        else:
            print(scoreboard)
            
        self.test_pred_list = []
        self.test_true_list = []
            
    def predict_step(self, batch, batch_idx):
        inputs, _ = batch
        pred = self.model(inputs)
        # pred = torch.argmax(pred, dim=1)
        pred = Softmax(dim=1)(pred)[:, 1]
        return pred

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), 
                          lr=self.learning_rate, 
                          weight_decay=self.weight_decay)
        return optimizer
    
class CrossEntropyLoss_yolo(CrossEntropyLoss_base):
    def __init__(self, infer = 10, **kwargs):
        super().__init__(**kwargs)
        self.infer = infer
        
    def predict_step(self, batch, batch_idx):
        inputs, _ = batch
        
        preds = []
        for _ in range(self.infer):
            pred = self.model(inputs)
            pred = torch.argmax(pred, dim=1)
            preds.append(pred)
            
        pred = torch.stack(preds, dim=1)
        pred = (torch.sum(pred, dim=1) > self.infer // 2).long()
            
        return pred
    
    