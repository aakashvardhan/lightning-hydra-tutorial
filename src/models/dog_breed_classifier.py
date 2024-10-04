import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchmetrics import Accuracy, MeanMetric, MaxMetric


class DogBreedClassifier(L.LightningModule):
    def __init__(self, num_classes: int = 10, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Use a more advanced model
        self.model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=self.num_classes)
        
        # Add more dropout for regularization
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # for logging best model
        self.val_acc_best = MaxMetric()
        
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
    
    def on_train_start(self):
        self.val_acc_best.reset()
        
    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        self.val_acc_best(val_acc)
        self.log("val_acc_best", self.val_acc_best.compute(), prog_bar=True)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        loss = self.criterion(logits, y)
        self.train_loss(loss)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        loss = self.criterion(logits, y)
        self.val_loss(loss)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", self.val_acc, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        loss = self.criterion(logits, y)
        self.test_loss(loss)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", self.test_acc, prog_bar=True, logger=True)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Use a fixed number of steps per epoch
        steps_per_epoch = 100  # You can adjust this based on your dataset size
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=self.trainer.max_epochs,
            pct_start=0.3,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }