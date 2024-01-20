from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch 
from torch import nn 
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms as T 
from torchmetrics import Accuracy
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger

from einops import rearrange
from models.networks import LinearModel
import hydra
from omegaconf import DictConfig
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MNISTDataMudle(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_dir = self.cfg.data_dir
        self.batch_size = self.cfg.batch_size
        self.num_workers = self.cfg.num_workers

        #self.prepare_data_per_node = True
        
    def setup(self, stage=None) -> None:
        compose = T.Compose([
            T.ToTensor(),
            T.Normalize(0.1307, 0.3081)
        ])
        self.full_df = MNIST(self.data_dir, train=True, transform=compose, download=False)
        self.train_df, self.val_df = random_split(self.full_df, [0.7,0.3])
        self.test_df = MNIST(self.data_dir, train=True, transform=compose, download=False)
    
    def train_dataloader(self):
        return DataLoader(self.train_df, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_df, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
        
    def test_dataloader(self):
        return DataLoader(self.test_df, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
        
class Model(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(Model, self).__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.num_workers = self.cfg.num_workers
        self.hidden_dim = self.cfg.hidden_dim
        self.learning_rate = self.cfg.learning_rate
        self.num_epochs = self.cfg.num_epochs
        
        self.acc = Accuracy(task='multiclass', num_classes=10)
        self.net = LinearModel(self.cfg.hidden_dim)
        self.validation_step_outputs = []
        
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = self.acc(preds, y)
        self.log('train_loss', torch.tensor([loss]))
        self.log('train_acc', torch.tensor([acc]))
        return {"loss":loss,"preds":preds.detach(),"y":y.detach()}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = self.acc(preds, y)
        self.log('val_loss', torch.tensor([loss]))
        self.log('val_acc', torch.tensor([acc]))
        return {"loss":loss, "preds":preds.detach(), "y":y.detach()}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = self.acc(preds, y)
        self.log('test_loss', torch.tensor([loss]))
        self.log('test_acc', torch.tensor([acc]))
        return {"loss":loss,"preds":preds.detach(),"y":y.detach()}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        T_max=self.num_epochs,
                                                        eta_min=self.learning_rate/1e2)
        return [optimizer], [scheduler]
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}


@hydra.main(config_path="config", config_name="pl_mnist_nn_ai", version_base="1.1")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    
    data_mnist = MNISTDataMudle(cfg)
    data_mnist.setup()
    
    ckpt_callback = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')
    pbar = pl.callbacks.TQDMProgressBar(refresh_rate=1)
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
    #csv_logger = CSVLogger('./', name='linear', version='0'),
    
    pl_logger = WandbLogger(
        name=cfg.exp_name,
        project="pl_mnist")
    
    model = Model(cfg)
    trainer = pl.Trainer(max_epochs=cfg.num_epochs,
                      callbacks=[ckpt_callback, pbar, early_stopping],
                      #logger=csv_logger,
                      logger=pl_logger,
                      enable_model_summary=True,
                      accelerator='auto',
                      devices=1,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      fast_dev_run=True)
    
    trainer.fit(model, data_mnist)
    '''model = model.load_from_checkpoint(ckpt_callback.best_model_path, cfg=cfg)'''

if __name__== '__main__':
    main()