import tqdm
import pandas as pd
import numpy as np
from tqdm import tqdm, notebook
#tqdm_notebook().pandas()
notebook.tqdm().pandas()
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn 
import pytorch_lightning as pl
from torch import optim
#from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from prettytable import PrettyTable


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset from here: https://github.com/karta282950/Multivariate-Time-Series-Data-Preprocessing-with-Pandas-in-Python
df = pd.read_csv(
    'Multivariate-Time-Series-Data-Preprocessing-with-Pandas-in-Python/Binance_BTCUSDT_minute.csv',
    parse_dates=['date'])
df = df.sort_values(by='date').reset_index(drop=True)

df['prev_close'] = df.shift(1)['close']
df['close_change'] = df.progress_apply(
    lambda row: 0 if np.isnan(row.prev_close) else row.close - row.prev_close, axis=1)


rows = []

for _, row in tqdm(df.iterrows(), total=df.shape[0], bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
    row_data = dict(
      day_of_week=row.date.day_of_week,
      day_of_month=row.date.day,
      week_of_year=row.date.week,
      month=row.date.month,
      open=row.open,
      high=row.high,
      low=row.low,
      close_change=row.close_change,
      close=row.close)
    rows.append(row_data)
features_df = pd.DataFrame(rows)

train_size = int(len(features_df)*0.9)
train_df, test_df = features_df[:train_size], features_df[train_size:]

scaler = MinMaxScaler(feature_range=(-1, 1)).set_output(transform="pandas")
scaler = scaler.fit(features_df)
train = scaler.transform(train_df)
test = scaler.transform(test_df) 


# 描述性統計，看缺失值，數據類型
table = PrettyTable()

table.field_names = ['Feature', 'Data Type','train Missing %', 'test Missing %']
for column in train.columns:
    #if column != 'Status':
    data_type = str(train[column].dtype)
    non_null_count_train = np.round(100-train[column].count()/train.shape[0]*100,1)
    non_null_count_test = np.round(100-test[column].count()/test.shape[0]*100,1)

    table.add_row([column, str(train[column].dtype), non_null_count_train, non_null_count_test])
print(table)

def create_sequences(input_data: pd.DataFrame, 
                     target_column: str, 
                     sequence_length: int):
    sequences = []
    data_size = len(input_data)
    #inputs: [0,....,sequence_length] labels: [sequence_length+1]
    for i in tqdm(range(data_size - sequence_length), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        sequence = input_data[i:i+sequence_length]

        label_position = i + sequence_length # 0,1,2...->120,121,122...
        label = input_data.iloc[label_position][target_column]

        sequences.append((sequence, label))
    return sequences

SEQUENCE_LENGTH = 120

train_sequences = create_sequences(train, 'close', SEQUENCE_LENGTH)
test_sequences = create_sequences(test, 'close', SEQUENCE_LENGTH)

print('Train Data Size:', len(train_sequences))
print('Data Shape', train_sequences[0][0].shape)
print('Training Dataset Size:',len(train_sequences),'Testing Dataset Size:', len(test_sequences))

class BTCDataset:
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]

        return torch.Tensor(sequence.to_numpy()), torch.tensor(label).float()
    
class BTCPriceDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size: int=64, num_workers: int=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.full_sequences = train_sequences
        self.test_sequences = test_sequences

        self.train_sequences = self.full_sequences[:int(len(self.full_sequences)*0.7)]
        self.validation_sequences = self.full_sequences[int(len(self.full_sequences)*0.7):]

    def setup(self, stage=None):
        self.train_dataset = BTCDataset(self.train_sequences)
        self.validation_dataset = BTCDataset(self.validation_sequences)
        self.test_dataset = BTCDataset(self.test_sequences)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
N_EPOCHS = 8
BATCH_SIZE = 64

data_module = BTCPriceDataModule(train_sequences, test_sequences, batch_size=BATCH_SIZE)
data_module.setup()

class PricePredictModel(nn.Module):
    def __init__(self, n_features: int=9, n_hidden: int=128, n_layers: int=2):
        super(PricePredictModel, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_features = n_features
        self.lstm = nn.LSTM(
            input_size = self.n_features,
            hidden_size = self.n_hidden,
            batch_first = True,
            num_layers = self.n_layers,
            dropout=0.2,
            bidirectional=False)
        self.regressor = nn.Linear(n_hidden, 1)
        
    def forward(self, x):
        #h0, c0未加lstm會自動產生
        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).to(device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).to(device)
        #out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm(x)
        out = out[:,-1,:]
        out = self.regressor(out)
        return out
    
class PricePredictModel(nn.Module):
    def __init__(self, 
                 n_features: int=9, n_hidden: int=128, n_layers: int=2, output_seq_len: int=120):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_features = n_features
        self.output_seq_len = output_seq_len
        self.net = PricePredictModel(self.n_features, self.n_hidden, self.n_layers)
    def forward(self, x):
        x = self.net(x)
        return x

class BTCPricePredictor(pl.LightningModule):
    def __init__(self, n_features: int=9, num_epochs: int=N_EPOCHS, lr: float=1e-3):
        super().__init__()
        self.num_epochs = num_epochs
        self.lr = lr
        self.model = PricePredictModel()
        self.criterion = nn.MSELoss()
    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.unsqueeze(dim=1))
        return loss, output
    
    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        loss, output = self(sequences, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        loss, output = self(sequences, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def test_step(self, batch, batch_idx):
        sequences, labels = batch
        loss, output = self(sequences, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        T_max=self.num_epochs,
                                                        eta_min=self.lr/1e2)
        return [optimizer], [scheduler]
    
model = BTCPricePredictor()
summary = pl.utilities.model_summary.ModelSummary(model, max_depth=-1)
print(summary)

pl.seed_everything(42)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath = 'checkpoints',
    filename = 'best-checkpoint',
    save_top_k = 3,
    verbose=True,
    monitor = 'val_loss',
    mode = 'min')

early_stopping_callback =  pl.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    mode='min')

logger = TensorBoardLogger(
    'lightning_logs', 
    name='btc-price')

trainer = pl.Trainer(
    logger = logger,
    callbacks=[checkpoint_callback, early_stopping_callback],
    max_epochs = N_EPOCHS,
    #gpus=1,
    accelerator="auto"
    #progress_bar_refresh_rate=30
)

trainer.fit(model, data_module)
trainer.test(model, data_module)

trained_model = BTCPricePredictor.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    n_features = train_df.shape[1])
trained_model.freeze()
test_dataset = BTCDataset(test_sequences)

predictions = []
labels = []
for sequence, label in tqdm(test_dataset, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
    _, output = trained_model(sequence.cuda().unsqueeze(dim=0))
    predictions.append(output.item())
    labels.append(label.item())

descaler = MinMaxScaler()
descaler.min_, descaler.scale_ = scaler.min_[-1], scaler.scale_[-1]
def descale(descaler, values):
    values_2d = np.array(values)[:, np.newaxis]
    return descaler.inverse_transform(values_2d).flatten()

predictions_descaled = descale(descaler, predictions)
labels_descaled = descale(descaler, labels)

test_data = df[train_size+SEQUENCE_LENGTH:]

import matplotlib.pyplot as plt
import matplotlib

dates = matplotlib.dates.date2num(test_data.date.tolist())
plt.figure(figsize = (10,6))
plt.plot_date(dates, predictions_descaled, '-', label = 'predicted')
plt.plot_date(dates, labels_descaled, '-', label = 'real')
plt.xticks(rotation = 45)
plt.title('Predicción BitCoin 2021')
plt.legend();