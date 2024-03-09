import tqdm
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn 
from torch import optim
from torchsummary import summary
import random

'''
[v1]
* 可以跑通，但預測值很怪

[Ref]
* https://peaceful0907.medium.com/time-series-prediction-lstm%E7%9A%84%E5%90%84%E7%A8%AE%E7%94%A8%E6%B3%95-ed36f0370204
* https://curow.github.io/blog/LSTM-Encoder-Decoder/
'''

#https://hk.finance.yahoo.com/quote/2330.TW/history?period1=1548374400&period2=1706140800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
df = pd.read_csv('data/2330.TW.csv')
#print(df.shape)
#print(df.head())
def preprocess(data_trend, train_ratio, n_past):
    
    scaler = StandardScaler() 
    data_trend = scaler.fit_transform(data_trend)
    
    train_ind = int(len(data_trend)*train_ratio)
    train_data = data_trend[:train_ind]
    val_data = data_trend[train_ind:]

    # convert our train data into a pytorch tensor
    X_train, Y_train = create_sequences(train_data, n_past)
    X_val, Y_val = create_sequences(val_data, n_past)

    return X_train, Y_train, X_val, Y_val, scaler


def create_sequences(data, n_past):
    X,Y = [],[]
    L = len(data)
    for i in range(L-(n_past+5)):
        X.append(data[i:i+n_past])
        Y.append(data[i+n_past:i+n_past+5][:,3])
    
    return torch.Tensor(np.array(X)), torch.Tensor(np.array(Y))

class WrappedDataLoader:
    def __init__(self, dataloader, func):
        self.dataloader = dataloader
        self.func = func
        
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        iter_dataloader = iter(self.dataloader)
        for batch in iter_dataloader:
            yield self.func(*batch)
            
def transpose(x, y):
    # x and y is [batch size, seq len, feature size]
    # to make them work with default assumption of LSTM,
    # here we transpose the first and second dimension
    # return size = [seq len, batch size, feature size]
    return x.transpose(0, 1), y.transpose(0, 1)

data = df[[c for c in df.columns if c not in ['Date','Adj Close']]].values
X_train, Y_train, X_val, Y_val, scaler = preprocess(data, train_ratio=0.8, n_past=20)

batch_size = 32
train_set = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
train_loader = WrappedDataLoader(train_loader, transpose)

#for inputs, labbels in train_loader():
#    print(inputs.shape)
#    print(labbels.shape)

val_set = torch.utils.data.TensorDataset(X_val, Y_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
val_loader = WrappedDataLoader(val_loader, transpose)

print('='*10,'Dataset','='*10)
print('X_train->', X_train.shape) # torch.Size([624, 20, 5])
print('Y_train->',Y_train.shape) # torch.Size([624, 5])
print('X_val->  ',X_val.shape)
print('Y_val->  ',Y_val.shape)

class Encoder(nn.Module):
    def __init__(self, input_size=5, embedding_size=36, hidden_size=128, n_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
        
    def forward(self, x):
        # x: input batch data, size: [input_seq_len, batch_size, feature_size]
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_size=5, embedding_size=36, hidden_size=128, n_layers=1, dropout=0.3):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(output_size, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        """
        x size = [batch_size, feature_size]
        --> x only has two dimensions since the input is batch of last coordinate of observed trajectory
        so the sequence length has been removed.
        """
        # add sequence dimension to x, to allow use of nn.LSTM
        x = x.unsqueeze(0)  # -->[1, batch_size, feature_size]
        
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.linear(output.squeeze(0))

        return prediction, hidden, cell    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, x, y, teacher_forcing_ratio=0.5):
        #print(y)
        #print('---------Seq2Seq---------')
        #print('x->',x.shape)
        #print('y->',y.shape)
        #print(x.shape)
        #print(x)
        
        #x = x.permute(1,0,2) #我們的 dataloader是 [batch, seq, dim]
        #print('x->',x.shape)
        #y = y.unsqueeze(0)#.permute(1,0,2) #但為了方便操作LSTM，直接把它擺成[seq, batch, dim]，output再把它擺回來
        #print('y->',y.shape)
        """
        x = [input_seq_len, batch_size, feature_size]
        y = [target_seq_len, batch_size, feature_size]
        """
        batch_size = x.shape[1]
        y = y.transpose(0, 1).unsqueeze(0)
        target_len = y.shape[0]
        #print('x->', x.shape)
        #print('y->', y.shape)
        # tensor to store decoder outputs of each time step
        #outputs = torch.zeros(y.shape).to(self.device) 
        outputs = torch.zeros(target_len, batch_size, self.decoder.output_size).to(self.device) 
        hidden, cell = self.encoder(x)
        decoder_input = x[-1, :, :] # first input to decoder is last of x
        
        for i in range(target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            # place predictions in a tensor holding predictions for each time step
            #print(output)
            #print(output.shape)
            #print(outputs.shape)

            outputs[i] = output #torch.squeeze(output,0)
            
            teacher_forcing = random.random() < teacher_forcing_ratio
            # output is the same shape as decorder input-->[batch_size, feature_size]
            # so we use output directly as input or use true lable depending on teacher_forcing flag
            decoder_input = y[i] if teacher_forcing else output #torch.squeeze(output,0)
        
        return outputs #.permute(1,0,2)

if __name__=='__main__':
    '''
    INPUT_DIM = 5
    OUTPUT_DIM = 5
    ENC_EMB_DIM = 36
    DEC_EMB_DIM = 36
    HID_DIM = 128
    N_LAYERS = 3
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(enc, dec, dev).to(dev)
    #                   [seq_len, batch_size, feature_size]
    inputs = torch.zeros((20, 32, 5))
    labels = torch.zeros((1, 32, 5))
    outputs = model(inputs, labels)
    criterion = nn.MSELoss()
    loss = criterion(outputs, labels)
    print(loss)
    '''
    INPUT_DIM = 5
    OUTPUT_DIM = 5
    ENC_EMB_DIM = 36
    DEC_EMB_DIM = 36
    HID_DIM = 128
    N_LAYERS = 3
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Seq2Seq(enc, dec, dev).to(dev)
    #summary(model, input_size=[(20, 1, 5), (1, 1, 5)])
    #构建损失器
    criterion = nn.MSELoss()
    #构建优化器
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
    def train(model, dataloader):
        model.train()
        print('='*10,'Training','='*10)
        epoch_loss = 0
        for i, datas in enumerate(dataloader):
            inputs, label = datas
            #print('inputs->', inputs.shape)
            #print('label->', label.shape)

            optimizer.zero_grad()
            output = model(inputs, label)
            output = output.squeeze(0)
            #print(output)
            #print(output.shape)
            #print(label.shape)
            #print('output->', output.shape, output)
            #print('label->', label.shape,label)
            label = label.permute(1,0)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
        return epoch_loss / len(dataloader)
            
    def evaluate(model, dataloader):
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, datas in enumerate(dataloader):
                inputs, label = datas
                
                # turn off teacher forcing
                y_pred = model(inputs, label, teacher_forcing_ratio = 0)
                inputs = label.permute(1,0)
                loss = criterion(y_pred, inputs)
                epoch_loss += loss.item()
        return epoch_loss / len(dataloader)
    
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    N_EPOCHES = 100
    best_val_loss = float('inf')
    model_dir = "saved_models/Seq2Seq"
    saved_model_path = model_dir + "/best_seq2seq.pt"
    if os.path.isfile(saved_model_path):
        model.load_state_dict(torch.load(saved_model_path))
        print("successfully load previous best model parameters")
        
    for epoch in range(N_EPOCHES):
        start_time = time.time()
        
        train_loss = train(model, train_loader)
        val_loss = evaluate(model, val_loader)
        
        end_time = time.time()
        
        mins, secs = epoch_time(start_time, end_time)
        
        print(F'Epoch: {epoch+1:02} | Time: {mins}m {secs}s')
        print(F'\tTrain Loss: {train_loss:.3f}')
        print(F'\t Val. Loss: {val_loss:.3f}')

        if val_loss < best_val_loss:
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), saved_model_path)
