import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn 
from torch import optim
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

data = df[[c for c in df.columns if c not in ['Date','Adj Close']]].values
X_train, Y_train, X_val, Y_val, scaler = preprocess(data, train_ratio=0.8, n_past=20)

batch_size = 32
train_set = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

val_set = torch.utils.data.TensorDataset(X_val, Y_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

print(X_train.shape) # torch.Size([624, 20, 5])
print(Y_train.shape) # torch.Size([624, 5])
print(X_val.shape)
print(Y_val.shape)

class My_loss(nn.Module):
    def __init__(self, scaler):
        super().__init__()
        self.scaler = scaler
        
    def forward(self, output, target):
        np_output = output.detach().cpu().numpy()
        np_target = target.detach().cpu().numpy()
        
        np_output = np.sqrt(self.scaler.var_[3]) * np_output + self.scaler.mean_[3]
        np_target = np.sqrt(self.scaler.var_[3]) * np_target + self.scaler.mean_[3]
        
        return np.mean(np.absolute(np_output - np_target))

class Encoder(nn.Module):
    def __init__(self, input_size=5, embedding_size=36, hidden_size=128, n_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
        
    def forward(self, x):
        # x: input batch data, size: [input_seq_len, batch_size, feature_size]
        output, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_size=5, embedding_size=36, hidden_size=128, n_layers=1, dropout=0.3):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

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
        prediction = self.linear(output)

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
        #print('x->',x.shape)
        #print('y->',y.shape)

        x = x.permute(1,0,2) #我們的 dataloader是 [batch, seq, dim]
        #print('x->',x.shape)
        y = y.unsqueeze(0)#.permute(1,0,2) #但為了方便操作LSTM，直接把它擺成[seq, batch, dim]，output再把它擺回來
        #print('y->',y.shape)
        """
        x = [input_seq_len, batch_size, feature_size]
        y = [target_seq_len, batch_size, feature_size]
        """
        batch_size = x.shape[1]
        target_len = y.shape[0]
        
        # tensor to store decoder outputs of each time step
        outputs = torch.zeros(y.shape).to(self.device) 
        
        hidden, cell = self.encoder(x)
        decoder_input = x[-1, :, :] # first input to decoder is last of x
        
        for i in range(target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            # place predictions in a tensor holding predictions for each time step
            outputs[i] = torch.squeeze(output,0)
            
            teacher_forcing = random.random() < teacher_forcing_ratio
            # output is the same shape as decorder input-->[batch_size, feature_size]
            # so we use output directly as input or use true lable depending on teacher_forcing flag
            decoder_input = y[i] if teacher_forcing else torch.squeeze(output,0)
        
        return outputs.permute(1,0,2)

if __name__=='__main__':
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
    #构建损失器
    criterion = nn.MSELoss()
    
    #构建优化器
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
    def train(epoch):
        runing_loss = 0
    
        for batchix, datas in enumerate(train_loader):
            inputs, label = datas
    
            optimizer.zero_grad()
            output = model(inputs, label)
            #print('output->', output.shape, output)
            #print('label->', label.shape,label)
            loss = criterion(output.squeeze(1),label)
            loss.backward()
            optimizer.step()
    
            runing_loss +=loss.item()
            if batchix%10 == 0:
                print('[%d %3d] %.3f'%(epoch+1,batchix+1, runing_loss/300))
                runing_loss = 0
        #print(runing_loss/len(train_loader))
        return print({'outputs': output, 'labels': label})#runing_loss/len(train_loader)
    
    def evaluate(model, dataloader, criterion):
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                x = x.to(dev)
                y = y.to(dev)
                
                # turn off teacher forcing
                y_pred = model(x, y, teacher_forcing_ratio = 0)
                
                loss = criterion(y_pred, y)
                epoch_loss += loss.item()
            #return epoch_loss / len(dataloader)
    #train(1000)