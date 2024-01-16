import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

dictionary = ['e', 'h', 'l', 'o']

x_data = [[1,0,2,2,3]] #hello
y_data = [3,1,2,3,2] #ohlol

inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)

num_class = len(dictionary)
inputs_size = num_class
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len = len(x_data) # 數據量的意思

print(inputs.size())
print(labels.size())

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.emb = nn.Embedding(inputs_size, embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)
    
    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)
        x, _ = self.rnn(x, hidden)

        x = self.fc(x)
        return x.view(-1, num_class)
    
model = MyModule()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(15):
    y_pred = model(inputs)
    loss = criterion(y_pred, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, index = y_pred.max(dim=1)
    index = index.data.numpy()
    print("Predicted:", "".join([dictionary[x] for x in index]), end="")
    print(", Epoch [%d/15] loss = %.3f" % (epoch + 1, loss.item()))