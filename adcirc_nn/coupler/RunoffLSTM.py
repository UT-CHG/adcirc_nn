import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RunoffLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0,
            activation="ReLU"):
        super(RunoffLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                           num_layers = num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        if activation == "ReLU":
            self.act = nn.ReLU()
        elif activation == "LeakyReLu":
            self.act = nn.LeakyReLU()
        elif activation == "ELU":
            self.act = nn.ELU()
        else:
            self.act = None

        h0 = torch.empty(num_layers, 1, hidden_size).to(device)
        c0 = torch.empty(num_layers, 1, hidden_size).to(device)
        nn.init.kaiming_normal_(h0)
        nn.init.kaiming_normal_(c0)
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)
        
    def forward(self, x, hidden):
        pred, hidden = self.lstm(x, hidden)
        pred = self.linear(pred).view(pred.data.shape[0], -1, 1)
        if self.act != None:
            pred = self.act(pred)
        return pred, hidden