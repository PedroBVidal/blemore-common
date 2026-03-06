import torch
import torch.nn as nn

class SequenceLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, num_layers=2, dropout=0.0):
        super(SequenceLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = lstm_out[:, -1, :]    # output of the last time step
        return out