import torch
import torch.nn as nn
import torch.nn.functional as F

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




class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        scores = self.attention(lstm_output)
        weights = F.softmax(scores, dim=1)
        weighted_output = weights * lstm_output
        context_vector = torch.sum(weighted_output, dim=1)
        return context_vector, weights

class LSTM_Attention(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, num_layers=2, dropout=0.0):
        super(SequenceLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = Attention(hidden_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        context_vector, attn_weights = self.attention(lstm_out)
        return context_vector
