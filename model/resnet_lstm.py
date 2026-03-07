import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetLSTM(nn.Module):
    def __init__(self, hidden_size=512, num_layers=2):
        super(ResNetLSTM, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        self.lstm = nn.LSTM(input_size=2048, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)


    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        
        x = x.view(batch_size * time_steps, C, H, W)
        features = self.feature_extractor(x) # (B*T, 2048, 1, 1)
        features = features.view(batch_size, time_steps, -1) # (B, T, 2048)
        
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        out = lstm_out[:, -1, :]
        return out