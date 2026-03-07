# Based on https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/iresnet.py

import torch
from torch import nn

def conv1d_3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv1d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)

def conv1d_1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class IBasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock1D, self).__init__()
        self.bn1 = nn.BatchNorm1d(inplanes, eps=1e-05)
        self.conv1 = conv1d_3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm1d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv1d_3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm1d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class IResNet1D(nn.Module):
    def __init__(self, block, layers, input_channels=1, dropout=0, num_features=512):
        super(IResNet1D, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.bn2 = nn.BatchNorm1d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1) 
        
        self.fc = nn.Linear(512 * block.expansion, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1d_1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x

def iresnet18_1d(**kwargs):
    return IResNet1D(IBasicBlock1D, [2, 2, 2, 2], **kwargs)

def iresnet50_1d(**kwargs):
    return IResNet1D(IBasicBlock1D, [3, 4, 14, 3], **kwargs)



if __name__ == "__main__":
    model = iresnet18_1d(input_channels=1, num_features=128)
    random_input = torch.randn(8, 1, 709)
    output = model(random_input)
    print(output.shape)