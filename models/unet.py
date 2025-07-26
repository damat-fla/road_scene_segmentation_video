import torch
from torch import nn


# To define a model in pytorch, we need to create a class that inherits 
# from nn.Module. The class must implement the __init__ method to define 
# the layers and the forward method to define the forward pass.

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.25)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.norm1(self.conv1(x))) # conv -> batch_norm -> relu
        x1 = self.dropout(x1)
        x2 = self.relu(self.norm2(self.conv2(x1))) # conv -> batch_norm -> relu
        x2 = self.dropout(x2) 
        return x2

class Down_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = Conv_Block(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.down(x)
        p = self.pool(down)
        return down, p
    
class Up_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = Conv_Block(in_channels, out_channels)
    
    def forward(self, x1, x2):
        u = self.up(x1)
        conv = self.conv(torch.cat([u, x2], dim=1)) # skip connections also
        return conv

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=16):
        super().__init__()
        self.down1 = Down_Block(in_channels, base_channels)
        self.down2 = Down_Block(base_channels, base_channels*2)
        self.down3 = Down_Block(base_channels*2, base_channels*4)

        self.bottleneck = Conv_Block(base_channels*4, base_channels*8)

        self.up1 = Up_Block(base_channels*8, base_channels*4)
        self.up2 = Up_Block(base_channels*4, base_channels*2)
        self.up3 = Up_Block(base_channels*2, base_channels)

        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1, p1 = self.down1(x)
        x2, p2 = self.down2(p1)
        x3, p3 = self.down3(p2)

        bottleneck = self.bottleneck(p3)

        up1 = self.up1(bottleneck, x3)
        up2 = self.up2(up1, x2)
        up3 = self.up3(up2, x1)

        out = self.out(up3)

        return out
