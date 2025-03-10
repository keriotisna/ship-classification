import torch
import torch.nn as nn

class ResidualBlock(torch.nn.Module):
    
    '''
    An implementation of the residual block as introduced in ResNet.
    This might noy be an exact replica, but it's a very close match.
    
    Paper: https://arxiv.org/abs/1512.03385
    '''
    
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 3,
            groups: int = 1,
            activation: torch.nn.Module = nn.ReLU,
            **kwargs):
        super().__init__(**kwargs)
        
        self.layer1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=kernel_size, 
            padding='same',
            groups=groups
        )
        
        self.layer2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding='same',
            groups=groups
        )
        
        self.norm1 = nn.BatchNorm2d(num_features=in_channels)
        self.norm2 = nn.BatchNorm2d(num_features=in_channels)

        self.activation1 = activation()
        self.activation2 = activation()
        self.activation3 = activation()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.layer1(x)
        out = self.activation1(out)
        out = self.norm1(out)
        
        out = self.layer2(out)
        out = self.activation2(out)
        out = self.norm2(out)
        
        out = out + x
        
        return self.activation3(out)