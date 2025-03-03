import torch
import torch.nn as nn

class ResidualBlockVersatile(torch.nn.Module):
    
    '''
    An attempt at making a residual block capable of handling
    more advanced CNN parameters like stride or channel compression/decompression.
    
    The goal is to allow for learned upsampling or downsampling while still maintaining the
    residual connections between layers.
    '''
    
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        activation: torch.nn.Module = nn.ReLU(),
        **kwargs
    ):
        super().__init__(**kwargs)
        
        
        self.layer1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        self.layer2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding='same',
            stride=1,
            bias=False
        )
        
        self.downsample = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.channelSwitchLayer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=groups,
            bias=False
        )
        
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)

        self.activation = activation
        
        
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.layer1(x)
        out = self.activation(out)
        out = self.norm1(out)

        out = self.layer2(out)
        out = self.activation(out)
        out = self.norm2(out)

        residual = self.downsample(x)
        residual = self.channelSwitchLayer(residual)

        out = out + residual

        return self.activation(out) 