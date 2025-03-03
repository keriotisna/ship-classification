import torch
import torch.nn as nn
from .SeparableConv2d import SeparableConv2d

class LinearBottleneck(torch.nn.Module):
    
    '''
    An implementation of the MobileNetV2 Linear Bottleneck block.
    
    This combines the Inverse Residual Connection with linear bottlenecking to ensure
    that non-linear activation functions don't destroy too much information.
    
    https://arxiv.org/pdf/1801.04381
    '''
    
    def __init__(
            self, 
            in_channels: int,
            kernel_size: int = 3,
            expansion_ratio: int = 4,
            stride: int = 1,
            activation: torch.nn.Module = nn.ReLU6(),
            **kwargs
        ):
        super().__init__(**kwargs)
        
        # The pointwise channel expansion layer, NOT followed by an activation layer
        self.pointwiseExpansion = nn.Conv2d(
            in_channels=in_channels,
            out_channels=expansion_ratio * in_channels,
            kernel_size=1,
            stride=stride,
            bias=False
        )
        
        # Primary depthwise convolution
        self.mainConv = SeparableConv2d(
            in_channels=expansion_ratio * in_channels,
            kernel_size=kernel_size,
            padding='same',
            groups=expansion_ratio * in_channels
        )
        
        # Pointwise channel compression back to original channel count before residual connection
        self.pointwiseCompression = nn.Conv2d(
            in_channels=expansion_ratio * in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )
        
        self.activation = activation
        self.norm = nn.BatchNorm2d(num_features=in_channels)
        
    
    def forward(self, x) -> torch.Tensor:
        
        # Don't apply the activation after the first expansion
        out = self.pointwiseExpansion(x)
        
        # Perform initial convolution & activation
        out = self.mainConv(out)
        out = self.activation(out)
        
        # Compress manifold back to lower dimensions while retaining more information
        out = self.pointwiseCompression(out)
        out = self.norm(out)
        out = self.activation(out)
        
        return out
    