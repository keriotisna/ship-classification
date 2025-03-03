import torch
import torch.nn as nn
from .SeparableConv2d import SeparableConv2d

class DepthwiseSeparableConv2d(torch.nn.Module):
    
    '''
    A (hopefully) more efficient convolutional block which breaks down operations into
    depthwise-separable (DWS) convolutions. Additionally, we perform separable convolutions
    where we break a convolution into multiple identical sized operations across the 
    different axes.
    
    For a better explanation of DWS convolutions, see this:
    https://www.youtube.com/watch?v=vVaRhZXovbw
    '''
    
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            activation: torch.nn.Module = nn.ReLU(),
            **kwargs
        ):
        
        super().__init__(**kwargs)

        # Downsamples resolution
        self.downsample = nn.Sequential(*[
            nn.AvgPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            activation
        ])

        # Expand channels to output size
        self.channelExpansion1 = nn.Sequential(*[
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            activation
        ])

        self.residualExpansion = nn.Sequential(*[
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            activation
        ])

        self.separableConv1 = SeparableConv2d(
            in_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=out_channels
        )
        
        self.pointwise1 = nn.Sequential(*[
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            activation
        ])
        
        self.separableConv2 = SeparableConv2d(
            in_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            groups=out_channels
        )
        
        self.pointwise2 = nn.Sequential(*[
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            activation
        ])
        
        self.activation = activation
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.channelExpansion1(x)

        out = self.separableConv1(out)
        out = self.pointwise1(out)

        out = self.separableConv2(out)
        out = self.pointwise2(out)

        
        residual = self.downsample(x)
        residual = self.residualExpansion(residual)

        out = out + residual
        
        return out