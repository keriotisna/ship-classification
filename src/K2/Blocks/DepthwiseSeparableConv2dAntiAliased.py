import torch
import torch.nn as nn
from .SeparableConv2d import SeparableConv2d
from .BlurPool import BlurPool

# TODO: See if I can make this actually shape agnostic, but it might be tricky.
#   The problem is that in BlurPool, we pad to ensure the post-blurred activations are
#   the same shape as the input which is correct, but when downsampling, we don't align
#   with shapes like conv2d(kernel_size=3, stride=2, padding=0) which makes combining
#   these techniques much trickier.

class DepthwiseSeparableConv2dAntiAliased(torch.nn.Module):
    
    '''
    An extension of the Separable Depthwise-Separable convolutions with additional anti-aliasing
    by using the BlurPool layer. The original version of this was actually using average pooling
    to downsample which was extremely similar to the BlurPool method, resulting in little to no
    '''
    
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            activation: torch.nn.Module = nn.ReLU,
            **kwargs
        ):
        
        super().__init__(**kwargs)

        
        # Downsamples input resolution with anti-aliasing filter
        self.downsampleInputResolution = BlurPool(
            out_channels=out_channels,
            stride=stride
        )

        
        # Downsample the residual connection resolution separately
        self.downsampleResidualResolution = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=1 # <-- This hard-coded padding really makes me sick, but it's the only way I can get the shapes to align
        )
        
        # Expand channels to output size for the main SDWSC path
        self.inputChannelExpansion = nn.Sequential(*[
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            activation()
        ])

        # Expand input channel count to match output channel count for the residual path
        self.residualChannelExpansion = nn.Sequential(*[
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            activation()
        ])


        # First Depthwise + Pointwise convolutions
        self.separableConv1 = SeparableConv2d(
            in_channels=out_channels,
            kernel_size=kernel_size,
            padding='same',
            stride=1,
            groups=out_channels
        )
        self.pointwise1 = nn.Sequential(*[
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            activation()
        ])
        
        
        # Second Depthwise + Pointwise convolutions
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
            activation()
        ])
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.inputChannelExpansion(x)

        # SDWSC 1
        out = self.separableConv1(out)
        out = self.pointwise1(out)
        out = self.downsampleInputResolution(out) # Downsample input path

        # SDWSC 2
        out = self.separableConv2(out)
        out = self.pointwise2(out)

        residual = self.downsampleResidualResolution(x)
        residual = self.residualChannelExpansion(residual)

        out = residual + out
        return out


