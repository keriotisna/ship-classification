import torch
import torch.nn as nn
from .SeparableConv2d import SeparableConv2d
from .ShuffleBlock import ShuffleBlock
from .SEBlock import SEBlock

# TODO: Implement this and maybe make it a proper subclass since I really only need 
#   to modify the forward behavior. Actually, could potentially refactor a lot of 
#   these to adhere to a proper OOP structure.

class ShuffledLinearBottleneckSE(torch.nn.Module):
    
    '''
    An implementation of the MobileNetV2 Linear Bottleneck block with grouped
    pointwise convolutions as provided by the ShuffleBlock.
    '''
    
    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            expansion_ratio: int = 4,
            stride: int = 1,
            groups: int = 1,
            activation: torch.nn.Module = nn.GELU,
            **kwargs
        ):
        super().__init__(**kwargs)
        
        # The pointwise channel expansion layer, NOT followed by an activation layer
        self.mainChannelExpansionLinearBottleneck = nn.Conv2d(
            in_channels=in_channels,
            out_channels=expansion_ratio * in_channels,
            kernel_size=1,
            groups=groups,
            bias=False
        )
        
        # Primary depthwise convolution
        self.mainConv = SeparableConv2d(
            in_channels=expansion_ratio * in_channels,
            kernel_size=kernel_size,
            padding='same',
            groups=expansion_ratio * in_channels
        )
        
        # Set the main blurred downsampling method if needed
        self.downsampleMain = None
        if stride > 1:
            self.downsampleMain = nn.AvgPool2d(
                kernel_size=3,
                stride=stride,
                padding=1
            )
        else:
            self.downsampleMain = nn.Identity()
        
        # Set residual downsampling if needed
        self.residualDownsample = None
        if stride > 1:
            self.residualDownsample = nn.AvgPool2d(
                kernel_size=3,
                stride=stride,
                padding=1
            )
        else:
            self.residualDownsample = nn.Identity()
        
        self.shuffle1 = ShuffleBlock(
            groups=groups
        )
        
        # Pointwise channel compression back to original channel count before residual connection
        self.mainChannelCompression = nn.Conv2d(
            in_channels=expansion_ratio * in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            groups=groups,
            bias=False
        )
        
        # If we have to change channel sizes, then do it with a pointwise convolution
        # otherwise just use an identity function.
        if in_channels != out_channels:
            self.mainChannelExpansionFinal = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                groups=groups
            )
            
            self.residualChannelExpansionFinal = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                groups=groups
            )
        else:
            self.mainChannelExpansionFinal = nn.Identity()
            self.residualChannelExpansionFinal = nn.Identity()


        
        # TODO: Decide whether to squeeze in uncompressed space or compressed channel space
        # Squeeze and excitement block as discount attention modules
        self.seBlock = SEBlock(
            in_channels=expansion_ratio * in_channels,
            squeeze_factor=0.25
        )
        
        # The path for the main information to travel
        self.mainPath = nn.Sequential(*[
            self.mainChannelExpansionLinearBottleneck,
            self.mainConv,
            self.seBlock,
            activation(),
            self.shuffle1,
            self.downsampleMain,
            self.mainChannelCompression,
            nn.BatchNorm2d(num_features=in_channels),
            activation(),
            self.mainChannelExpansionFinal,
            nn.BatchNorm2d(num_features=out_channels),
            activation()
        ])
        
        # The path for the residuals
        self.residualPath = nn.Sequential(*[
            self.residualDownsample,
            self.residualChannelExpansionFinal,
            nn.BatchNorm2d(num_features=out_channels),
            activation()
        ])
        
    
    def forward(self, x) -> torch.Tensor:

        out = self.mainPath(x)
        residual = self.residualPath(x)
        
        out = out + residual
        
        return out
    
