import torch
import torch.nn as nn
from .SeparableConv2d import SeparableConv2d
from .ShuffleBlock import ShuffleBlock

# TODO: Maybe see if I can get BlurPool to work?
#   I am currently just using average pooling because that's consistent
#   and close enough, but the reflected padding may help avoid information
#   loss around the edges of the image.
# TODO: Try comparing double-blurred downsampling vs a blur/maxPool.
#   I could blur the main convolution and use max pooling to match the residuals,
#   or I could just blur both. Not sure which is better but I should check



class ShuffledLinearBottleneck(torch.nn.Module):
    
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


        
        # The path for the main information to travel
        self.mainPath = nn.Sequential(*[
            self.mainChannelExpansionLinearBottleneck,
            self.mainConv,
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
    
