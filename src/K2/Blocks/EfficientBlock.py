import torch
import torch.nn as nn
from .BlurPool import BlurPool
from .SeparableConv2d import SeparableConv2d
from .ShuffleBlock import ShuffleBlock

class EfficientBlock(torch.nn.Module):
    
    '''
    A custom implementation combining Separable Depthwise Separable convolutions
    and grouped pointwise convolutions with ShuffleBlocks. Additionally, if downsampling
    is required, we utilize the BlurPool operation to prevent aliasing from harming 
    performance.
    '''
    
    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
            groups: int,
            kernel_size: int = 3,
            stride: int = 1,
            activation: torch.nn.Module = nn.GELU,
            **kwargs
        ):
        
        super().__init__(**kwargs)
    
    pass