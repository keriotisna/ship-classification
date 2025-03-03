import torch
import torch.nn as nn

class SeparableConv2d(torch.nn.Module):
    
    '''
    A simple separable convolution implementation where we split a traditional
    (N, N) convolution into two separate (N, 1) and (1, N) convolutions. This
    has the benefit of reducing the parameter count from N^2 to 2N at the expense of 
    additional computation.
    '''

    def __init__(
            self, 
            in_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            **kwargs
        ):
        super().__init__(**kwargs)
        
        verticalPadding = (padding, 0) if isinstance(type(padding), int) else padding
        horizontalPadding = (0, padding) if isinstance(type(padding), int) else padding
        
        self.verticalConv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=verticalPadding,
            groups=groups,
            bias=False
            )
        
        self.horizontalConv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=horizontalPadding,
            groups=groups,
            bias=False
            )
        
    def forward(self, x) -> torch.Tensor:
        out = self.verticalConv(x)
        out = self.horizontalConv(out)
        return out