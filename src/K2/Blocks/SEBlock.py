import torch
import torch.nn as nn


class SEBlock(torch.nn.Module):
    
    '''
    An implementation of the Squeeze and Excitation Block as introduced in the
    "Squeeze and Excitation Networks" paper: https://arxiv.org/abs/1709.01507
    
    This layer is a lightweight method of getting attention-like operations
    inside a CNN. Surprisingly enough, this entire block which showed significant
    performance improvements on ImageNet and others is just scaling of channels.
    '''
    
    def __init__(
            self,
            in_channels: int,
            squeeze_factor: float,
            activation: nn.Module = nn.GELU,
            **kwargs
        ):
        super().__init__(**kwargs)
        
        assert 0 < squeeze_factor, f'ERROR: squeeze_factor should be greater than 0! Got {squeeze_factor=}'
        
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.squeezeChannelCount = int(in_channels * squeeze_factor)
        
        self.squeeze = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.squeezeChannelCount,
                kernel_size=1,
                bias=False
            )
        
        self.unsqueeze = nn.Conv2d(
            in_channels=self.squeezeChannelCount,
            out_channels=in_channels,
            kernel_size=1,
            bias=False
        )

        self.activation1 = activation()
        self.activation2 = activation()
        
    def forward(self, x) -> torch.Tensor:
        
        # Get a global representation of the channels via simple channel-wise average.
        out = self.pooling(x)
        
        # Squeeze the channel count to compress information among channels
        out = self.squeeze(out)
        out = self.activation1(out)
        
        # Unsqueeze and apply sigmoid over channels to get scaling factors
        out = self.unsqueeze(out)
        out = self.activation2(out)
        out = nn.functional.sigmoid(out)
        
        # Scale the original channels by these new "activations" and return
        return x * out