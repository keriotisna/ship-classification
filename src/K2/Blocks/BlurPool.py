import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

# TODO: Maybe allow different strides in different directions
# TODO: Add an argument for changing the padding type like if we want 0 padding or something else
# TODO: ADD TEST CASES SOMEWHERE


class BlurPool(torch.nn.Module):
    
    '''
    An implementation of the BlurPool layer as introduced in the paper 
    "Making Convonlutional Networks Shift-Invariant Again"
    https://arxiv.org/abs/1904.11486
    
    This class utilizes a more robust way of downsampling an image while avoiding
    the common problem of aliasing by first applying a blurring kernel to the features
    of a dense (stride 1) convolution before performing any downsampling.
    
    If we wanted to apply a convolution with stride s using this method, we would:
        1. Apply the convolution with stride 1
        2. Apply the activation function
        3. Blur the features with some blurring kernel of size m
        4. Perform naive subsampling with the desired stride
    
    This same process can also be done to other downsampling methods like max pooling,
    we just apply the original strided layer with a dense layer with stride 1 and add the
    following layers.
    '''
    
    def __init__(
            self,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 2,
            paddingOffset: int = 0,
            **kwargs
        ):
        super().__init__(**kwargs)

        # How much to pad in each direction dependent on kernel size to ensure
        # the feature shapes after filtering match the input shape
        
        # Stolen from the author's github lol
        self.paddingSizes = [
                int((kernel_size-1)/2),                 # padding_left
                int(np.ceil((kernel_size-1)/2)),        # padding_right
                int((kernel_size-1)/2),                 # padding_top
                int(np.ceil((kernel_size-1)/2))         # padding_bottom
            ]
        
        # Add additional padding if desired
        self.paddingSizes = [padSize+paddingOffset for padSize in self.paddingSizes]


        # Create a standard Normal and sample it as much as we need to to create a 1D filter
        distribution = Normal(loc=0, scale=1)
        gaussianFilter1d = distribution.log_prob(torch.linspace(-3, 3, kernel_size)).exp()
        
        # Convolve the 1D filter with itself to get a 2d filter we can use for images
        gaussianFilter2d = gaussianFilter1d[:, None] * gaussianFilter1d[None, :]
        gaussianFilter2d /= torch.sum(gaussianFilter2d)
        
        # Reshape the filter to a suitable format for acting as the weights of a convolution
        # These weights take a form (out_channels, in_channels, kernel_height, kernel_width)
        gaussianFilter2d = gaussianFilter2d.reshape((1, 1, kernel_size, kernel_size))
        
        # The number of "in channels" is 1 since we want to apply this filter separately
        # to each channel of an activation. This also means we will need to ensure
        # we use grouped convolution in the forward pass
        gaussianFilter2d = gaussianFilter2d.repeat((out_channels, 1, 1, 1))

        # This is like doing self.gaussianFilter = gaussianFilter2d, but it keeps it on the same device
        # as the module itself which is needed for training on GPUs
        self.register_buffer('gaussianFilter', gaussianFilter2d)
        
        # Add padding to ensure the sizes match up BEFORE we do our strides
        self.pad = nn.ReflectionPad2d(padding=self.paddingSizes)
        
        self.stride = stride
        self.kernel_size = kernel_size

    
    def forward(self, x):
        paddedInput = self.pad(x)
        # Filter each channel individually with 1 group per channel after padding
        return nn.functional.conv2d(paddedInput, self.gaussianFilter, stride=self.stride, groups=x.shape[1])

        # TODO: Keep investigating this later
        # This ended up being slightly worse, but I still feel like it would be something worth trying
        # because it would save a lot of headache in network design if the SHAPES JUST LINED UP.
        # out = nn.functional.conv2d(paddedInput, 
        #     self.gaussianFilter, 
        #     stride=1, 
        #     groups=x.shape[1]
        # )
        # # We will adjust where we start our striding process to ensure the output shape from the BlurPooling 
        # # matches the shape of a standard convolution. Checking now if this kills performance or not.
        # hOffset = self.kernel_size - 1
        # wOffset = self.kernel_size - 1
        # return out[:, :, hOffset::self.stride, wOffset::self.stride]

