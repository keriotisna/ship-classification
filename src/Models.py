import torch.nn as nn
from K2.Blocks import (
    ResidualBlockVersatile, 
    DepthwiseSeparableConv2d, 
    DepthwiseSeparableConv2dAntiAliased,
    ShuffledLinearBottleneck
)


baseNet = nn.Sequential(*[
    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.BatchNorm2d(num_features=8),
    
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.BatchNorm2d(num_features=16),

    
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.BatchNorm2d(num_features=32),


    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.BatchNorm2d(num_features=64),


    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.BatchNorm2d(num_features=128),
    
    nn.Flatten(),
    
    nn.Linear(in_features=128, out_features=64),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=64),

    nn.Linear(in_features=64, out_features=32),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=32),
    
    nn.Linear(in_features=32, out_features=2)
])

# Faster, takes 3:11 to run
residualNet = nn.Sequential(*[
    ResidualBlockVersatile(in_channels=3, out_channels=12, kernel_size=3, stride=2, groups=3),
    ResidualBlockVersatile(in_channels=12, out_channels=16, kernel_size=3, stride=2, groups=4),
    ResidualBlockVersatile(in_channels=16, out_channels=32, kernel_size=3, stride=2, groups=16),
    ResidualBlockVersatile(in_channels=32, out_channels=64, kernel_size=3, stride=2, groups=32),
    ResidualBlockVersatile(in_channels=64, out_channels=128, kernel_size=3, stride=2, groups=64),

    nn.Flatten(),
    
    nn.Linear(in_features=128, out_features=64),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=64),

    nn.Linear(in_features=64, out_features=32),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=32),
    
    nn.Linear(in_features=32, out_features=2)
])

# Slower, takes 4:07 to run
separableNet = nn.Sequential(*[
    DepthwiseSeparableConv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2),
    DepthwiseSeparableConv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
    nn.BatchNorm2d(num_features=16),
    
    DepthwiseSeparableConv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
    DepthwiseSeparableConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
    nn.BatchNorm2d(num_features=64),
    
    DepthwiseSeparableConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),

    nn.Flatten(),
    
    nn.Linear(in_features=128, out_features=64),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=64),

    nn.Linear(in_features=64, out_features=32),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=32),
    
    nn.Linear(in_features=32, out_features=2)
])

shiftNet = nn.Sequential(*[
    DepthwiseSeparableConv2dAntiAliased(in_channels=3, out_channels=8, kernel_size=3, stride=2),
    DepthwiseSeparableConv2dAntiAliased(in_channels=8, out_channels=16, kernel_size=3, stride=2),
    nn.BatchNorm2d(num_features=16),

    DepthwiseSeparableConv2dAntiAliased(in_channels=16, out_channels=32, kernel_size=3, stride=2),
    DepthwiseSeparableConv2dAntiAliased(in_channels=32, out_channels=64, kernel_size=3, stride=2),
    nn.BatchNorm2d(num_features=64),

    DepthwiseSeparableConv2dAntiAliased(in_channels=64, out_channels=128, kernel_size=3, stride=2),
    nn.AvgPool2d(kernel_size=3),

    nn.Flatten(),
    
    nn.Linear(in_features=128, out_features=64),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=64),

    nn.Linear(in_features=64, out_features=32),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=32),
    
    nn.Linear(in_features=32, out_features=2)
])


shuffleNet = nn.Sequential(*[
    ShuffledLinearBottleneck(in_channels=3, out_channels=8, kernel_size=3, stride=2, groups=1),
    ShuffledLinearBottleneck(in_channels=8, out_channels=16, kernel_size=3, stride=2, groups=2),
    nn.BatchNorm2d(num_features=16),

    ShuffledLinearBottleneck(in_channels=16, out_channels=32, kernel_size=3, stride=2, groups=4),
    ShuffledLinearBottleneck(in_channels=32, out_channels=64, kernel_size=3, stride=2, groups=4),
    nn.BatchNorm2d(num_features=64),

    ShuffledLinearBottleneck(in_channels=64, out_channels=128, kernel_size=3, stride=2, groups=4),
    nn.AvgPool2d(kernel_size=2),
    
    nn.Flatten(),
    
    nn.Linear(in_features=128, out_features=64),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=64),

    nn.Linear(in_features=64, out_features=32),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=32),
    
    nn.Linear(in_features=32, out_features=2)
])


shuffleNetMini = nn.Sequential(*[
    ShuffledLinearBottleneck(in_channels=3, out_channels=18, kernel_size=3, stride=3, groups=3),
    ShuffledLinearBottleneck(in_channels=18, out_channels=16, kernel_size=3, stride=3, groups=2),
    nn.BatchNorm2d(num_features=16),

    ShuffledLinearBottleneck(in_channels=16, out_channels=32, kernel_size=3, stride=3, groups=4),
    nn.BatchNorm2d(num_features=32),

    nn.AvgPool2d(kernel_size=3),
    
    nn.Flatten(),
    
    nn.Linear(in_features=32, out_features=32),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=32),

    nn.Linear(in_features=32, out_features=32),
    nn.ReLU(),
    nn.LayerNorm(normalized_shape=32),
    
    nn.Linear(in_features=32, out_features=2)
])
