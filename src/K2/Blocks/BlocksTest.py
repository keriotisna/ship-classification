import torch
import torch.nn as nn
from K2.Blocks import DepthwiseSeparableConv2d


def main():
    from tqdm import tqdm
    
    H, W = torch.arange(16, 64), torch.arange(16, 64)
    pbar = tqdm(total=(len(H) * len(W)), desc='Tests Complete')
    
    # Sandbox for testing blocks
    for h in H:
        for w in W:
            randomTensor = torch.randn((16, 3, h, w))
            
            # layer = ResidualBlock(in_channels=3, kernel_size=3)
            # layer = ResidualBlockVersatile(
            #     in_channels=3,
            #     out_channels=32
            # )
            layer = DepthwiseSeparableConv2d(
                in_channels=3, 
                out_channels=16,
                kernel_size=3,
                stride=2
            )
            output = layer(randomTensor).shape
            pbar.update(1)

    print('Tests complete!')
    



if __name__ == '__main__':
    main()