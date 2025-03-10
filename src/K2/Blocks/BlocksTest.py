import torch
import torch.nn as nn
from BlurPool import BlurPool
import numpy as np

def main():
    from tqdm import tqdm
    
    H, W = torch.arange(16, 64), torch.arange(16, 64)
    K = torch.arange(1, 5)
    pbar = tqdm(total=(len(H) * len(W) * len(K)), desc='Tests Complete')
    
    # Sandbox for testing blocks
    for h in H:
        for w in W:
            for k in K:
                randomTensor = torch.randn((16, 3, h, w))
                
                # layer = ResidualBlock(in_channels=3, kernel_size=3)
                # layer = ResidualBlockVersatile(
                #     in_channels=3,
                #     out_channels=32
                # )
                # layer = DepthwiseSeparableConv2d(
                #     in_channels=3, 
                #     out_channels=16,
                #     kernel_size=3,
                #     stride=2
                # )
                
                layer = BlurPool(
                    out_channels=3,
                    stride=1,
                    kernel_size=k
                )
                
                output = layer(randomTensor).shape
                assert output == randomTensor.shape
                # expectedShape = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)(randomTensor).shape

                # assert output == expectedShape, f'ERROR: Output shape {output} doesn\'t match expected shape {expectedShape} for input shape {randomTensor.shape}'
                pbar.update(1)

    print('Tests complete!')
    



if __name__ == '__main__':
    main()