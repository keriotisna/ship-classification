import torch
import torch.nn as nn
import numpy as np
from K2.Blocks import ShuffledLinearBottleneck

def main():
    from tqdm import tqdm
    
    printShapes = False

    MAX_INPUT_SIZE = 81
    if printShapes:
        MAX_INPUT_SIZE = 20
    
    H, W = torch.arange(32, MAX_INPUT_SIZE), torch.arange(32, MAX_INPUT_SIZE)
    K = torch.arange(3, 4)
    pbar = tqdm(total=(len(H) * len(W) * len(K)), desc='Tests Complete')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda'
    
    
    # Sandbox for testing blocks
    for h in H:
        for w in W:
            for k in K:
                randomTensor = torch.randn((16, 3, h, w), device=device)
                
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
                
                # layer = BlurPool(
                #     out_channels=3,
                #     stride=1,
                #     kernel_size=k
                # )

                layer = ShuffledLinearBottleneck(in_channels=3, out_channels=8, kernel_size=3, stride=2).to(device)
                

                
                try:
                    output = layer(randomTensor)
                    if printShapes:
                        print(f'Input shape: {randomTensor.shape}\nOutput shape: {output.shape}')
                        print()
                except Exception as e:
                    print()
                    print(f'ERROR')
                    print(f'{randomTensor.shape=}')
                    print(f'{output.shape=}')
                    print(f'Kernel size: {k}')
                    print()
                    raise e



                # assert output.shape == randomTensor.shape, f'{output.shape=}, {randomTensor.shape=}'
                # expectedShape = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)(randomTensor).shape
                # assert output == expectedShape, f'ERROR: Output shape {output} doesn\'t match expected shape {expectedShape} for input shape {randomTensor.shape}'
                pbar.update(1)

    print('Tests complete!')
    



if __name__ == '__main__':
    main()