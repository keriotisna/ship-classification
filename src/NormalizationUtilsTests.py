import torch
from torch.utils.data import DataLoader, TensorDataset
from NormalizationUtils import getNormalizationStats
import random, time
from tqdm import tqdm

# TODO: Figure out where to put test classes like me

def testNormalizationUtils(
        device: str=None
    ) -> None:
    
    '''
    Validates the correctness of the Welford online algorithm by testing randomly
    against random samples.
    '''
    
    BATCH_SIZE = 16
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Validate the accuracy of the Welford normalization
    pbar = tqdm(range(1000))
    for i in pbar:
        
        B = random.randint(1, 2000)
        C = random.randint(1, 20)
        H = random.randint(1, 40)
        W = random.randint(1, 40)
        datasetSize = (B, C, H, W)
        
        randomData = torch.rand(size=datasetSize, device=device)
        randomDataset = TensorDataset(randomData, torch.ones((B,)))
        
        rawMean, rawStd = torch.mean(randomData, dim=(0,2,3)), torch.std(randomData, dim=(0,2,3))
        
        startTime = time.time()
        dl = DataLoader(randomDataset, batch_size=BATCH_SIZE, shuffle=True)
        fastMean, fastStd = getNormalizationStats(dl)
        onlineTime = round((time.time() - startTime)*1000, 2)

        pbar.set_description(f'Online time: {onlineTime}ms')
        
        assert torch.allclose(rawMean, fastMean), f'Error! Means are not close!'
        assert torch.allclose(rawStd, fastStd), f'Error! Standard deviations are not close!'
        
    print('Tests passed')



def main():
    testNormalizationUtils()

if __name__ == '__main__':
    main()