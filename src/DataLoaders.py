import torch
from typing import Iterator, List, Optional

'''
An attempt to try and parallelize image transformations by creating a new
batched dataset which will perform its transform on the GPU instead of the CPU.

This will (hopefully) allow for faster iteration on augmented datasets but
I haven't actually benchmarked this implementation. I do know that GPU transformations
can be **much** faster than their CPU counterparts. This could be done inside the training
loop itself, but I wanted a cleaner solution for future projects. 
'''

class BatchSampler(torch.utils.data.Sampler[List[int]]):
    
    def __init__(
            self, 
            dataset: torch.utils.data.Dataset, 
            batch_size: int, 
            shuffle: bool=True
        ):
        
        self.numSamples = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    
    def __iter__(self) -> Iterator[List[int]]:
        
        indices = torch.randperm(self.numSamples) if self.shuffle \
            else torch.arange(self.numSamples)
        
        # Yield batch indices for each iteration
        for startIdx in range(0, self.numSamples, self.batch_size):
            yield indices[startIdx:startIdx+self.batch_size].tolist()
        
    def __len__(self) -> int:
        # This is just ceil(numSamples / batch_size) but doesn't use another function.
        return (self.numSamples + self.batch_size - 1) // self.batch_size


class GPUDataSet(torch.utils.data.Dataset):
    
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transform: Optional[callable] = None,
        device = None
    ):
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.dataset = dataset
        self.transform = transform
        self.device = device
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx) -> torch.Tensor:
        
        if not isinstance(type(idx), list):
            idx = [idx]
        
        idx = torch.Tensor(idx).to(torch.int64).to(self.device)
        x, y = self.dataset[idx]
        x, y = x.to(self.device), y.to(self.device)
        
        if self.transform:
            x = self.transform(x)
        
        return (x, y)



def getGPUDataLoader(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        transform: Optional[callable] = None,
        shuffle: bool = True,
        device = None
    ):
    
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    gpuDataset = GPUDataSet(
        dataset=dataset,
        transform=transform,
        device=device
    )
    
    sampler = BatchSampler(
        gpuDataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    return torch.utils.data.DataLoader(
        gpuDataset,
        batch_sampler=sampler
        # When using a batch sampler, both batch_size and shuffle should be false
        # batch_size=None,
        # shuffle=False
    )
