import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import v2
from tqdm import tqdm


class WelfordsOnlineStats:
    
    '''
    A class which computes running mean and variances using a numerically stable online algorithm. 
    
    https://changyaochen.github.io/welford/#numerical-stability
    
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    
    The dataset may be too large for us to store in memory all at once, and the cumulative sum
    of each sample may go out of range of fp32 precision, so we have to use a more numerically
    stable and online algorithm. 
    
    The Welford algorithm is a method to compute a running mean and variance while staying numerically stable. 
    '''
    
    def __init__(self, n_channels: int):
        '''
        Arguments:
            n_channels: The number of channels the input image has.
        '''
        self.n_channels = n_channels
        self.mean = torch.zeros(n_channels)
        self.M2 = torch.zeros(n_channels)
        self.count = 0
    
    def update(self, batch: torch.Tensor) -> None:
        '''
        Update running statistics with a batch of data all at once where the batch is size (B, C, H, W)
        
        Args:
            batch: Tensor of shape (B, C, H, W)
        '''
        B, C, H, W = batch.shape
        
        # Reshape to (B*H*W, C) to process all pixels for each channel
        pixels = batch.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Update count with total number of pixels
        numPixels = pixels.shape[0]
        self.count += numPixels
        
        # Update mean and M2 using the vectorized form of Welford's algorithm
        delta = pixels - self.mean
        self.mean += delta.sum(dim=0) / self.count
        delta2 = pixels - self.mean
        self.M2 += (delta * delta2).sum(dim=0)
    
    def to(self, device):
        '''Imitate the standard pytorch .to() function to send tensors to another device'''
        self.mean = self.mean.to(device)
        self.M2 = self.M2.to(device)
        return self

    @property
    def var(self) -> torch.Tensor:
        '''Compute variance from M2'''
        return self.M2 / (self.count-1)
    
    @property
    def std(self) -> torch.Tensor:
        '''Compute standard deviation from variance'''
        return torch.sqrt(self.var + 1e-8)



def getNormalizationStats(
    X: DataLoader, 
    transform: v2.Compose=None,
    epochs: int=1,
    max_iter: int=10_000
    ) -> tuple[torch.Tensor, torch.Tensor]:
    
    '''
    Gets the image normalization parameters per channel like the mean and standard deviation.
    You can change which device this is done on by changing the device of the input data.
    
    Arguments: 
        X: A DataLoader object containing the training dataset
        transform: An optional transform to include in the normalization process
        epochs: How many times to iterate over the dataset. Can be useful if you want to ensure
            you capture all possible image augmentations.
        max_iter: The max number of samples we want to handle before returning the results. Good if we 
            don't want to normalize over all of ImageNet or something.
    '''
    
    firstBatch = next(iter(X))[0]
    N, C, H, W = firstBatch.shape
        
    stats = WelfordsOnlineStats(n_channels=C)
    stats = stats.to(firstBatch.device)
    samplesSeen = 0
    
    for e in range(epochs):
        # TODO: We go through a DataLoader instead of the tensor itself just to double-triple check 
        # that how we normalize is exactly what the model will see. We can't screw up if there are
        # hidden transforms embedded in a custom Dataset object so they will get applied here.
        for x, _ in X:
            if transform:
                x = transform(x)
            stats.update(batch=x)
            samplesSeen += x.shape[0]
            
            if samplesSeen >= max_iter:
                return stats.mean, stats.std
    
    return stats.mean, stats.std
