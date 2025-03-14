import torch
import torch.nn as nn

# TODO: Check out the ShuffleNet block and copy it because it looks goofy ahh
# https://arxiv.org/abs/1707.01083


class ShuffleBlock(torch.nn.Module):
    
    '''
    An implementation of the grouped shuffling as introduced in the ShuffleNet paper.
    https://arxiv.org/abs/1707.01083
    
    This block serves as a simple, yet effective way of distributing information
    from grouped convolutions between groups. If we perform multiple grouped convolutions
    without passing information between channels with a pointwise convolution, the groups
    remain effectively separated from one another meaning one group's information can't
    contribute to another group's. This leads to isolated channel activations where 
    individual groups cannot influence one another.
    
    To get around this, ShuffleNet introduces a shuffling operation, where we simply shuffle 
    the channel activations between groups, leading to improved performance since each group
    now has information from all other groups.
    
    # Example with 32 channels and 4 groups.
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
    [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3]
    # Marked for clarity
    [[0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 2, 2, 3, 3]]

    # Note that for cases where groups > channels, we may get imperfect information spreading
    since there are not enough channels in each group to hold a channel from every other group
    
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7]
    [0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7]
    # Here, information only gets spread through half the channels
    [[0, 2, 4, 6], [0, 2, 4, 6], [0, 2, 4, 6], [0, 2, 4, 6], [1, 3, 5, 7], [1, 3, 5, 7], [1, 3, 5, 7], [1, 3, 5, 7]]

    '''
    
    def __init__(
        self,
        groups,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.groups = groups
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, C, H, W = x.shape
        # Here, we abuse the read order for reshaping to spread the channels how we want.
        x = x.reshape(B, self.groups, -1, H, W)
        x = x.transpose(2, 1)
        x = x.reshape(B, C, H, W)
        return x