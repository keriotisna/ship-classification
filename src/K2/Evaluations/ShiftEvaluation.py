import torch
import torch.nn as nn
from torchvision.transforms import v2 as v2
from torchmetrics.classification import BinaryF1Score


'''
An evaulation metric to provide statistics about how a model's logits change
when the input image is shifted diagonally by some amount. These shifts are
done in the 8 cardinal directions 1 pixel at a time for an entire DataLoader
'''


from collections import defaultdict

def _get8CardinalShifts(shift: int) -> list[tuple]:
    
    '''
    Gets the 8 cardinal direction shifts within a given shift range
    '''
    
    directions = [
        (1, 0),   # East
        (1, 1),   # Northeast
        (0, 1),   # North
        (-1, 1),  # Northwest
        (-1, 0),  # West
        (-1, -1), # Southwest
        (0, -1),  # South
        (1, -1)   # Southeast
    ]
    
    shifts = []
    for i in range(-shift, shift+1):
        for d in range(8):
            shifts.append((i*directions[d][0], i*directions[d][1]))
        
    return shifts



def getShiftLosses(
    shiftRange: int,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    normalizationTransform: v2.Compose,
    iterations: int = None
) -> dict[list]:

    '''
    Gets a set of losses from a given model and dataloader after shifting the image
    diagonally in each of the 8 cardinal directions by a given shift range.
    
    This can be used to probe for potential aliasing artifacts causing additional loss for 
    shifts of various degrees by analyzing the logits or probabilities of the correct class.
    
    Arguments:
        shiftRange: The range of shifts, a value of 3 means we will shift in each direction up to 3 pixels.
        model: The model to be evaluated.
        dataloader: The source of data for the model
        normalizationTransform: A transform for normalizing the data in a more effective GPU way
        iterations: The max number of batches to do, default is to traverse the entire dataloader.
    
    Returns:
        statistics: A dict containing various raw logits, probabilities, and losses. 
    '''

    
    shifts = _get8CardinalShifts(shiftRange)
    toImageTransform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    lossFunction = nn.CrossEntropyLoss(reduction='none')

    SHIFTED_DATA_STATISTICS = defaultdict(lambda: defaultdict(list))
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for shiftX, shiftY in shifts:
            f1 = BinaryF1Score().to(device)
            shiftMagnitude = max(abs(shiftX), abs(shiftY))
            if shiftMagnitude not in SHIFTED_DATA_STATISTICS['F1']:
                SHIFTED_DATA_STATISTICS['F1'][shiftMagnitude] = f1

            for i, (x, y) in enumerate(dataloader):
                
                x, y = x.to(device), y.to(device)
                x = normalizationTransform(toImageTransform(x))

                shifted = torch.roll(x, shifts=(shiftX, shiftY), dims=(2, 3))
                
                shiftedPredictions = model(shifted)
                shiftedLoss = lossFunction(shiftedPredictions, y).cpu().tolist()
                classProbabilities = nn.functional.softmax(shiftedPredictions, dim=1)
                correctClassProbabilities = classProbabilities[torch.arange(shiftedPredictions.shape[0]), y].cpu().tolist()
                
                _, maxIndices = torch.max(shiftedPredictions, dim=1)
                
                SHIFTED_DATA_STATISTICS['losses'][shiftMagnitude].extend(shiftedLoss)
                SHIFTED_DATA_STATISTICS['correctProbabilities'][shiftMagnitude].extend(correctClassProbabilities)
                
                SHIFTED_DATA_STATISTICS['F1'][shiftMagnitude].update(maxIndices, y)
            
            if i == iterations:
                break
        
        # Compute all the F1 statistics before returning
        for sm in SHIFTED_DATA_STATISTICS['F1']:
            SHIFTED_DATA_STATISTICS['F1'][sm] = SHIFTED_DATA_STATISTICS['F1'][sm].compute().detach().item()
        
    return SHIFTED_DATA_STATISTICS

