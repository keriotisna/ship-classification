import torch
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
import pandas as pd
from contextlib import nullcontext
import numpy as np
import copy

# TODO: Upgrade the profileModel function to better handle nested custom
# sequential blocks all the way to the pytorch primitive events.
# If a single sequential consists of another sequential of lower-level operations,
# this should be decomposed and broken into separate portions/events

# TODO: Refactor parts of the profiling, it gets really messy in some places.

# A set of keys that are skipped for accumulation across multiple runs
SKIPPED_KEYS = set(['layerName', 'param_count'])


def _profileModelSingleRun(
    model: torch.nn.Sequential,
    input_size: tuple
) -> list[dict]:
    '''
    Runs a single profiling instance on a model for a given instance size and returns a
    list containing per-layer statistics for the given model across training and testing regimes.
    
    Arguments:
        model: The model to be tested as a single torch.nn.Sequentian
        input_size: A tuple containing the shape of the inputs the model uses, used for random inputs
    
    Outputs:
        attributesList: A per-layer list of model profiling events and statistics broken down by train and test values.
    '''

    trainingEvents = _runModelProfileSingleRun(model, input_size, trainMode=True)
    inferenceEvents = _runModelProfileSingleRun(model, input_size, trainMode=False)

    # Filter events with a custom name
    trainingEvents = [event for event in trainingEvents if event.key.startswith('_')]
    inferenceEvents = [event for event in inferenceEvents if event.key.startswith('_')]
    attributesList = []

    assert len(list(model.children())) == len(trainingEvents) == len(inferenceEvents), \
        f'ERROR: Event count mismatch between training and inference events!'

    # Manually extract parameters from each event
    for trainingEvent, inferenceEvent, layer in zip(trainingEvents, inferenceEvents, model.children()):
        layerName = trainingEvent.key
        attributesList.append({
            'layerName': layerName,
            'param_count': sum([p.numel() for p in layer.parameters()]),
            
            'train_stats': {
                'cpu_time_μs': trainingEvent.cpu_time_total,
                'device_time_μs': trainingEvent.device_time,
                'device_memory_usage_mb': round(trainingEvent.device_memory_usage / (1024**2), 3),
                'megaflops': round(trainingEvent.flops / 1e6, 2),
            },
            'inference_stats': {
                'cpu_time_μs': inferenceEvent.cpu_time_total,
                'device_time_μs': inferenceEvent.device_time,
                'device_memory_usage_mb': round(inferenceEvent.device_memory_usage / (1024**2), 3),
                'megaflops': round(inferenceEvent.flops / 1e6, 2),
            }
        })
    
    return attributesList



def _runModelProfileSingleRun(
    model: torch.nn.Sequential,
    input_size: tuple,
    trainMode: bool
) -> list:
    '''
    Runs the actual profiling process for a model a single time. Can be done
    in training or evaluation mode to include additional resources required by 
    the training process.
    
    Arguments:
        model: The model to be evaluated
        input_size: A tuple representing the input shape
        trianMode: Whether to profile including gradients or not. 
    
    Outputs:
        events: A list of all events obtained by the profiler.
    '''
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    randomInput = torch.randn(input_size, device=device, dtype=torch.float)

    if trainMode:
        model.train()
        profileModeContext = nullcontext()
    else:
        model.eval()
        profileModeContext = torch.no_grad()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        with_flops=True, 
        with_modules=True, 
        record_shapes=True, 
        profile_memory=True) as profilerContext, profileModeContext:
        
        output = randomInput
        for i, layer in enumerate(model.children()):
            # Get the class name for each layer and add an "_" so we can filter it
            with record_function(f"_{type(layer).__name__} {i}"):
                output = layer(output)

    events = profilerContext.events()

    # Conglomerate FLOPS from parent events
    parentEvent = None
    for e in events:
        if e.key.startswith('_'):
            parentEvent = e
        else:
            parentEvent.flops += e.flops
    

    return events



def profileModelLayers(
    model: torch.nn.Sequential,
    input_size: tuple,
    num_runs: int = 50,
    average_runs: bool = True
) -> list[dict]:
    
    '''
    Profiles an entire model a number of times before returning a set of statistics
    '''
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    layerStatistics = None
    for i in range(num_runs):
        runAttributes = _profileModelSingleRun(model, input_size)
        if layerStatistics == None:
            layerStatistics = copy.deepcopy(runAttributes)
        else:
            # Accumulate attribute totals to average them later
            for totalStats, layerStats in zip(layerStatistics, runAttributes):
                for key, val in layerStats.items():
                    if key in SKIPPED_KEYS: # Don't accumulate string values for things like the layer name
                        continue
                    for statKey, statVal in layerStats[key].items():
                        totalStats[key][statKey] += statVal
    
    if average_runs:
        # Average out statistics from the previous runs
        for totalStats in layerStatistics:
            for key, val in totalStats.items():
                if isinstance(val, str) or key == 'param_count':
                    continue
                for statKey, statVal in totalStats[key].items():
                    totalStats[key][statKey] = statVal / num_runs
            
    return layerStatistics

def plotModelProfileGraphs(
        model: torch.nn.Sequential, 
        input_size: tuple, 
        printOriginalTable=False # TODO: Do something with this eventually
    ):
    
    """
    Prints and plots relevant model information to get a sense of model size and expected performance.
    
    Arguments:
        model: A Sequential representation of a model
        input_size: The shape of the expected input in the form (B, C, W, H)
        printoriginalTable: Whether or not to print the original tables from the profiling library
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    attributesList = profileModelLayers(model, input_size)
    
    deviceTimesTrain = []
    deviceMemoryUsagesTrain = []
    
    deviceTimesInference = []
    deviceMemoryUsagesInference = []
    
    megaflopCountsTrain = []
    megaflopCountsInference = []
    
    paramCounts = []
    
    for attr in attributesList:
        deviceTimesTrain.append(attr['train_stats']['device_time_μs']/1000)
        deviceMemoryUsagesTrain.append(attr['train_stats']['device_memory_usage_mb'])
        megaflopCountsTrain.append(attr['train_stats']['megaflops'])

        deviceTimesInference.append(attr['inference_stats']['device_time_μs']/1000)
        deviceMemoryUsagesInference.append(attr['inference_stats']['device_memory_usage_mb'])
        megaflopCountsInference.append(attr['inference_stats']['megaflops'])

        
        paramCounts.append(attr['param_count'])
    
    layerCount = len(attributesList)

    reverseSlice = slice(None, None, -1)
    layerNames = [a['layerName'] for a in attributesList][reverseSlice]


    y_pos = np.arange(layerCount)
    barWidth = 0.4
    plt.figure(figsize=(10, 8))
    plt.barh(y_pos - barWidth/2, deviceTimesTrain[reverseSlice], barWidth, label='Train')
    plt.barh(y_pos + barWidth/2, deviceTimesInference[reverseSlice], barWidth, label='Inference')
    plt.title('Device times')
    plt.yticks(y_pos, layerNames)
    plt.ylabel('Layer number')
    plt.xlabel('Layer time (ms)')
    plt.legend()
    plt.show()

    # Grouped bar chart for memory usage
    plt.figure(figsize=(10, 8))
    plt.barh(y_pos - barWidth/2, deviceMemoryUsagesTrain[reverseSlice], barWidth, label='Train')
    plt.barh(y_pos + barWidth/2, deviceMemoryUsagesInference[reverseSlice], barWidth, label='Inference')
    plt.title('Device Memory Usage (MB)')
    plt.yticks(y_pos, layerNames)
    plt.ylabel('Layer number')
    plt.xlabel('Memory usage (MB)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.barh(y_pos - barWidth/2, megaflopCountsTrain[reverseSlice], barWidth, label='Train')
    plt.barh(y_pos + barWidth/2, megaflopCountsInference[reverseSlice], barWidth, label='Inference')
    plt.title('MegaFLOPS per layer')
    plt.yticks(y_pos, layerNames)
    plt.ylabel('Layer number')
    plt.xlabel('MFLOPS')
    plt.legend()
    plt.show()

    plt.barh(range(layerCount), paramCounts[reverseSlice])
    plt.title('Parameter Counts')
    plt.yticks(range(layerCount), layerNames)
    plt.ylabel('Layer number')
    plt.xlabel('Parameters')
    plt.show()


def compareModelStatistics(
    models: dict[str, torch.nn.Module],
    input_size: tuple
) -> pd.DataFrame:
    
    '''
    A simple way to compare the total statistics for multiple models at once
    in a tabular format.
    
    Arguments:
        models: A dictionary formatted as modelName: model
    
    Returns:
        statistics: A Pandas DataFrame holding stats for each individual model profiled
    '''
    
    modelStats = []
    # TODO: This whole process is hideous, see if there is a cleaner way to do it at some point.
    for name, model in models.items():
        layerStats = profileModelLayers(model, input_size)
        layerDict = {'Model Name': name}
        for layerStat in layerStats:
            for statName, val in layerStat.items():
                # Accumulate numeric statistics for all the layers
                if statName == 'layerName':
                    continue
                
                if statName == 'param_count':
                    layerDict[statName] = layerDict.get(statName, 0) + val
                    continue
                
                if statName == 'train_stats':
                    keySuffix = '_train'
                elif statName == 'inference_stats':
                    keySuffix = '_inference'
                
                for statKey, statVal in layerStat[statName].items():
                    layerDict[statKey+keySuffix] = layerDict.get(statKey+keySuffix, 0) + statVal
        modelStats.append(layerDict)
        

    return pd.DataFrame(modelStats, columns=[
            'Model Name',
            'cpu_time_μs_train',
            'device_time_μs_train',
            'device_memory_usage_mb_train',
            'megaflops_train',
            
            'cpu_time_μs_inference',
            'device_time_μs_inference',
            'device_memory_usage_mb_inference',
            'megaflops_inference',
            
            'param_count',
        ])
    
