import torch
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt

# TODO: Upgrade the profileModel function to better handle nested custom
# sequential blocks all the way to the pytorch primitive events.
# If a single sequential consists of another sequential of lower-level operations,
# this should be decomposed and broken into separate portions/events

def _profileModelSingleRun(
    model: torch.nn.Sequential,
    input_size: tuple
):
    '''
    Runs a single profiling instance on a model for a given instance size.
    '''

    device = "cuda" if torch.cuda.is_available() else "cpu"
    randomInput = torch.randn(input_size, device=device, dtype=torch.float)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        with_flops=True, 
        with_modules=True, 
        record_shapes=True, 
        profile_memory=True) as profilerContext:
        
        output = randomInput
        for i, layer in enumerate(model.children()):
            # Get the class name for each layer and add an "_" so we can filter it
            with record_function(f"_{type(layer).__name__} {i}"):
                output = layer(output)

    originalEvents = profilerContext.events()

    # Conglomerate FLOPS from parent events
    parentEvent = None
    for e in originalEvents:
        if e.key.startswith('_'):
            parentEvent = e
        else:
            parentEvent.flops += e.flops


    # Filter events with a custom name
    events = [event for event in originalEvents if event.key.startswith('_')]
    attributesList = []

    assert len(list(model.children())) == len(events)

    # Manually extract parameters from each event
    for event, layer in zip(events, model.children()):
        key = event.key
        attributesList.append({
            'key': key,
            'cpu_time_μs': event.cpu_time_total,
            'device_time_μs': event.device_time,
            'device_memory_usage_mb': round(event.device_memory_usage / (1024**2), 3),
            'param_count': sum([p.numel() for p in layer.parameters()]),
            'megaflops': round(event.flops / 1e6, 2)
        })
    
    return attributesList



def profileModel(
        model:torch.nn.Sequential, 
        input_size:tuple, 
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

    attributesList = None
    NUM_RUNS = 50
    for i in range(NUM_RUNS):
        runAttributes = _profileModelSingleRun(model, input_size)
        if attributesList == None:
            attributesList = runAttributes
        else:
            # Accumulate attribute totals to average them later
            for a in attributesList:
                for totalStats, layerStats in zip(attributesList, runAttributes):
                    for key, val in layerStats.items():
                        if isinstance(val, str):
                            continue
                        totalStats[key] += val
    
    # Average out statistics from the previous runs
    for totalStats in attributesList:
        for key, val in totalStats.items():
            if isinstance(val, str):
                continue
            totalStats[key] = val / NUM_RUNS
    
    deviceTimes = []
    deviceMemoryUsages = []
    paramCounts = []
    megaflopCounts = []
    for attr in attributesList:
        deviceTimes.append(attr['device_time_μs']/1000)
        deviceMemoryUsages.append(attr['device_memory_usage_mb'])
        paramCounts.append(attr['param_count'])
        megaflopCounts.append(attr['megaflops'])
    
    layerCount = len(attributesList)

    reverseSlice = slice(None, None, -1)
    layerNames = [a['key'] for a in attributesList][reverseSlice]

    plt.barh(range(layerCount), deviceTimes[reverseSlice])
    plt.title('Device times')
    plt.yticks(range(layerCount), layerNames)
    plt.ylabel('Layer number')
    plt.xlabel('Layer time (ms)')
    plt.show()
    
    plt.barh(range(layerCount), deviceMemoryUsages[reverseSlice])
    plt.title('Device Memory Usage (MB)')
    plt.yticks(range(layerCount), layerNames)
    plt.ylabel('Layer number')
    plt.xlabel('Memory usage (MB)')
    plt.show()
    
    plt.barh(range(layerCount), paramCounts[reverseSlice])
    plt.title('Parameter Counts')
    plt.yticks(range(layerCount), layerNames)
    plt.ylabel('Layer number')
    plt.xlabel('Parameters')
    plt.show()

    plt.barh(range(layerCount), megaflopCounts[reverseSlice])
    plt.title('MegaFLOPS per layer')
    plt.yticks(range(layerCount), layerNames)
    plt.ylabel('Layer number')
    plt.xlabel('MFLOPS')
    plt.show()
