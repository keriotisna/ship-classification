import matplotlib.pyplot as plt

# TODO: Make this function work with mismatched image and grid sizes.
#   Just display as many images as we can or as many to fill the grid as needed, 
#   leaving empty space if needed.

def plotImageGridClassification(
        data: list, 
        labels: list,
        plotShape: tuple[int, int], 
        figsize: tuple[int, int], 
        **kwargs
    ):
    
    '''
    An easy way to plot multiple images and their corresponding labels for exploratory data visualization.
    
    Arguments:
        data: A list of individual images in the format (H, W, C).
        labels: A list of label names for each corresponding image.
        plotShape: The number of images to display in the format (width, height).
        figsize: The matplotlib figure size to display.
        **kwargs: 
            subplots_adjust: kwargs for the plt.subplots_adjust function.
    '''
    
    assert len(data) == len(labels), f'Length of data should equal length of labels! {len(data)=}, {len(labels)=}'
    assert plotShape[0] * plotShape[1] == len(data), f'Plot shape dimensions and number of samples do not match! {plotShape=} {len(data)=}'
    
    rows, cols = plotShape
    fig, axs = plt.subplots(*plotShape)
    fig.set_size_inches(*figsize)
    idx = 0
    
    for r in range(rows):
        for c in range(cols):
            axs[r,c].imshow(data[idx])
            axs[r,c].set_title(labels[idx])
            axs[r,c].set_axis_off()
            idx += 1
    
    if 'subplots_adjust' in kwargs:
        plt.subplots_adjust(
            **kwargs['subplots_adjust']
        )
    plt.show()