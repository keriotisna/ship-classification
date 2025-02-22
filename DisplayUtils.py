import matplotlib.pyplot as plt

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
        data: A list of individual images in any format. Can be reshaped to match the display format with the reshapeFunc argument.
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
    
    plt.subplots_adjust(
        **kwargs['subplots_adjust']
    )
    plt.show()