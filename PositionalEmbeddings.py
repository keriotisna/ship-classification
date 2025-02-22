import numpy as np


def getSinusoidalEmbedding(
    sequenceLength: int,
    embeddingDimensions: int,
    minFrequency: float = 1e-4
) -> np.ndarray:
    '''
    Gets sinusoidal embeddings as defined in the paper "Attention is All You Need".
    
    Arguments: 
        sequenceLength: The max expected token sequence length.
        
        embeddingDimensions: How many dimensions each embedding should have.
        
        minFrequency: The lowest frequency a positional embedding should have.
    '''
    
    assert sequenceLength > 0, f"Seqwuence length must be greater than 0!"
    assert embeddingDimensions > 0, f"Embedding dimension must be greater than 0!"
    assert minFrequency > 0, f"Minimum frequency must be greater than 0!"
    
    # Refer to my notes on Sinusoidal Embeddings in Obsidian for a detailed description on how this works.
    tokenPositions = np.arange(1, sequenceLength+1)
    frequencies = minFrequency**((2*(np.arange(embeddingDimensions)//2))/embeddingDimensions)
    pos_enc = tokenPositions.reshape(-1,1) * frequencies.reshape(1,-1)
    pos_enc[:, ::2] = np.sin(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
    # Normalize encodings so each one has a unit norm
    pos_enc /= np.sqrt(embeddingDimensions/2)
    return pos_enc