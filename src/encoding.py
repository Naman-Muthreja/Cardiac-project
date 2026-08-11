import numpy as np
import torch

BASES = ["A", "C", "G", "T"]

# Enumerates the bases with b (the string letter) as the key, and i as the value
# Returns dictionary {"A" : 0, "B" : 1, "C" : 2, "D" :3}, to dedicate each base it's own coordinate
BASES_TO_INDEX = {b:i for i,b in enumerate(BASES)}

def one_hot_encode(seq):

    # Defines the matrix to be the amount of bases, which is 4, times the length of the sequence,
    # so it will return a 4 x 201 tensor.  It expects channels, length, because it is a 1D CNN,
    # and that is why it is 4, len(seq), and not len(seq), 4.
    mat = np.zeros((4,len(seq)), dtype = np.float32)

    
