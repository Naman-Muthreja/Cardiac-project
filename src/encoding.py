import numpy as np
import torch

BASES = ["A", "C", "G", "T"]

# Enumerates the bases with b (the string letter) as the key, and i as the value
# Returns dictionary {"A" : 0, "C" : 1, "T" : 2, "G" :3}, to dedicate each base it's own coordinate
BASES_TO_INDEX = {b:i for i,b in enumerate(BASES)}

def one_hot_encode(seq):

    # Defines the matrix to be the amount of bases, which is 4, times the length of the sequence,
    # so it will return a 4 x 201 tensor.  It expects channels, length, because it is a 1D CNN,
    # and that is why it is 4, len(seq), and not len(seq), 4. It is creating the dataset with all 0s
    # but one-hot encoding will return 1s for the bases actually present.
    mat = np.zeros((4,len(seq)), dtype = np.float32)

    # Creates a for loop for each index-base pair
    for i, base in enumerate(seq): 

        # Makes sure its either A,C,T, or G
        if base in BASES_TO_INDEX:

            # Check the base, and see if it matches the index. When it does, assign the value of
            # 1.0, meaning that the base is present at that position.
            mat[BASES_TO_INDEX[base],i] = 1.0
    return mat

def encode_dataset(sequences):
    encoded_list = []
    for s in sequences:

        # Passing the string s into the function, taking each string one at a time and encoding it
        encoded_matrix = one_hot_encode(s) 

        # appending all of the one hot encodes into the encoded_lists list, because encoded_matrix
        # changes every time, and we need to store all of the one_hot_encode(s)
        encoded_list.append(encoded_matrix)

    # Makes an array of all N sequences, .stack is used to make another dimension, unifying 
    # all of the 4,201 arrays into one 3d array, which can later be chunked into parts, because
    # Pytorch uses batch_size for the 1D CNN.
    arr = np.stack(encoded_list)    
    
    # Converts to a pytorch array
    return torch.tensor(arr)