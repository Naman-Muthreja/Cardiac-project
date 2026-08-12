import torch
import torch.nn as nn 

class CardiacCNN(nn.module):

    # Takes in input length 201, matching encoding.py, and also outputs from a range of 3 classes,
    # which are HCM, DCM, and Benign
    def __init__(self, seq_len = 201, n_classes = 3):
        super.__init__()

        # Does the first convolutional layer, goes from 4 bases input, to 32 different layers for 
        # the 1D CNN to use to improve efficiency. Kernel size 11 takes in biological data without 
        # overwhelming the model and taking too much time, and padding gets added as 5 to prevent biases
        # between the amount of times the first base is read when compared to the middle.
        self.conv1= nn.Conv1d(in_channels = 4, out_channels = 32, kernel_size = 11, padding = 5)
        # Output equals new input
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)