import torch
import torch.nn as nn 

# Functions as a blueprint for training.py
class CardiacCNN(nn.Module):

    # Takes in input length 201, matching encoding.py, and also outputs from a range of 3 classes,
    # which are HCM, DCM, and Benign.
    def __init__(self, seq_len = 201, n_classes = 3):
        super().__init__()

        # Does the first convolutional layer, goes from 4 bases input, to 32 different layers for 
        # the 1D CNN to use to improve efficiency. Kernel size 11 takes in biological data without 
        # overwhelming the model and taking too much time, and padding gets added as 5 to prevent biases
        # between the amount of times the first base is read when compared to the middle.
        self.conv1= nn.Conv1d(in_channels = 4, out_channels = 32, kernel_size = 11, padding = 5)
        # Output equals new input.
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Stores and normalizes all the out_channels layers using BatchNorm1d, whiches
        # forces the mean to be 0 and standard deviation as 1, while using learnable parameters to prevent
        # ruining ReLU (half of the values will go from negative to 0 if no learnable parameters 
        # were there) or amplifying background noise.
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        # Creates a max pooling function, which takes in the highest activation value from
        # each 2-base window, filtering out some noise.
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Defines the ReLU activation function, which uses nonlinear activations for complex
        # learning.
        self.relu = nn.ReLU()

        # Defines the dropout function to prevent overfitting.
        self.dropout = nn.Dropout(0.50)

        # Defines the reduced length after 3 max poolings (1 max pool per layer) using 
        # floor division to yield an integer answer.
        reduced_len = seq_len // 2 // 2 // 2 

        # Having too many parameters may cause my model to overfit and memorize the training set
        # rather than understand the biology. To prevent this, I defined fc1 and fc2 
        # before the forward pass, which uses nn.Linear() to ompress the 128 units into 16. 
        # Those 32 units will then map to 3 output classes (HCM, DCM, Benign). The formula used for
        # fc1 is xW^t + b.
        self.fc1= nn.Linear(reduced_len * 128, 16)
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):

        # Defining the first, second, and third convolutional layers
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # Flattening into a single vector per batch (by merging the sequence_length and 
        # number of channels to one). Start_dim = 1 ensures that flattening only occurs on the
        # first dimension (index starts with 0), meaning that the batch_size is not multiplied.
        # Note: If batch_size were to be multiplied, it would cause errors, because batch_size
        # seperates individual samples.
        x = x.flatten(start_dim = 1)

        # Calls the dropout function to prevent overfitting, does the first fully connected layer
        x = self.dropout(self.relu(self.fc1(x)))

        # Returns the logit score of each of the 3 outputs (HCM, DCM, Benign)
        x = self.fc2(x)
        return x
    
        # Note that SoftMax is not used yet, because CrossEntropyLoss has an inbuilt SoftMax function