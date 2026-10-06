"""
Model.py functions as a blueprint for training.py, including the batch normalization, dropout, convolutional layers, and neuron outputs.
"""
import torch
import torch.nn as nn 


# CardiacCNN reads the genomic window, converts it into just 16 numbers, and uses these numbers to give one score per variant class.
# Note that unlike the previous pooling, CardiacPoolNet does not have the error of too many weights for a little amount of rows, so it is less prone to overfitting
class CardiacCNN(nn.Module):

    # Initializes the model. Conv_dropout reduces model dependencies on certain weights.
    def __init__(self, n_features, n_classes=3, conv_dropout=0.5):
        super().__init__()           

        # nn.Sequential() runs the convolutional layers in order 
        # BatchNorm1d makes sure that the mean is 0 and std is 1, normalizing the data and fighting Vanishing/Exploding Gradients
        # ReLU is a nonlinear transformation that makes all negative numbers 0, also fights vanishing gradients by making sure that no gradients become 0
        # Padding prevents biases between the amount of times the first base is read vs the middle ones, and keeps the sequence length 201.
        # Flatten multiplies sequence length with the number of channels, in this case just removing the extra dimension of size 1.
        self.conv = nn.Sequential(
        nn.Conv1d(in_channels = 4, out_channels = 16, kernel_size = 11, padding=5), nn.BatchNorm1d(16), nn.ReLU(), # Takes 4 DNA channels, uses 16 filters, kernel size of 11
        nn.Conv1d(in_channels= 16, out_channels = 16, kernel_size = 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),  # Second convolutional layer keeps the amount of channels as 16, to prevent overfitting.
        nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Dropout(conv_dropout)) # AdaptiveMaxPool1d takes the highest activation score for each of the 16 filters

        # Makes a feature layer for the model to adapt and learn from the table biological data constructed in model_extra_features
        # Similarily uses dropout to prevent dependancies.
        self.features = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU())
        
        # Joins the two branches: one score per class (HCM, DCM, Benign), for each variant
        self.out = nn.Linear(16 + 16, n_classes)
        
    # Forward pass that joins the convolutional layers and features together, returning a score for each class.
    def forward(self, x, f):
        return self.out(torch.cat([self.conv(x), self.features(f)], dim=1))

        # Note that SoftMax is not used yet, because CrossEntropyLoss has an inbuilt SoftMax function

    
# -------- OUT OF USE (Old version, just for comparison) ---------
class OldCardiacCNN(nn.Module):

    # Takes in input length 201, matching encoding.py, and also outputs from a range of 3 classes,
    # which are HCM, DCM, and Benign.
    def __init__(self, seq_len = 201, n_classes = 3):
        super().__init__()

        # Does the first convolutional layer, goes from 4 bases input, to 32 different layers for 
        # the 1D CNN to use to improve efficiency. Kernel size 11 takes in biological data without overwhelming the model and taking too much time.
        # Padding gets added as 5 to prevent biases between the amount of times the first base is read when compared to the middle. Also keeps sequence length as 201.
        self.conv1= nn.Conv1d(in_channels = 4, out_channels = 32, kernel_size = 11, padding = 5)

        # Output equals new input.
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Stores and normalizes all the out_channels layers using BatchNorm1d, whiches
        # forces the mean to be 0 and standard deviation as 1, while using learnable parameters to prevent
        # ruining ReLU (half of the values will go from negative to 0 if no learnable parameters were there) or amplifying background noise.
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

        # Flattening into a single vector per batch (by multiplying the sequence_length and thenumber of channels). Start_dim = 1 ensures that flattening 
        # only occurs on the first dimension (index starts with 0), meaning that the batch_size is not multiplied.
        # Note: If batch_size were to be multiplied, it would cause errors, because batch_size seperates individual samples.
        x = x.flatten(start_dim = 1)

        # Calls the dropout function to prevent overfitting, does the first fully connected layer
        x = self.dropout(self.relu(self.fc1(x)))

        # Returns the logit score of each of the 3 outputs (HCM, DCM, Benign)
        x = self.fc2(x)
        return x

