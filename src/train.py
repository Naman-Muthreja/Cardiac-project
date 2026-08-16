# train.py is the script responsible for training the CNN to optimize prediction accuracy

import copy
import numpy as np
import torch
import torch.nn as nn

# Importing the data analysis methods
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, random_split

from encoding import encode_dataset
from model import CardiacCNN

# Intializes the 3 classes, and assigns a unique index position for each one
LABELS = ["HCM", "DCM", "Benign"]
LABELS_TO_INDEX = {label:idx for idx, label in enumerate(LABELS)}

# One hot encodes the dataset(X), and then returns the unique index for each label(Y)
def prepare_tensors(df):
    X = encode_dataset(df["sequence"].tolist())
    y = torch.tensor([LABELS_TO_INDEX[l] for l in df["label"]])
    return X,y

# Defines train_model, with several important parameters. 
def train_model(df, epochs = 40, batch_size = 32, lr = 1e-3, weight_decay = 1e-4, seed = 42):

    # Returns X, which is the matrix of encoded sequences, and y, a vector of the label indexes.
    X, y = prepare_tensors(df)

    # Tries to use GPU before going to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"