# train.py is the script responsible for training the CNN to optimize prediction accuracy
import pandas as pd
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

# Counts the amount of DCM and HCM cases, and then caps the amount of benign variants
# based on that amount, so that capping the # of benign variants works for smaller datasets too. 

def cap_benign(df, max_benign = None, seed = 42):

    # Counts the total number of pathogenic variants
    counts = df["label"].value_counts()
    total_pathogenic = counts.get("HCM", 0) + counts.get("DCM, 0")

    # Makes # benign variants 4 times HCM and DCM combined, because benignity is more common
    # than HCM and DCM. However, the previous amount of benign variants could case data biases.
    if max_benign is None:
        max_benign = 4 * total_pathogenic

    benign = df[df["label"] == "Benign"]
    other = df[df["label"] != "Benign"]

    # Caps the number of benign variants
    if len(benign) > max_benign:
        benign = benign.sample(n = max_benign, random_state = seed)

    capped = pd.concat([other, benign], ignore_index=True)
    print(f"Benign capped: {len(capped)}")
    return capped
 
# One hot encodes the dataset(X), and then returns the unique index for each label(Y)
def prepare_tensors(df):
    X = encode_dataset(df["sequence"].tolist())
    y = torch.tensor([LABELS_TO_INDEX[l] for l in df["label"]])
    return X,y

# Defines train_model, with several important parameters. 
def train_model(df, epochs = 40, batch_size = 32, lr = 1e-3, weight_decay = 1e-4, seed = 42, max_benign = None):

    df = cap_benign(df, max_benign=max_benign, seed=seed)
    # Returns X, which is the matrix of encoded sequences, and y, a vector of the label indexes.
    X, y = prepare_tensors(df)

    # Tries to use GPU before going to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Binds the DNA sequences and labels together
    full_ds = TensorDataset(X, y)
    n_total = len(full_ds)

    # Allocates 70 percent of data to training, 15 percent to testing, and 15 percent to validation
    # Int is used to chop off the decimal, since a whole number is needed.
    n_train = int(0.70 * n_total)
    n_validation = int(0.15 * n_total)

    # Subtraction used instead of int(0.15 * n_total) to prevent rounding issues and to make
    # sure every row is accounted for.
    n_test = n_total - n_train - n_validation

    # Makes a random seed for reproductibility, and then splits the data randomly
    seed_generator = torch.Generator().manual_seed(seed)
    train_ds, validate_ds, test_ds = random_split(full_ds, [n_train, n_validation, n_test], generator=seed_generator)

    # Returns the dictionary of how many times each class appeared, to verify class counts
    train_labels = y[train_ds.indices].numpy()
    counts = np.bincount(train_labels, min_length = 3)
    print("Counts of each class in the training:", dict(zip(LABELS, counts)))

    # Loads the dataset, 32 at a time
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validate_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = CardiacCNN(seq_len = X.shape[2], n_classes=3).to(device)

    # Intializes the weights, with a higher penalty for misclassifying DCM and HCM, to further 
    # prevent the model from biasing towards the benign class.
    weights = torch.tensor(counts.sum() / (3 * np.maximum(counts,1)), dtype = torch.float32).to(device)

   # Intializes CrossEntropyLoss, which compares the model's prediction to the correct
   # answer and scales loss logarithimically

    criterion = nn.CrossEntropyLoss(weights = weights)

