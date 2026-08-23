# train.py is the script responsible for training the CNN to optimize prediction accuracy
import pandas as pd
import copy
import numpy as np
import torch
import torch.nn as nn

# Importing the data analysis methods
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

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
    total_pathogenic = counts.get("HCM", 0) + counts.get("DCM", 0)

    # Makes # benign variants 4 times HCM and DCM combined, because benignity is more common
    # than HCM and DCM. However, the previous amount of benign variants could cause data biases.
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

    # Tries to use GPU before going to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The rest of the data
    rest_df, demo_df = train_test_split(df, test_size=0.02, stratify=df["label"], random_state=seed)
    train_val_df, test_df = train_test_split(rest_df, test_size=0.10/0.98, stratify=rest_df["label"], random_state=seed)
    train_df, val_df = train_test_split(train_val_df, test_size=0.20/0.88, stratify=train_val_df["label"], random_state=seed)

    X_train, y_train = prepare_tensors(train_df)
    X_val, y_val = prepare_tensors(val_df)
    X_test, y_test = prepare_tensors(test_df)

    train_ds = TensorDataset(X_train, y_train)
    validate_ds = TensorDataset(X_val, y_val)
    

    # Prints the dictionary of how many times each class appeared, to verify class counts
    counts = np.bincount(y_train.numpy())
    print("Counts of each class in the training:", dict(zip(LABELS, counts)))

    # Loads each dataset based on the proportional fraction of each split. Loads 32 sequences 
    # at a time.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validate_ds, batch_size=batch_size, shuffle=False)
    model = CardiacCNN(seq_len = X_train.shape[2], n_classes=3).to(device)

    # Intializes the inverse-frequency weights, with a higher penalty for misclassifying DCM and HCM, to further 
    # prevent the model from biasing towards the benign class.
    weights = torch.tensor(counts.sum() / (3 * np.maximum(counts,1)), dtype = torch.float32).to(device)

    # Intializes CrossEntropyLoss, which compares the model's prediction to the correct
    # answer and scales loss logarithimically. It combines Softmax too.
    criterion = nn.CrossEntropyLoss(weight = weights)

    # Sets the optimizer to the Adam (Adaptive moment estimation) optimizer, which uses an
    # adaptive parameter for each model weight. Weight decay penalizes large weights, so the model
    # learns patterns and not noise.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Goes to evaluation mode and returns accuracy
    def evaluate(loader):

        print("Evaluation has started for the training predictions")
        # Evaluation mode sets self.training = false, stopping the dropout function, to give the 
        # real accuracy
        model.eval()

        # Starts the count of correct and total at 0 (to be added)
        correct, total = 0, 0

        # torch.no_grad() disables gradient calculation to speed up computation
        with torch.no_grad():

            # x batch is the sequence, y is the label (see prepare_tensors)
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)

                # Sets preds as the index of the highest prediction score a class got. For example,
                # a benign variant would most likely have the highest score be from the benign class,
                # so the output would be 2.
                preds = model(xb).argmax(dim = 1)

                # Sets correct to the amount of indexes the model returned that matched the data
                # .sum().item() is used rather than mean() because mean assumes that each batch
                # size is the same, which may not be the case here.
                correct += (preds == yb).sum().item()
                total += xb.size(0)

                # Compares correct guesses to the total, finding accuracy as a percentage.
            return correct/total * 100
        
    # Defines two variables that will help remember the best version of the model (if it
    # performs worse later on).
    best_val_acc, best_state = 0.0, None
    train_loss = None
    
    for epoch in range(epochs):

      model.train()
      print("Training has started")   

      # Resets the running_loss (how a model is performing per epoch)
      running_loss = 0.0

      #x batch is the sequence, y is the label (see prepare_tensors)
      for xb, yb in train_loader:

          xb, yb = xb.to(device), yb.to(device)

          # Clears gradients so gradients do not accumulate for each pass
          optimizer.zero_grad()

          # Calls the forward pass defined in model.py on xb
          out = model(xb)

          # Calls Cross Entropy Loss to compare between the model output and correct output
          loss = criterion(out, yb)

          # Calls backprop, then takes a step in the Adams optimizer to update weights
          loss.backward()
          optimizer.step()

          # Calculates a running total of the error, xb.size maintains an accurate accumulator
          # by multiplying average loss by batch size (size(0)).
          running_loss += loss.item() * xb.size(0)

          # Also calculates the average loss per training example
          # Note that putting train_loss +=loss.item() would introduce biases because some
          # batches are smaller than others, but they would be given equal weight.
          train_loss = running_loss/len(train_ds)

          # Evaluates validation accuracy
      val_acc = evaluate(val_loader)

          # Tracks and deep-copies the best model weights based on highest validation accuracy
      if val_acc > best_val_acc:
         best_val_acc = val_acc
         best_state = copy.deepcopy(model.state_dict())

      print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}%")

    # Loads the parameters of the best trained model
    model.load_state_dict(best_state)
    model.eval()


    # Turns on no_grad to reduce RAM usage and speed up the forward pass process
    with torch.no_grad():

        print("Evaluation has started for the test predictions")

        # Plugs in the test dataset to model.py
        test_logits = model(X_test.to(device))

        # Converts the logits to probabilities from 0 to 1
        test_probs = torch.softmax(test_logits, dim =1).cpu().numpy()

        # Horizontally checks each row for the highest prediction value (HCM, DCM, or Benign,
        # depending on the variant)
        preds = test_logits.argmax(dim=1).cpu() 

        # Finds the test accuracy, different from the validation check because it
        # does not use a DataLoader. Finds accuracy by finding the mean truth values
        # of 1.0 or 0.0 (hence the float) and using .item() to extract the values. Mean is safe
        # here because there is only one big batch.
        test_acc = (preds == y_test).float().mean().item()

        # Prints the best model version's accuracy
        print(f"\nFinal test accuracy: {test_acc * 100:.3f}")

        # Calculates One-vs-Rest Macro AUC-ROC scores
        ovr_auc = roc_auc_score(y_test.numpy(), test_probs, multi_class="ovr", average = "macro")

        print(f"Three-class macro one-vs-rest AUC-ROC: {ovr_auc:.3f}")

        # For binary AUC-ROC for comparison against REVEL and CADD, a binary class is made.
        benign_idx = LABELS.index("Benign")
        y_binary = (y_test.numpy() != benign_idx).astype(int)

        # Defines pathogenic_prob, which uses the fact that the probability of pathogenicity is
        # 1 - the probability of benignity.
        pathogenic_prob = 1.0 - test_probs[:, benign_idx]

        # Calculates binary AUC-ROC score
        binary_auc = roc_auc_score(y_binary, pathogenic_prob)

        print(f"Binary Pathogenic-vs-Benign AUC-ROC: {binary_auc:.3f}")

        # Prints the classification report, with various data analysis methods like 
        # f1 score, prints by name rather than index.
        print(classification_report(y_test, preds, target_names = LABELS))

        print(confusion_matrix(y_test, preds))

        return model, (X_test, y_test), demo_df