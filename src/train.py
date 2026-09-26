"""
train.py is the script responsible for training the CNN to optimize prediction accuracy. It also conducts the validation and test
data, using analysis methods like AUC-ROC and F1 scores.
"""
import pandas as pd
import copy
import numpy as np
import torch
import torch.nn as nn

# Importing the data analysis methods
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from encoding import encode_dataset
from model import CardiacCNN

# Intializes the 3 classes, and assigns a unique index position for each one
LABELS = ["HCM", "DCM", "Benign"]
LABELS_TO_INDEX = {label:idx for idx, label in enumerate(LABELS)}

WINDOW = 201


# Makes the # of benign variants match with the amount of pathogenic variants 1:1, also removing synonymous variants. This is so
# a model cannot just use simple guessing rules to inflate accuracy
def match_cells(df, ratio = 1.0, seed = 42, verbose = True):

    kept = []
    # Defines the genes and consequence from the dataset, and stores the rows in cells.
    for (gene, cons), cell in df.groupby(["gene", "consequence"]):

         # Pathogenic and benign are the subsets of cell that don't match and match with the benign label, respectively.
         pathogenic = cell[cell["label"] != "Benign"]
         benign = cell[cell["label"] == "Benign"]

        # Doesn't use cells wih 0 pathogenic or 0 benign variants
         if len(pathogenic) == 0 or len(benign) == 0:
             if verbose:
                 print(f"Dropped {gene}/{cons}: {len(pathogenic)} pathogenic, {len(benign)} benign")
             continue

         # Makes the number of pathogenic variants and benign variants to be of a ratio of each other. This drops synonymous variants because
         # there are no pathogenic synonymous variants
         n_pathogenic = min(len(pathogenic), int(len(benign) / ratio))
         n_benign = round(ratio*n_pathogenic)
         
         # Appends the pathogenic and benign variants
         kept.append(pathogenic.sample(n = n_pathogenic, random_state = seed))
         kept.append(benign.sample(n = n_benign, random_state = seed))

    # Concats the list of dfs from kept
    out = pd.concat(kept, ignore_index= True)
    return out


# Splits by genomic region instead of by row. Two variants 50 bases apart have
# 201-base windows that overlap, so under a random split one can land in train and the other in test, and the
# model has effectively already seen the test sequence. Block size is the size of each block used for the splitting of the genes.
def make_regions_split (df, block_size = 3000, seed = 1, test_frac = 0.15, val_frac = 0.15, demo_frac = 0.02, verbose=True ):

    # Rng for reproducibility, then the dataframe is copied
    rng = np.random.default_rng(seed)
    df = df.copy()

    # Defines block to be the gene name plus the position divided by the block_size. Returns the chunk number for each gene.
    df["block"] = df["gene"].astype(str) + "_" + (df["pos"]//block_size).astype(str)

    # Defines offset to be the modulus of pos divided by block_size, tells you how far into a chunk a variant sits
    offset = df["pos"] % block_size

    # Near edge is defined as when the variant is sitting less far along than 201 base pairs, from either back or front.
    df["near_edge"] = (offset < WINDOW) | (offset >= block_size - WINDOW)

    assign = {}

    # For each gene, and the its contents/columns, get the genes in sorted order based on gene type and position, and 
    # then shuffle them
    for gene, gene_rows in df.groupby("gene"):
        blocks = np.array(sorted(gene_rows["block"].unique()))
        rng.shuffle(blocks)
        n=len(blocks)

        # Splits the dataframe using each split's respective fraction and the length of blocks
        n_demo = max(1, round(demo_frac * n))
        n_test = max(1, round(test_frac * n))
        n_val = max(1, round(val_frac * n))

        # Apply labels "demo", "test", "val", and "train" to the "assign" dictionary, using previously defined sizes for 
        # how many variants should be in each dataset.
        for i, b in enumerate(blocks):
            if i < n_demo:
                assign[b] = "demo"
            elif i < n_demo + n_test:
                assign[b] = "test"
            elif i < n_demo + n_test + n_val:
                assign[b] = "val"
            else:
                assign[b] = "train"

        # Defines the split column to be block mapped to assign (returns the label of the dataframe each variant is in)
    df["split"] = df["block"].map(assign)

    # Drops variants that are near the edge and keeps training rows, then records the number of variants dropped. I keep 
    # all training rows, because the 201 base pairs rule from before already makes sure that test variants and training variants
    # don’t overlap, so applying the same rule to the training rows would unnecessarily throw away data
    before = len(df)
    df = df[(~df["near_edge"] | (df["split"] == "train"))]
    n_dropped = before - len(df)

    # Makes each individual dataframe as well, deletes unneccesary columns from the final CSV
    scratch = ["block", "near_edge", "split"]
    demo_df = df[df["split"] == "demo"].drop(columns=scratch).reset_index(drop = True)
    train_df = df[df["split"] == "train"].drop(columns = scratch).reset_index(drop = True)
    val_df = df[df["split"] == "val"].drop(columns = scratch).reset_index(drop = True)
    test_df = df[df["split"] == "test"].drop(columns = scratch).reset_index(drop = True)

    # Prints out useful data collected by this function for data interpretability
    if verbose:
        print(f"Region split: {df['block'].nunique()} blocks of {block_size} bases")
        print(f"{n_dropped} edge variants dropped from test/val/demo")
        print(f"{n_dropped/before*100:.1f}%, zero from train")

        # For each dataset, print out the number of rows, the percentage of the total dataframe that specific dataset covers, and
        # how many of each label (HCM, DCM, Benign) it contains
        for name, part in [("train",train_df), ("val",val_df), ("test",test_df), ("demo",demo_df)]:
            print(f"{name:6s} {len(part):5d} rows ({len(part)/len(df)*100:4.1f}%)")
            print(f"{dict(part['label'].value_counts())}")

    return train_df, val_df, test_df, demo_df


# Takes the minimum gap between two variants, later to be compared with window to see if make_regions_split is working
def min_gap(a_df, b_df):

    # For the each set of genes, if the length is 0, move on
    worst = np.inf
    for g in set(a_df["gene"]) & set(b_df["gene"]):
        a = np.sort(a_df[a_df["gene"] == g]["pos"].values)
        b = np.sort(b_df[b_df["gene"] == g]["pos"].values)
        if len(a) == 0 or len(b) == 0:
            continue

        # Searchsorted finds where each 'a' position would fit in sorted 'b', allowing us to check the nearest 'b' positions.
        idx = np.searchsorted(b, a)

        # Iteratively finds potential candidates, assigns the minimum gap as the "worst" gap.
        for i, p in zip(idx, a):
            candidates = []
            if i > 0:
                candidates.append(abs(p - b[i - 1]))
            if i < len(b):
                candidates.append(abs(p - b[i]))
            if candidates:
                worst = min(worst, min(candidates))
    return worst

# One hot encodes the dataset(X), and then returns the unique index for each label(Y)
def prepare_tensors(df):
    X = encode_dataset(df["sequence"].tolist())
    y = torch.tensor([LABELS_TO_INDEX[l] for l in df["label"]])
    return X,y

# Returns the auc of each gene-consequence pair
def within_stratum_auc(df, probs, min_cell = 10, verbose = True):

    # Looks up benign in the LABELS_TO_INDEX lookup, yielding 2
    benign_idx = LABELS_TO_INDEX["Benign"]

    # Defines pathogenic_prob, which is the probability of a 1 - the probability of being a benign variant
    pathogenic_prob = 1.0 - probs[:, benign_idx]

    # Resets the index back to [0,1,2,3...]
    df = df.reset_index(drop=True)

    # Makes an empty dictionary and intializes variables later to be used (starting at 0)
    per_cell = {}
    weighted_sum = 0.0
    total_weight = 0

    # Takes the gene, consequence pairs, and labels the pathogenic
    for (gene, cons), cell in df.groupby(["gene", "consequence"]):
        y = (cell["label"] != "Benign").astype(int).to_numpy()

    # Filters for untrustable cells, like cells with only one class or too little rows, gives a print statement of the reason
        if y.min() == y.max() or len(cell) < min_cell:
            if verbose:
                reason = "one class only" if y.min() == y.max() else "below min_cell"
                print(f" skipped {gene}/{cons}: n={len(cell)}, {reason}")
            continue

        # Compares ‘s’ (model predictions) to ‘y’, the answer keys, returning roc_auc_score
        s = pathogenic_prob[cell.index.to_numpy()]
        auc = roc_auc_score(y,s)

        # Stores AUC and size into the per_cell dict, gene and cons as unique keys
        per_cell[f"{gene}/{cons}"] = (auc, len(cell))

        # Calculates the weighted sum by multiplying by each auc and the total weight
        weighted_sum += auc * len(cell)
        total_weight += len(cell)

    # Returns the weighted_sum over total_weight
    overall = weighted_sum / total_weight

    # Returns important statistics, to 3 decimal places for numerical values
    if verbose:
         print("\n--- WITHIN-STRATUM AUC ---")
         for k, (auc, n) in sorted(per_cell.items()):
             print(f"  {k:22s} {auc:.3f} (n={n})")
         print(f"  {'WEIGHTED MEAN':22s} {overall:.3f} (n={total_weight})")
    return overall, per_cell

# Gives a confidence interval / margin of error for the AUC-ROC metric
def bootstrap_auc(y_true, scores, n_boot = 2000, seed = 42):

    # Picks random row numbers, repeats allowed, using a seed so the picks
    # come out the same every run.
    rng = np.random.default_rng(seed)

    # Converts the answer key and the predictions to NumPy arrays
    y_true = np.asarray(y_true)
    scores = np.asanyarray(scores)

    # Sets "n" as the number of variants and initializes out to be an empty list
    n = len(y_true)
    out = []

    # Randomly chooses variants from 0 to the final number's position. It does this process of randomly choosing these n amount
    # of variants n_boot amount of times. 
    for _ in range(n_boot):
        idx = rng.integers(0,n,n)
        yb = y_true[idx]

        # I skip all selections where all true labels are the same.
        if yb.min() == yb.max():
            continue

        # Appends the confidence interval to "out"
        out.append(roc_auc_score(yb, scores[idx]))

    # Sorts "out" from least to greatest, returns the average and the percentile
    out = np.sort(out)
    lower = out[int(0.025 * len(out))]
    upper = out[int(0.975 * len(out))]
    return float(np.mean(out)), float(lower), float(upper)
    

# Defines train_model, with several important parameters. 
def train_model(df, epochs = 25, batch_size = 32, lr = 7e-4, weight_decay = 3e-4, seed = 42, evaluate_test = True):

    torch.manual_seed(seed)
    # Validates the number of benign variants equals the number of pathogenic variants, assert stops the program if the
    # pathogenic variants count don't match benign
    for (gene, cons), cell in df.groupby(["gene", "consequence"]):
        n_path = (cell["label"] != "Benign").sum()
        n_ben  = (cell["label"] == "Benign").sum()
        assert n_path == n_ben, (f"{gene}/{cons} has {n_path} pathogenic and {n_ben} benign. ")

    # Tries to use GPU before going to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Calls the make_splits function
    train_df, val_df, test_df, demo_df = make_regions_split(df, seed = seed)

    # Stop the program if the minimum gap is less than window, which is 201, meaning that DNA windows overlap
    gap = min_gap(test_df, train_df)
    assert gap >= WINDOW, f"test and train are only {gap} bases apart, windows overlap"
    print(f"min gap test <-> train: {gap:.0f} bases (need >= {WINDOW})")

    #One hot encodes the splits
    X_train, y_train = prepare_tensors(train_df)
    X_val, y_val = prepare_tensors(val_df)
    X_test, y_test = prepare_tensors(test_df)

    train_ds = TensorDataset(X_train, y_train)
    validate_ds = TensorDataset(X_val, y_val)
    
    # Prints the dictionary of how many times each class appeared, to verify class counts
    counts = np.bincount(y_train.numpy(), minlength=3)
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

            # x batch is the sequence, y batch is the label (see prepare_tensors)
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)

                # Sets preds as the index of the highest prediction score a class got. Uses softmax to turn logits to probabilities.
                # For example, a benign variant would most likely have the highest score be from the benign class, so the output would be 2.
                preds = model(xb).argmax(dim=1)

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

    with torch.no_grad():

        # Plugs in validation dataset to model.py
        val_logits = model(X_val.to(device))

        # Converts the logits to probabilities from 0 to 1
        val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()

        # Horizontally checks each row for the highest prediction value (HCM, DCM, or Benign,
        # depending on the variant)
        val_preds = val_logits.argmax(dim=1).cpu()

        # Finds the test accuracy, different from the validation check because it
        # does not use a DataLoader. Finds accuracy by finding the mean truth values
        # of 1.0 or 0.0 (hence the float) and using .item() to extract the values. Mean is safe
        # here because there is only one big batch.
        val_acc = (val_preds == y_val).float().mean().item()

        # Prints the accuracy on the training rows that the model trained on (not validation_accuracy, that is unseen)
        print(f"\nFinal Validation naccuracy: {val_acc * 100:.3f}")
        
        # Calculates One-vs-Rest Macro AUC-ROC scores for validation
        val_ovr_auc = roc_auc_score(y_val.numpy(), val_probs, multi_class="ovr", average = "macro")
        
        print(f"[VALIDATION] Three-class macro one-vs-rest AUC-ROC: {val_ovr_auc:.3f}")

        # For binary AUC-ROC for comparison against REVEL and CADD, a binary class is made.
        benign_idx = LABELS.index("Benign")
        val_y_binary = (y_val.numpy() != benign_idx).astype(int)
        
        # Defines pathogenic_prob, which uses the fact that the probability of pathogenicity is
        # the probability of benignity.
        val_pathogenic_prob = 1.0 - val_probs[:, benign_idx]
        
         # Calculates binary AUC-ROC score
        val_binary_auc = roc_auc_score(val_y_binary, val_pathogenic_prob)
        print(f"[VALIDATE] Binary Pathogenic-vs-Benign AUC-ROC: {val_binary_auc:.3f}")   

        # Accuracy and AUC-ROC finding for training is very similar
        train_logits = model(X_train.to(device))

        train_probs = torch.softmax(train_logits, dim =1).cpu().numpy()

        train_preds = train_logits.argmax(dim=1).cpu() 

        train_acc = (train_preds == y_train).float().mean().item()

        print(f"\nFinal train accuracy: {train_acc * 100:.3f}")

        train_ovr_auc = roc_auc_score(y_train.numpy(), train_probs, multi_class="ovr", average = "macro")
        print(f"[TRAIN] Three-class macro one-vs-rest AUC-ROC: {train_ovr_auc:.3f}")

        benign_idx = LABELS.index("Benign")
        train_y_binary = (y_train.numpy() != benign_idx).astype(int)

        train_pathogenic_prob = 1.0 - train_probs[:, benign_idx]

        train_binary_auc = roc_auc_score(train_y_binary, train_pathogenic_prob)

        print(f"[TRAIN] Binary Pathogenic-vs-Benign AUC-ROC: {train_binary_auc:.3f}")        

    # If evaluate_test is false, just return the model weights without doing the final test.
    if not evaluate_test:
        return model, (X_test, y_test), demo_df, test_df

    # Turns on no_grad to reduce RAM usage and speed up the forward pass process (for the test this time)
    # Finding AUC-ROC and accuracy for test is very similar to train and validation
    with torch.no_grad():

       
        test_logits = model(X_test.to(device))


        test_probs = torch.softmax(test_logits, dim =1).cpu().numpy()

        
        test_preds = test_logits.argmax(dim=1).cpu() 

       
        test_acc = (test_preds == y_test).float().mean().item()

       
        print(f"\nFinal test accuracy: {test_acc * 100:.3f}")

        
        test_ovr_auc = roc_auc_score(y_test.numpy(), test_probs, multi_class="ovr", average = "macro")

        print(f"Three-class macro one-vs-rest AUC-ROC: {test_ovr_auc:.3f}")

        
        benign_idx = LABELS.index("Benign")
        test_y_binary = (y_test.numpy() != benign_idx).astype(int)

        # Defines pathogenic_prob, which uses the fact that the probability of pathogenicity is
        # 1 - the probability of benignity.
        pathogenic_prob = 1.0 - test_probs[:, benign_idx]

        # Calculates binary AUC-ROC score
        test_binary_auc = roc_auc_score(test_y_binary, pathogenic_prob)

        print(f"Binary Pathogenic-vs-Benign AUC-ROC: {test_binary_auc:.3f}")

        # Prints the classification report, with various data analysis methods like 
        # f1 score, prints by name rather than index.
        print(classification_report(y_test, test_preds, target_names = LABELS))

        # Prints the confusion matrix, which makes a table on exactly which variants were confused by the model
        print(confusion_matrix(y_test, test_preds))

        # Returns the binary AUC 95% confidence interval
        m, low, high = bootstrap_auc(test_y_binary, pathogenic_prob)
        print(f"Binary AUC 95% Confidence Interval: {low:.3f} to {high:.3f}")

        # AUC-ROC for each gene-consequence pair
        within_stratum_auc(test_df, test_probs)

        return model, (X_test, y_test), demo_df, test_df