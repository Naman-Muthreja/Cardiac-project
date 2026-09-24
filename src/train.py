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
def make_regions_split(df, block_size = 3000, seed = 1, test_frac = 0.15, val_frac = 0.15, demo_frac = 0.02, verbose=True):

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

    # For each gene, get its blocks in sorted order, then shuffle them
    for gene, gene_rows in df.groupby("gene"):
        blocks = np.array(sorted(gene_rows["block"].unique()))
        rng.shuffle(blocks)
        n = len(blocks)

        # Splits the dataframe using each split's respective fraction and the length of blocks
        n_demo = max(1, round(demo_frac * n))
        n_test = max(1, round(test_frac * n))
        n_val = max(1, round(val_frac * n))

        # Apply labels "demo", "test", "val", and "train" to the "assign" dictionary
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

    # Drops variants that are near the edge and keeps training rows. I keep all training rows, because
    # removing test variants within 201 bases of an edge already guarantees no test window overlaps a training window.
    before = len(df)
    df = df[(~df["near_edge"]) | (df["split"] == "train")]
    n_dropped = before - len(df)

    # Makes each individual dataframe, deletes the working columns
    scratch = ["block", "near_edge", "split"]
    demo_df = df[df["split"] == "demo"].drop(columns = scratch).reset_index(drop = True)
    train_df = df[df["split"] == "train"].drop(columns = scratch).reset_index(drop = True)
    val_df = df[df["split"] == "val"].drop(columns = scratch).reset_index(drop = True)
    test_df = df[df["split"] == "test"].drop(columns = scratch).reset_index(drop = True)

    # Prints out useful data collected by this function for data interpretability
    if verbose:
        print(f"Region split: {df['block'].nunique()} blocks of {block_size} bases")
        print(f"{n_dropped} edge variants dropped from test/val/demo")
        print(f"{n_dropped/before*100:.1f}%, zero from train")
        for name, part in [("train", train_df), ("val", val_df), ("test", test_df), ("demo", demo_df)]:
            print(f"{name:6s} {len(part):5d} rows ({len(part)/len(df)*100:4.1f}%)")
            print(f"{dict(part['label'].value_counts())}")
            if part["label"].nunique() < 3:
                print(f"       WARNING: {name} is missing a class")

    return train_df, val_df, test_df, demo_df


# Takes the minimum gap between two variants, later to be compared with window to see if make_regions_split is working
def min_gap(a_df, b_df):

    worst = np.inf
    for g in set(a_df["gene"]) & set(b_df["gene"]):
        a = np.sort(a_df[a_df["gene"] == g]["pos"].values)
        b = np.sort(b_df[b_df["gene"] == g]["pos"].values)
        if len(a) == 0 or len(b) == 0:
            continue

        # Searchsorted finds where each 'a' position would fit in sorted 'b', so only the two nearest 'b' positions need checking.
        idx = np.searchsorted(b, a)

        # Checks the left and right neighbour, keeps the smallest gap seen as "worst"
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
    return X, y


# Returns the auc of each gene-consequence pair. Inside one cell, gene and consequence are the same for every row,
# so a lookup rule scores exactly 0.500 there. Anything above that came from the DNA.
def within_stratum_auc(df, probs, min_cell = 10, verbose = True):

    # Looks up benign in the LABELS_TO_INDEX lookup, yielding 2
    benign_idx = LABELS_TO_INDEX["Benign"]

    # Probability of being pathogenic is 1 minus the probability of being benign
    pathogenic_prob = 1.0 - probs[:, benign_idx]

    # Resets the index back to [0,1,2,3...]
    df = df.reset_index(drop = True)

    # Makes an empty dictionary and intializes variables later to be used (starting at 0)
    per_cell = {}
    weighted_sum = 0.0
    total_weight = 0

    # Takes the gene, consequence pairs, and labels the pathogenic
    for (gene, cons), cell in df.groupby(["gene", "consequence"]):
        y = (cell["label"] != "Benign").astype(int).to_numpy()

        # Filters for untrustable cells, like cells with only one class or too few rows
        if y.min() == y.max() or len(cell) < min_cell:
            if verbose:
                reason = "one class only" if y.min() == y.max() else "below min_cell"
                print(f" skipped {gene}/{cons}: n={len(cell)}, {reason}")
            continue

        # Compares 's' (model predictions) to 'y', the answer key
        s = pathogenic_prob[cell.index.to_numpy()]
        auc = roc_auc_score(y, s)

        # Stores AUC and size into the per_cell dict
        per_cell[f"{gene}/{cons}"] = (auc, len(cell))

        # Weighted sum, so bigger cells count more
        weighted_sum += auc * len(cell)
        total_weight += len(cell)

    # After every cell: the weighted mean. nan instead of a crash if every cell was skipped.
    overall = weighted_sum / total_weight if total_weight else float("nan")

    # Prints the table once, after the loop
    if verbose:
        print("\n--- WITHIN-STRATUM AUC (lookup floor is 0.500 in every cell) ---")
        for k, (auc, n) in sorted(per_cell.items()):
            print(f"  {k:22s} {auc:.3f} (n={n})")
        print(f"  {'WEIGHTED MEAN':22s} {overall:.3f} (n={total_weight})")
    return overall, per_cell


# Gives a 95% confidence interval for the AUC-ROC metric
def bootstrap_auc(y_true, scores, n_boot = 2000, seed = 42):

    # Picks random row numbers, repeats allowed, using a seed so the picks come out the same every run.
    rng = np.random.default_rng(seed)

    # Converts the answer key and the predictions to NumPy arrays
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    # n is the number of test variants, out collects one AUC per round
    n = len(y_true)
    out = []

    # n_boot rounds. Each round picks n rows with repeats allowed and scores them.
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]

        # Skip any round where every picked label is the same
        if yb.min() == yb.max():
            continue

        out.append(roc_auc_score(yb, scores[idx]))

    # Sorts "out" from least to greatest, returns the average and the 2.5th and 97.5th percentiles
    out = np.sort(out)
    lower = out[int(0.025 * len(out))]
    upper = out[int(0.975 * len(out))]
    return float(np.mean(out)), float(lower), float(upper)


# Binary pathogenic vs benign AUC from a 3-class probability table
def binary_auc(y, probs):
    benign_idx = LABELS_TO_INDEX["Benign"]
    return roc_auc_score((y != benign_idx).astype(int), 1.0 - probs[:, benign_idx])


# Defines train_model, with several important parameters.
# patience = how many epochs in a row validation AUC may fail to improve before training stops early.
def train_model(df, epochs = 25, batch_size = 32, lr = 7e-4, weight_decay = 3e-4, seed = 42,
                patience = 8, evaluate_test = True):

    # Fixes the starting weights so the same split always gives the same result
    torch.manual_seed(seed)

    # Validates the number of benign variants equals the number of pathogenic variants in every cell
    for (gene, cons), cell in df.groupby(["gene", "consequence"]):
        n_path = (cell["label"] != "Benign").sum()
        n_ben  = (cell["label"] == "Benign").sum()
        assert n_path == n_ben, f"{gene}/{cons} has {n_path} pathogenic and {n_ben} benign. Run match_cells first."

    # Tries to use GPU before going to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Splits by genomic region
    train_df, val_df, test_df, demo_df = make_regions_split(df, seed = seed)

    # Stop the program if any test window overlaps a training window
    gap = min_gap(test_df, train_df)
    assert gap >= WINDOW, f"test and train are only {gap} bases apart, windows overlap"
    print(f"min gap test <-> train: {gap:.0f} bases (need >= {WINDOW})")

    # One hot encodes the splits
    X_train, y_train = prepare_tensors(train_df)
    X_val, y_val = prepare_tensors(val_df)
    X_test, y_test = prepare_tensors(test_df)

    train_ds = TensorDataset(X_train, y_train)

    # Prints how many times each class appeared in training
    counts = np.bincount(y_train.numpy(), minlength = 3)
    print("Counts of each class in the training:", dict(zip(LABELS, counts)))

    # Loads 32 sequences at a time, shuffled
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    model = CardiacCNN(seq_len = X_train.shape[2], n_classes = 3).to(device)

    # Inverse-frequency class weights, so the smaller classes are not ignored
    weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype = torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight = weights)

    # Adam optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    # CHANGED: the best epoch is chosen by validation BINARY AUC, not validation accuracy.
    # Binary AUC is the headline metric, and validation accuracy swung 34% to 53% between epochs,
    # which made choosing on it close to random. Training also stops early if validation AUC
    # has not improved for `patience` epochs in a row, so the model stops before it memorises.
    best_val_auc, best_state, best_epoch = -1.0, None, 0
    epochs_without_improvement = 0

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        # Average loss per training example, computed once after every batch has run
        train_loss = running_loss / len(train_ds)

        # Validation binary AUC for this epoch
        model.eval()
        with torch.no_grad():
            val_probs = torch.softmax(model(X_val.to(device)), dim = 1).cpu().numpy()
        val_auc = binary_auc(y_val.numpy(), val_probs)

        # Keep a copy of the weights whenever validation AUC improves
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val binary AUC: {val_auc:.4f}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Stopping early: no validation improvement for {patience} epochs.")
            break

    # Loads the parameters of the best epoch
    model.load_state_dict(best_state)
    model.eval()
    print(f"\nUsing weights from epoch {best_epoch} (best validation binary AUC {best_val_auc:.4f})")

    with torch.no_grad():

        # Validation metrics
        val_probs = torch.softmax(model(X_val.to(device)), dim = 1).cpu().numpy()
        val_preds = val_probs.argmax(axis = 1)
        print(f"\nFinal Validation accuracy: {(val_preds == y_val.numpy()).mean() * 100:.3f}")
        print(f"[VALIDATION] Three-class macro one-vs-rest AUC-ROC: "
              f"{roc_auc_score(y_val.numpy(), val_probs, multi_class = 'ovr', average = 'macro'):.3f}")
        print(f"[VALIDATION] Binary Pathogenic-vs-Benign AUC-ROC: {binary_auc(y_val.numpy(), val_probs):.3f}")

        # Training metrics, to compare against validation and spot memorising
        train_probs = torch.softmax(model(X_train.to(device)), dim = 1).cpu().numpy()
        train_preds = train_probs.argmax(axis = 1)
        print(f"\nFinal train accuracy: {(train_preds == y_train.numpy()).mean() * 100:.3f}")
        print(f"[TRAIN] Three-class macro one-vs-rest AUC-ROC: "
              f"{roc_auc_score(y_train.numpy(), train_probs, multi_class = 'ovr', average = 'macro'):.3f}")
        print(f"[TRAIN] Binary Pathogenic-vs-Benign AUC-ROC: {binary_auc(y_train.numpy(), train_probs):.3f}")

    # If evaluate_test is false, just return without touching the test set
    if not evaluate_test:
        return model, (X_test, y_test), demo_df, test_df

    with torch.no_grad():

        test_probs = torch.softmax(model(X_test.to(device)), dim = 1).cpu().numpy()
        test_preds = test_probs.argmax(axis = 1)
        yt = y_test.numpy()

        print(f"\nFinal test accuracy: {(test_preds == yt).mean() * 100:.3f}")
        print(f"Three-class macro one-vs-rest AUC-ROC: "
              f"{roc_auc_score(yt, test_probs, multi_class = 'ovr', average = 'macro'):.3f}   (lookup floor 0.717)")

        benign_idx = LABELS_TO_INDEX["Benign"]
        test_y_binary = (yt != benign_idx).astype(int)
        pathogenic_prob = 1.0 - test_probs[:, benign_idx]
        print(f"Binary Pathogenic-vs-Benign AUC-ROC: {roc_auc_score(test_y_binary, pathogenic_prob):.3f}   (lookup floor 0.500)")

        # Classification report and confusion matrix
        print(classification_report(yt, test_preds, target_names = LABELS, zero_division = 0))
        print(confusion_matrix(yt, test_preds))

        # 95% confidence interval on the binary AUC
        m, low, high = bootstrap_auc(test_y_binary, pathogenic_prob)
        print(f"\nBinary AUC 95% Confidence Interval: {low:.3f} to {high:.3f}")

        # AUC inside each gene-consequence cell
        within_stratum_auc(test_df, test_probs)

    return model, (X_test, y_test), demo_df, test_df