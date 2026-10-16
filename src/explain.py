"""
Explain.py is the explainability mechanism for the model, and uses Integrated Gradients to convert "black box" predictions to data
that physicians can actually use. Integrated Gradients fulfill the completeness axiom and sensitivity axiom, meaning that all data is accounted for,
which helps provide the valuable information needed for the clinical report generation. In addition to Integrated Gradients from Captum, the answers must pass many thresholds.
""" 
import numpy as np
import pandas as pd
import torch
import os
from captum.attr import IntegratedGradients

from encoding import encode_dataset
from model import CardiacCNN
from model_extra_features import build
from sklearn.metrics import roc_auc_score
from train import (LABELS, LABELS_TO_INDEX, make_regions_split, prepare_tensors, make_features_tensors, fit_one, predict, train_model, binary_auc)


# Defines the directories, and that it will run on a cpu.
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(HERE, "..", "results", "final_models.pt")
DEVICE = "cpu"

# Loads the final models; matched is my table of variants, while setup comes from the final_setup function.
# If the models cannot be accessed, then it rebuilds them.
def load_final_models(matched, setup, path = MODELS_PATH):

    # Prepares the testing tensors
    X_test, y_test = prepare_tensors(setup["test_df"])

    # If the path of the saved models doesn't exist, rebuild them and then save them
    if not os.path.exists(path):
        models, (_, F_test_trained, _), _, _ = train_model(matched, seed = 49, evaluate_test = False)

        # Uses allclose() to compare F_test_trained and setup["F_test"] tensors, roughly checking them to be equal, and flagging them if they are not
        assert torch.allclose(F_test_trained, setup["F_test"]), "final_setup builds different features than the train_model function"
        probs = predict(models, X_test, setup["F_test"], DEVICE)
        os.makedirs(os.path.dirname(path), exist_ok = True)
        torch.save({"states": [m.state_dict() for m in models], "test_probs": torch.tensor(probs)}, path)
        print(f"Saved the 3 final models to {path}")

    # Loads the saved models and gets into evaluation mode, also appending each loaded model to "models"
    saved = torch.load(path)
    models = []
    for state in saved["states"]:
        m = CardiacCNN(n_features = state["features.0.weight"].shape[1])
        m.load_state_dict(state)
        m.eval()
        models.append(m)

    # Find the probabilities, the biggest gap between the probabilities and what was saved, and returns the models.
    probs = predict(models, X_test, setup["F_test"], DEVICE)
    gap = float(np.abs(probs - saved["test_probs"].numpy()).max())
    macro = roc_auc_score(y_test.numpy(), probs, multi_class = "ovr", average = "macro")
    print(f"Loaded models give the saved test predictions (largest difference {gap:.1e}); "
          f"test binary AUC {binary_auc(y_test.numpy(), probs):.3f}, macro {macro:.3f}")
    assert gap < 1e-5, "the loaded models are not the models that were saved"
    return models

def final_setup(matched, extra = None):

    # Splits the regions of the dataframe (train, val, test, demo) into genomic blocks
    train_df, val_df, test_df, demo_df = make_regions_split(matched, seed = 49, verbose = False)

    # fit_df is the validation and training dataframes, while the others are test and demo
    fit_df = pd.concat([train_df, val_df], ignore_index = True)
    others = [test_df, demo_df] + ([extra] if extra is not None else [])

    # Makes the feature tables (biological knowledge inputs)
    tables = make_features_tensors(fit_df, others)

    # Clean dictionary to be returned
    setup = {"fit_df": fit_df, "test_df": test_df, "demo_df": demo_df,
             "F_fit": tables[0], "F_test": tables[1], "F_demo": tables[2]}

    # For the VUS case study
    if extra is not None:
        setup["extra_df"], setup["F_extra"] = extra, tables[3]

    # Highest and lowest positions per gene
    gene_span = {g: (d["pos"].min(), d["pos"].max()) for g, d in fit_df.groupby("gene")}

    # Raw, unscaled features
    setup["raw"] = lambda df: build(df, gene_span = gene_span)

    # Returns the dictionary setup
    return setup

# Finds the chance of DCM or HCM (a pathogenic variant), which is 1 minus the chance of being benign.
def pathogenic_prob(models, x, f):
    benign = LABELS_TO_INDEX["Benign"]
    return torch.stack([1 - torch.softmax(m(x, f), dim = 1)[:, benign] for m in models]).mean(dim = 0)

# Finds the typical benign variant by finding the average of the feature columns in each gene-consequence block
def typical_benign(F_fit, fit_df):
     
    # Gene/Consequence becomes "cell"
    cell = (fit_df["gene"] + "/" + fit_df["consequence"]).to_numpy()

    # Filters for benign variants
    benign = (fit_df["label"] == "Benign").to_numpy()

    # Returns the average of each feature column from "cell"
    return {c: F_fit[(cell == c) & benign].mean(dim = 0) for c in np.unique(cell)}

def baselines(df, F, typical):

    # One-hot encodes the reference windows; gene/consequence becomes "cell"
    x_base = encode_dataset(df["ref_sequence"].tolist())
    cells = (df["gene"] + "/" + df["consequence"]).tolist()

    # Tries to find any missing rows, and asserts if there are missing cells.
    missing = sorted(set(cells) - set(typical))
    assert not missing, f"no typical benign variant for {missing}; the model never trained on this kind"

    # Returns the DNA baseline and the feature baseline
    f_base = torch.stack([typical[c] for c in cells])
    return x_base, f_base

# Runs the integrated gradients, x is the DNA sequences, f is the features, x_base is the DNA sequence baseline, and f_base is the features baseline. 
def integrated_gradients(models, x, f, x_base, f_base, steps= 64, batch_size = 256):

    # "ig" is Integrated Gradients being called on the DNA sequences and features
    ig = IntegratedGradients(lambda x, f: pathogenic_prob(models, x, f))

    # Intializes empty lists later to be used. Dna_parts will be the DNA sequences used, feat_parts will be the features used, p_parts will be the model's predictions, 
    # p0_parts will be the predictions for the baseline, and delta_parts will be the convergence delta for each variant.
    dna_parts, feat_parts, p_parts, p0_parts, delta_parts = [], [], [], [], []

    # From the start to the end of the DNA sequence, go batch_size at a time, to split the dataset into chunks instead of IG running all at once.
    for start in range(0, len(x), batch_size):

        # Splits the dataset into parts, with slice specifying the range of indices.
        part = slice(start, start + batch_size)

        # Calculates integrated gradients, using both baslines and the model inputs, and also returns a convergence delta. 
        # Integrated gradients is  the difference between the input and baseline multiplied by the average sensitivity of the path, in order to find 
        # how sensitive a model is to a change in that input. It uses Middle Riemann Approximation in this case because a pure integral cannot be done by the computer,
        # but small chunks with the middle of each rectangle found can approximate IG.
        # Convergence delta has the formula |sum of attributions - (model output change from the baseline to variant)|.
        (dna, feat), delta = ig.attribute((x[part], f[part]), baselines=(x_base[part], f_base[part]), 
                                          n_steps=steps, method="riemann_middle",  return_convergence_delta=True)

        # Appends the IG results
        dna_parts.append(dna.detach())
        feat_parts.append(feat.detach())
        delta_parts.append(delta.detach())

        # Finds the pathogenic probability of each part using probabilities
        with torch.no_grad():
            p_parts.append(pathogenic_prob(models, x[part], f[part]))
            p0_parts.append(pathogenic_prob(models, x_base[part], f_base[part]))

        return (torch.cat(dna_parts),torch.cat(feat_parts),torch.cat(p_parts),torch.cat(p0_parts),torch.cat(delta_parts))

# Checks whether the completeness axiom has been satisified, with less than 0.01 being that it has. Completeness axiom satisfication means that every attribution is
# accounter for; nothing comes out of thin air or disappears. 
def check_completeness(dna, feat, p, p0, delta, tolerance = 0.01):
    gap = (dna.sum(dim = (1, 2)) + feat.sum(dim = 1) - (p - p0)).abs()

    # Average gap must be < 0.01 for completeness axiom satisification
    print(f"Completeness: largest gap {gap.max():.4f}, average gap {gap.mean():.5f} (must be below {tolerance}); "
          f"Captum's convergence delta agrees to {(gap - delta.abs()).abs().max():.1e}")
    assert gap.max() < tolerance, "attributions do not add up; use more steps"
    assert (gap - delta.abs()).abs().max() < 1e-4, "my gap and Captum's convergence delta disagree"
    return gap
