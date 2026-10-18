"""
Explain.py is the explainability mechanism for the model, and uses Integrated Gradients to convert "black box" predictions to data
that physicians can actually use. Integrated Gradients fulfill the completeness axiom and sensitivity axiom, meaning that credit for the score
adds up to roughly the variant score - baseline score, and that an input that changes the score gets a non-zero attribution, both of 
which helps provide the valuable information needed for the clinical report generation. 
In addition to Integrated Gradients the answers must pass many gates.
""" 
import numpy as np
import pandas as pd
import torch
import os
from captum.attr import IntegratedGradients

from encoding import encode_dataset
from scipy.stats import spearmanr
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

    # Intializes empty lists later to be used. Dna_parts will be the DNA credits (how much each mattered), feat_parts will be the feature credits (how valuable each was),
    # p_parts will be the model's predictions, p0_parts will be the predictions for the baseline, 
    # and delta_parts will be the convergence delta for each variant.
    dna_parts, feat_parts, p_parts, p0_parts, delta_parts = [], [], [], [], []

    # Runs on all the variants, going batch_size at a time, to split the dataset into chunks instead of IG running all at once.
    for start in range(0, len(x), batch_size):

        # Splits the dataset into parts, with slice specifying the range of indices.
        part = slice(start, start + batch_size)

        # Calculates integrated gradients, using both baslines and the model inputs, and also returns a convergence delta. 
        # Integrated gradients is  the difference between the input and baseline multiplied by the average sensitivity of the path, in order to find 
        # how sensitive a model is to a change in that input. It uses Middle Riemann Approximation in this case because a pure integral cannot be done by the computer,
        # but small chunks with the middle of each rectangle found can approximate IG.
        # Convergence delta has the formula sum of attributions - (model output change from the baseline to variant).
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

    # Adds each variant’s DNA credits over the 4 letters and 201 positions, plus its 49 feature credits, minus the difference
    # between the variant's score and baseline score.
    gap = (dna.sum(dim = (1, 2)) + feat.sum(dim = 1) - (p - p0)).abs()

    # Largest gap is found, and the largest gap must be < 0.01 for completeness axiom satisification
    print(f"Completeness: largest gap {gap.max():.4f}, average gap {gap.mean():.5f} (must be below {tolerance}); ")

     # Checks whether my formula and Captum agree
    print(f"Captum's convergence delta agrees to {(gap - delta.abs()).abs().max():.1e}")
    
    assert gap.max() < tolerance, "attributions do not add up; use more steps"
    assert (gap - delta.abs()).abs().max() < 1e-4, "my gap and Captum's convergence delta disagree"
    return gap

# Explains each variant, with score, the baseline score, and DNA total.
def explain(models, df, F, typical, feature_names, steps = 64):

    # Resets the index of the dataframe, and then one-hot encodes the dataset
    df = df.reset_index(drop = True)
    X = encode_dataset(df["sequence"].tolist())

    # Gets the models baseline sequences and features
    x_base, f_base = baselines(df, F, typical)

    # Calls the integrated gradients function and check_completeness function on these variants
    dna, feat, p, p0, delta = integrated_gradients(models, X, F, x_base, f_base, steps = steps)
    check_completeness(dna, feat, p, p0, delta)

    # Defines "out" using detatch() to create a new dataframe, with new columns, while still preserving the old one
    out = pd.DataFrame(feat.detach().numpy(), columns = feature_names)

    # Uses the functions called previously in this function to define several columns in "out"
    out.insert(0, "dna_attribution_total", dna.sum(dim = (1, 2)).numpy())
    out.insert(0, "baseline_score", p0.numpy())
    out.insert(0, "score", p.numpy())

    probs = predict(models, X, F, DEVICE)

    # Finds the probabilities (HCM, DCM, Benign) for each variant in a gene (MYH7, MYBPC3, TTN)
    for i, name in enumerate(LABELS):
        out[f"p_{name}"] = probs[:, i]

    # Finds the pathogenic score of each of the 3 models averaged in train.py
    with torch.no_grad():
        for k, m in enumerate(models):
            out[f"score_model_{k + 1}"] = pathogenic_prob([m], X, F).numpy()

    return out, dna.numpy()

# Checks if the IG of the 3-model average is equal to the IG of each individual model, averaged out.
def check_linearity(models, X, F, x_base, f_base, steps = 32):

    # Models together includes it being called as normal, but to find the models seperate, I use a list comprehension
    together = integrated_gradients(models, X, F, x_base, f_base, steps)[1]
    separate = torch.stack([integrated_gradients([m], X, F, x_base, f_base, steps)[1] for m in models]).mean(dim = 0)

    # Finds the highest difference between the models together and seperately
    gap = (together - separate).abs().max().item()

    # Prints the gap, stops the script if it is greater than 1e-5, and returns the seperate models
    print(f"Linearity: largest gap between IG(average) and average(IG) = {gap:.2e}")
    assert gap < 1e-5
    return separate

# Determines whether Integrated Gradients correctly predicts the direction in which a model’s score changes when an 
# important feature is reverted to its baseline value.
def check_ablation(models, X, F, f_base, feat_attr, top = 3):
    ig_values, real_changes = [], []

    with torch.no_grad():

        # Finds the pathogenicity probabilities of the model
        p = pathogenic_prob(models, X, F)

        # Picks the column numbers of its 3 biggest credits
        order = feat_attr.abs().argsort(dim = 1, descending = True)[:, :top]

        # For loop for each variant's top 3 features
        for j in range(top):
            cols = order[:, j]

            # Creates a clone and arranges the rows with indices
            F_moved = F.clone()
            rows = torch.arange(len(F))

            # I duplicate the features dataframe, revert the dataframe to the baseline, and calculate the signed value 
            # of the difference between those two pathogenic scores to see how much of an impact this change had.
            F_moved[rows, cols] = f_base[rows, cols]
            real_changes.append((p - pathogenic_prob(models, X, F_moved)).numpy())
            ig_values.append(feat_attr[rows, cols].numpy())

    # Compares the sign changes and score changes
    ig_values, real_changes = np.concatenate(ig_values), np.concatenate(real_changes)
    same_sign = np.mean(np.sign(ig_values) == np.sign(real_changes))

    # Correlation coefficicient is calculated to see if the IG attributions have a positive, 0, or negative relationship
    # with the changes in pathogenicity score.
    r = np.corrcoef(ig_values, real_changes)[0, 1]

    # Prints % of time sign is agreed on, and the correlation coefficient
    print(f"Ablation: IG and the real one-feature change agree in sign {same_sign:.0%} of the time, correlation {r:.2f}")
    assert same_sign > 0.9 and r > 0.7, "Ablation or correlation has failed"
    return same_sign, r

# Check_randomization calculates Integrated Gradients for my model, an untrained one, and one trained on shuffled labels. 
# If the test passes, the untrained model and shuffled label model should not have attributions that look similar to my trained model.
def check_randomization(models, X, F, x_base, f_base, feat_attr, fit_df, F_fit, seed = 42):

    torch.manual_seed(seed)

    # Defines "untrained" to be the CNN with random weights
    untrained = CardiacCNN(n_features = F.shape[1])
    untrained.eval()

    # Shuffles the labels inside of fit_df for the second check: a model that trains on shuffled labels.
    shuffled = fit_df.assign(label = np.random.default_rng(seed).permutation(fit_df["label"].to_numpy()))
    X_fit, y_shuffled = prepare_tensors(shuffled)
    shuffled_model = fit_one(X_fit, F_fit, y_shuffled, 80, 32, 7e-4, 3e-4, seed = 0, device = DEVICE)

    # Saves the original model's explanations
    real = feat_attr.numpy()

    # Identifies features that differ from the baseline
    differs = ((F - f_base).abs() > 1e-6).numpy()

    results = {}

    # For the untrained model and the shuffled-label model, calculate the Integrated gradients
    for name, m in [("untrained model", untrained), ("shuffled-label model", shuffled_model)]:

        # Calaculates integrated gradients on the untrained and the shuffled model
        other = integrated_gradients([m], X, F, x_base, f_base, steps = 32)[1].numpy()

        # Calculates the spearman correlation coefficient, rho, on variants that differ by more than two features
        # from the baseline, also excluding features that didn't differ from the baselines. 
        # The spearman correlation coefficient calculates how similarily in order two models rank features.
        rhos = [spearmanr(real[i][differs[i]], other[i][differs[i]]).correlation
                  for i in range(len(real)) if differs[i].sum() > 2]

        # Calculates the median of rhos
        median_rho = float(np.nanmedian(rhos))

        # Finds the index of the largest attribution per variant, and checks to see for what fraction that feature matches
        # between the control models and my model.
        same_top = float(np.mean(np.abs(real).argmax(1) == np.abs(other).argmax(1)))

        print(f"Randomization, {name}: median per-variant rank correlation {median_rho:.2f}," f" same top feature {same_top:.0%}")

        # Stops the script if my model is too similar to the control model
        assert median_rho < 0.3 and same_top < 0.4, f"the {name} gets explanations like the real model's"
        results[name] = (median_rho, same_top)

    return results

# Artificially implants a feature in the inputs for every variant, and changes all the labels (HCM, DCM, Benign) 
# associated with that particular variant, in order to see if Integrated Gradients is truly working.
# If Integrated Gradients is really working, then it will flag this new artificial feature above any other feature
def check_planted_feature(fit_df, F_fit, test_df, F_test, epochs = 80, seed = 42):
    rng = np.random.default_rng(seed)

    # Creates one randomized feature per fitting variant, mean 0 and std 1 to normalize the data and for feature equalization
    planted_fit = torch.tensor(rng.normal(size=(len(F_fit), 1)), dtype=torch.float32)

    # New random numbers for the test variants
    planted_test = torch.tensor(rng.normal(size=(len(F_test), 1)), dtype=torch.float32)

    disease = {"MYH7": "HCM","MYBPC3": "HCM", "TTN": "DCM"}

    # For each variant, if the planted value "v" is greater than 0, assign it to the respective disease based on gene name,
    # otherwise make the label be benign.
    label = lambda df, col: [disease[g] if v > 0 else "Benign" for g, v in zip(df["gene"], col[:, 0].numpy())]

    # Replaces the labels with synthetic labels found on conducting the label function on planted_fit and planted_test
    fit, test = (fit_df.assign(label = label(fit_df, planted_fit)), test_df.assign(label = label(test_df, planted_test)))

    # Concatenates the fit and test dataframes
    Ff, Ft = (torch.cat([F_fit, planted_fit], dim=1), torch.cat([F_test, planted_test], dim=1))

    # Prepares the tensors (training inputs + labels)
    X_fit, y_fit = prepare_tensors(fit)
    X_test, _ = prepare_tensors(test)

    # Trains my CNN on this synthetic dataframe
    m = fit_one(X_fit, Ff, y_fit, epochs, 32, 7e-4, 3e-4, seed=seed, device=DEVICE)

    # Finds the typical baseline for the Integrated Gradients to compare against, both DNA & feature
    typical = typical_benign(Ff, fit)
    x_base, f_base = baselines(test, Ft, typical)

    # Calculates the feature credits for the newly trained model
    attr = integrated_gradients([m], X_test, Ft, x_base, f_base, steps=32)[1]

    # Checks whether the aritifically implanted feature ranks first
    top_is_planted = (attr.abs().argmax(dim=1) == Ft.shape[1] - 1).float().mean().item()

    # Prints out the percent of variants that IG ranks the artificially implanted score #1. and stops script if that isn't > 90%
    print(f"Planted feature: IG ranks the artificially implanted column first for {top_is_planted:.0%} of test variants")
    assert top_is_planted >= 0.9

    return top_is_planted
    
    
    
