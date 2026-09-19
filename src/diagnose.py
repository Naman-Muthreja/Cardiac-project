"""
Diagnose.py is a simple lookup table that uses basic biological rules to classify the variants.

If the returned data metrics for diagnose.py are high, that means that the data is flawed and should include more data 
from varied consequence classes. 

If the score is close to the neural network, it also means that the neural network is learning genetic code 
rather than real biological signals.
"""
import pandas as pd
from sklearn.metrics import roc_auc_score

from consequence import parse_consequence
from train import make_splits, match_cells

# Adds the consequence column to a copied dataframe, which is later used to validate that the metrics are calculated on a dataframe with a variety of 
# consequences. If the amount of consequences on a particular label are heavily one-sided, this favors the lookup table. The inverse favors a well
# trained model.
def add_consequence(df):

    # Copies the dataframe because withhout copy, the function would add a column to the original caller dataframe
    df = df.copy()

    if "consequence" not in df.columns:
        df["consequence"] = df["name"].apply(parse_consequence)
    return df

# Basic rule #1
def genetic_code_baseline(df):

    # Turns pathogenic into 1, benign into 0, using vectorized comparison
    pathogenic_vs_benign_labels = (df["label"] != "Benign").astype(int)

    # If the variant is missense or nonsense, the lookup table rules makes it so that it will be labeled as pathogenic.
    score = df["consequence"].isin(["missense", "nonsense"]).astype(int)

    # returns the AUC_ROC score, will be called to check the lookup table's prediction classifying accuracy
    return roc_auc_score(pathogenic_vs_benign_labels, score)


# Similar to genetic_code_baseline, assesses all variants with a stop codon as DCM. Basic rule #2
def stop_codon_baseline(df):
    dcm_vs_nondcm_labels = (df["label"] == "DCM").astype(int)
    score = (df["consequence"] == "nonsense").astype(int)
    return roc_auc_score(dcm_vs_nondcm_labels, score)

# Basic rule #3
def gene_only_hcm_dcm(fit_df, score_df):

    # Defines the train and testing data
    tr = fit_df[fit_df["label"] != "Benign"]
    te = score_df[score_df["label"] != "Benign"]

    # Defines rate to be the mean of the amount of DCM cases (1) and HCM (0), calculates per gene. Should be high for TTN.
    rate = tr.assign(d= (tr["label"] == "DCM").astype(int)).groupby("gene")["d"].mean()
    overall = (tr["label"] == "DCM").mean()

    # Defines y to be 1 for DCM and 0 for HCM
    dcm_vs_hcm = (te["label"] == "DCM").astype(int)

    # If there aren't both DCM and HCM classes, return nan
    if dcm_vs_hcm.nunique() < 2:
        return float("nan")


    return roc_auc_score(dcm_vs_hcm, te["gene"].map(rate).fillna(overall))

def gene_only_pathogenic(fit_df, score_df):
    

    # Checks for the label column, and returns true if not equal to benign (pathogenic), and false if it is equal to benign.
    # From there, it converts these truth values to integers, and adds this list of truth values to the dataframe, using groupby()
    # to separate by gene. I then take the mean, to see the pathogenicity indicator of each gene.
    rate = fit_df.assign(p = (fit_df["label"] != "Benign").astype(int)).groupby("gene")["p"].mean()

    # Returns the pathogenicity indicator of all genes (rate is per gene)
    overall = (fit_df["label"] != "Benign").mean()

    # The answer key
    pathogenic_vs_benign = (score_df["label"] != "Benign").astype(int)

    # Compares the "rate" predictions with the answer key, overall being the replacement if rate does not include a specific gene
    return roc_auc_score(pathogenic_vs_benign, score_df["gene"].map(rate).fillna(overall))

def run(dataset_path, seed = 42):

    
    df = add_consequence(pd.read_csv(dataset_path))

    # Runs the crosstabs (frequency counts of genes and consequence per label)
    print("\n--- CONSEQUENCE vs LABEL ---")
    print(pd.crosstab(df["consequence"], df["label"]))
    print("\n--- GENE vs LABEL ---")
    print(pd.crosstab(df["gene"], df["label"]))

    # Runs match_cells, which makes the ratio of benign to pathogenic roughly 1:1
    matched = match_cells(df, seed = seed)

    # Inherits match_cells from "matched" too, makes the dfs.
    fit_df, val_df, score_df, demo_df = make_splits(matched, seed = seed)

    # Whole dataset data metrics returned
    print(f"WHOLE DATASET")
    print(f"Genetic-code rule (path vs benign) : {genetic_code_baseline(matched):.4f}   <-- MUST be 0.5000")
    print(f"Gene-name rule    (path vs benign) : {gene_only_pathogenic(matched, matched):.4f}   <-- MUST be 0.5000")
    print(f"Stop-codon rule   (DCM vs rest)    : {stop_codon_baseline(matched):.4f}")
    print(f"Train-to-test rule(fit scores vs test scores): {gene_only_hcm_dcm(matched, matched):.4f}")

    # Test dataset data metrics returned
    print(f"TEST SPLIT, n={len(score_df)}")
    print(f"Genetic-code rule (path vs benign) : {genetic_code_baseline(score_df):.3f}")
    print(f"Stop-codon rule   (DCM vs rest)    : {stop_codon_baseline(score_df):.3f}")
    print(f"Gene-name rule    (HCM vs DCM)     : {gene_only_hcm_dcm(fit_df, score_df):.3f}")
    print(f"Train-to-test rule (fit scores vs test scores ) : {gene_only_pathogenic(fit_df, score_df):.4f}")
  