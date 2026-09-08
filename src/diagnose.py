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
from train import make_splits, cap_benign 

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
def gene_only_hcm_dcm(train_df, test_df):

    # Defines the train and testing data
    tr = train_df[train_df["label"] != "Benign"]
    te = test_df[test_df["label"] != "Benign"]

    # Defines rate to be the mean of the amount of DCM cases (1) and HCM (0), calculates per gene. Should be high for TTN.
    rate = tr.assign(d= (tr["label"] == "DCM").astype(int)).groupby("gene")["d"].mean()
    overall = (tr["label"] == "DCM").mean()

    # Defines y to be 1 for DCM and 0 for HCM
    dcm_vs_hcm = (te["label"] == "DCM").astype(int)

    # If there aren't both DCM and HCM classes, return nan
    if y.nunique() < 2:
        return float("nan")


    return roc_auc_score(dcm_vs_hcm, te["gene"].map(rate).fillna(overall))

def run(dataset_path, seed = 42):

    
    df = add_consequence(pd.read_csv(dataset_path))

    # Runs the crosstabs (frequency counts of genes and consequence per label)
    print("\n--- CONSEQUENCE vs LABEL ---")
    print(pd.crosstab(df["consequence"], df["label"]))
    print("\n--- GENE vs LABEL ---")
    print(pd.crosstab(df["gene"], df["label"]))

    # Runs cap_benign and make_splits 
    capped = cap_benign(df, seed = seed)
    train_df, val_df, test_df, demo_df = make_splits(capped, seed = seed)

    print(f"Genetic-code rule (path vs benign) : {genetic_code_baseline(test_df):.3f}")
    print(f"Stop-codon rule   (DCM vs rest)    : {stop_codon_baseline(test_df):.3f}")
    print(f"Gene-name rule    (HCM vs DCM)     : {gene_only_hcm_dcm(train_df, test_df):.3f}")
