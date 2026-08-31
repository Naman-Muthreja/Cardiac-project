"""
Annotate.py fetches REVEL and CADD Binary AUC-ROC scores for comparison with my model.
"""

# Imports (sk_roc_auc_score is used to avoid confusion with roc_auc score from train.py)
import time
import numpy as np
import pandas as pd
import torch
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# Defines the latest CADD version, must use a string
CADD_VERSION = "GRCh38-v1.7"

# Uses a formatted string literal to fetch the chrom, pos, ref, alt of the right CADD version
def fetch_cadd_score(chrom, pos, ref, alt, cadd_version = CADD_VERSION):
    url = f"https://cadd.gs.washington.edu/api/v1.0/{cadd_version}/{chrom}:{pos}_{ref}_{alt}"

    # Uses requests to get the url, makes sure there is no error
    r = requests.get(url)
    if r.status_code != 200:
        return None

    # Json takes the API's response and makes it a list; the request specifies a specific variant, so record and extract its PHRED score
    cadd_returned_data = r.json()

    # If there is data returned, find the first
    if not cadd_returned_data:
        return None
    try:
        return float(cadd_returned_data[0]["PHRED"])

    # Returns None if there is a key error rather than crashing everthing
    except (KeyError, TypeError, ValueError):
        return None

# Uses a more complex formatted string literal to fetch the chrom, pos, ref, alt of the right REVEL version
def fetch_revel_score(chrom, pos, ref, alt, assembly = "hg38"):

    # Sets the ID in HGVS genomic notation, uses chr to refer to a chromosome (MyVariant.info needs it) and .g to refers to a genomic coordinate
    # > refers to a change in DNA Sequence (like from G to C)
    hgvs_id = f"chr{chrom}:g.{pos}{ref}>{alt}"

    # Uses adjacent string literal concatenation to query the GRCh38 assembly, making sure it only returns the REVEL data 
    url = (f"https://myvariant.info/v1/variant/{hgvs_id}"
           f"?assembly={assembly}&fields=dbnsfp.revel")

    r = requests.get(url)

    # Uses requests to get the url, makes sure there is no error
    if r.status_code != 200:
        return None

    # Gets the API's response
    revel_returned_data = r.json()

    # Checks if the returned data is a dictorionary, and gets the dbnsfp key
    dbnsfp = revel_returned_data.get("dbnsfp",{}) if isinstance(revel_returned_data, dict) else {}

    # Checks specifially for REVEL data from dbnsfp
    revel = dbnsfp.get("revel")
    if revel is None:
        return None    
    
    # REVEL can return either a dictionary or a list of dictionaries, this one is the simple dictionary case
    if isinstance(revel, dict):
        return revel.get("score")

    if isinstance(revel,list):

        # I define scores to be the “score” section of each entry that includes “score” and is a dictionary.
        scores = [entry["score"] for entry in revel if isinstance(entry,dict) and "score" in entry]

        # Returns the highest score, which has the most clincal significance, rather than the lowest score, which would bias towards
        # benignity
        return max(scores) if scores else None
    return None

def annotate_dataset(in_path, out_path, df = None):

    # If the dataframe is the default value, use that as the dataframe, else, use the new df value provided
    if df is None:
        df = pd.read_csv(in_path)
    
    # Starts off both cadd_scores and revel_scores as empty lists
    cadd_scores, revel_scores = [], []

    # This takes the values, asks CADD and REVEL “What score do you have for these specific chrom, pos, ref, and alt values?” and then 
    # appends those scores to empty lists called cadd_scores and revel_scores respectively. 
    for _, row in df.iterrows():

        # Takes the values, sleeps to avoid exhausting API
        chrom, pos, ref, alt = row["chrom"], int(row["pos"]), row["ref"], row["alt"]
        cadd = fetch_cadd_score(chrom, pos, ref, alt)
        time.sleep(0.2)
        revel = fetch_revel_score(chrom, pos, ref, alt)
        time.sleep(0.2)

        # Appends the scores
        cadd_scores.append(cadd)
        revel_scores.append(revel)

    # Converts the scores to dataframes, and then sets those dataframes to the out_path
    df["cadd_phred"] = cadd_scores
    df["revel_scores"]  = revel_scores
    
    df.to_csv(out_path, index= False)

    # Calculates the CADD coverage by taking the mean of the data with values and then multiplying it by 100. 
    print(f"CADD Coverage: {df['cadd_phred'].notna().mean() * 100:.1f}%")
    print(f"REVEL Coverage: {df['revel_scores'].notna().mean() * 100:.1f}%")

    # Coverage by class, filters for revel_scores and cadd_scores by each label (HCM, DCM, Benign)
    print("REVEL coverage by class:")
    print(df.groupby("label")["revel_scores"].apply(lambda s: f"{s.notna().mean() * 100:.1f}%"))

    print("CADD coverage by class:")
    print(df.groupby("label")["cadd_phred"].apply(lambda s: f"{s.notna().mean() * 100:.1f}%"))

def evaluate_baseline(train_df, test_df, feature_cols, label = "label", seed = 42):

    # Uses vectorized comparison to yield "1" for pathogenic and "0" for benign. Values is needed to convert into a NumPy array 
    # for logistic regression.
    y_train = (train_df[label] != "Benign").astype(int).to_numpy()
    y_test = (test_df[label] != "Benign").astype(int).values

    # Makes sure the data is a 2D dataframe, for logistic regression to actually work.
    X_train= train_df[feature_cols].to_numpy()
    X_test = test_df[feature_cols].to_numpy()

    # Sets the classifer, which uses logistic regression to train the baseline CADD and REVEL models, and then uses .fit() to compare the raw
    # data with the "answer key" and update weights.
    clf = LogisticRegression(max_iter = 1000, random_state= seed).fit(X_train, y_train)

    y_pred = clf.pred(X_test)

     # Prints the classification report and then the confusion matrix by comparing predictions with "answer key"

    print(confusion_matrix(y_test, y_pred))

    # Target names is just for clarity (0 is benign, 1 is pathogenic)
    print(classification_report(y_test, y_pred, target_names=["Benign", "Pathogenic"]))
    
    # Returns the model's Binary AUC-ROC score, returning the first index, which is the pathogenicity chance prediction
    return roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

   
    



    