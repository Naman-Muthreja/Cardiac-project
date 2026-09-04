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
from train import cap_benign
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Defines the latest CADD version, must use a string
CADD_VERSION = "GRCh38-v1.7"

# Uses a formatted string literal to fetch the chrom, pos, ref, alt of the right CADD version
def fetch_cadd_score(chrom, pos, ref, alt, cadd_version = CADD_VERSION):
    url = f"https://cadd.gs.washington.edu/api/v1.0/{cadd_version}/{chrom}:{pos}_{ref}_{alt}"

    # Uses requests to get the url, makes sure there is no error
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
    except (requests.exceptions.RequestException, OSError):
        return None

    # Json takes the API's response and makes it a list; the request specifies a specific variant, so record and extract its PHRED score
    cadd_returned_data = r.json()

    # Returns the PHRED portion needed
    for record in cadd_returned_data:
        try:
            if isinstance(record, dict) and record.get("Alt") == alt:
                return float(record["PHRED"])
        except (KeyError, TypeError, ValueError):
            return None

    # If nothing matches, return None (just like if there is a key error, type error, or value error)
    return None

# Fetches the REVEL score
def fetch_revel_score(chrom, pos, ref, alt, assembly = "hg38"):

    # Sets the ID in HGVS genomic notation, uses chr to refer to a chromosome (MyVariant.info needs it) and .g to refers to a genomic coordinate
    # > refers to a change in DNA Sequence (like from G to C)
    hgvs_id = f"chr{chrom}:g.{pos}{ref}>{alt}"

    # Uses adjacent string literal concatenation to query the GRCh38 assembly, making sure it only returns the REVEL data 
    url = (f"https://myvariant.info/v1/variant/{hgvs_id}"
           f"?assembly={assembly}&fields=dbnsfp.revel")

    r = requests.get(url)

    # Uses requests to get the url, makes sure there is no requests error
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
    except (requests.exceptions.RequestException, OSError):
        return None

    # Gets the API's response
    revel_returned_data = r.json()

    for record in revel_returned_data:
            try:
                if isinstance(record, dict) and record.get("Alt") == alt:
                    return float(record["PHRED"])
            except (KeyError, TypeError, ValueError):
                return None
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
        time.sleep(0.15)
        revel = fetch_revel_score(chrom, pos, ref, alt)
        time.sleep(0.15)

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
    y_test = (test_df[label] != "Benign").astype(int).to_numpy()

    # Makes sure the data is a 2D dataframe, for logistic regression to actually work.
    X_train= train_df[feature_cols].to_numpy()
    X_test = test_df[feature_cols].to_numpy()

    # Sets the classifer, which uses logistic regression to train the baseline CADD and REVEL models, and then uses .fit() to compare the raw
    # data with the "answer key" and update weights.
    clf = LogisticRegression(max_iter = 1000, random_state= seed).fit(X_train, y_train)

    y_pred = clf.predict(X_test)

     # Prints the classification report and then the confusion matrix by comparing the predictions with the "answer key"

    print(confusion_matrix(y_test, y_pred))

    # Target names is just for clarity (0 is benign, 1 is pathogenic)
    print(classification_report(y_test, y_pred, target_names=["Benign", "Pathogenic"]))
    
    # Returns the model's Binary AUC-ROC score, returning the first index, which is the pathogenicity chance prediction
    return roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

def split_by_test_variants(df, test_df):

    # Drops duplicates that may be inside of the dataset
    df = df.drop_duplicates(subset = ["chrom", "pos", "ref", "alt"]).copy()

    # This grabs the 4 aforementioned columns from the test table, combines those 4 values into a tuple for each variant, 
    # and then puts all of those tuples into a lookup book using set(). This is useful because it functions as a variant key.
    test_keys = set(map(tuple, test_df[["chrom","pos","ref","alt"]].to_numpy()))

    # Compares the CADD/REVEL dataset to the test_keys dataset also used to test my 1D CNN. .apply() has the first parameter as the function
    # being applied, and the second parameter is where it is being applied. In this case, it is converting each row to a tuple.
    is_test = df[["chrom", "pos", "ref", "alt"]].apply(tuple, axis = 1).isin(test_keys)
   
    # Returns a copy of everything not in the CNN's test dataframe (for training), and everything that IS in the CNN's test dataframe (for testing)
    # ~ flips the truth values
    return df[~is_test].copy(), df[is_test].copy()

# So that I do not overwhelm CADD and REVEL APIs with bulk lookups, # CADD asks not to be used for bulk lookups, 
# so I annotate only what the benchmark needs, which is all 716 of the CNN's test variants to score on, plus 1500 others to fit
# the CADD/REVEL logistic regression on.

def build_annotation_subset(df, test_df, n_fit=1500, seed=42):

    # Caps the amount of benign variants inside of the dataset
    capped = cap_benign(df, max_benign=None, seed=seed)

    # Splits the capped dataset into the CNN's 716 test variants (to grade the
    # baseline on) and everything else (to fit the baseline on)
    train_pool, test_pool = split_by_test_variants(capped, test_df) 

    # Takes the minimum of n_fit and train_pool, just incase n_fit is not exactly 1500
    n_fit = min(n_fit, len(train_pool))

    frac = n_fit / len(train_pool) 
    print(f"Kept {frac * 100:.1f} % of train_pool variants, which is {len(train_pool)}")

    # Makes an empty list later to be added onto
    pieces = []

    # for loop that basically says "for every label, find the number of rows using formula len(g) * frac, and then append it to 
    # an empty list called pieces."
    for _, g in train_pool.groupby("label"):

        n_rows = max(1, int(round(frac * len(g))))
        pieces.append(g.sample(n=n_rows, random_state=seed))

    train_rows = pd.concat(pieces)

    # Stacks the 716 testing rows on top of the 1500 train rows
    subset = pd.concat([test_pool, train_rows], ignore_index=True)

    print(f"Annotation subset: {len(subset)} rows "
          f"({len(test_pool)} to test on + {len(train_rows)} to train with)")

    return subset

# Shows how well REVEL and CADD do by themselves.
def revel_cadd_benchmark(df, test_df, label = "label", seed = 42):

    # Splits by the test variants before doing the benchmark
    train_df, test_df = split_by_test_variants(df, test_df)

    # Creates an empty dict, seperating by name rather than number, and then defines the CADD alone and REVEL alone
    results = {}
    feature_sets = [("CADD Alone", ["cadd_phred"] ), ("REVEL alone (only missense)", ["revel_scores"])]

    for name, cols in feature_sets:

        # Drop the missing values and only look at the specific columns without the missing values.
        train_ready = train_df.dropna(subset = cols)
        test_ready = test_df.dropna(subset=cols)

        # If there are a too few amount of variants, don't follow through the test.
        if len(train_ready) <= 20 or len(test_ready) <= 20:
            print(f"train_ready amount of rows is {len(train_ready)}, and test ready amount of rows is {len(test_ready)}")
            print("This means that there are too few variants.")
            continue

        # If there aren't two unique classes, having AUC-ROC is pointless. nunique checks how many unique truth values there are.
        # In this case, two seperate classes would yield "True" for the pathogenic class, and "False" for "Benign", meaning there are 2
        # truth values
        if(test_ready[label] != "Benign").nunique() < 2:
            print(f"There are not enough unique classes, ")
            continue

        # Print the amount of training variants and testing variants
        print(f"\n====={name} | training = {len(train_ready)} | testing = {len(test_ready)} ======")

        # Evaluate_baseline (previously defined) is called, yielding several data metrics.
        results[name] = evaluate_baseline(train_ready, test_ready, cols, label, seed)

    return results







    