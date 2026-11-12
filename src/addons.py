"""
addons.py includes 3 addons that are used as additional benchmarks for my model. 
1. TTN vs the curator's rule. Are TTN stop codons ranked better than simple exon usage?
2. Hard variants. How does the model do on hard variants that were VUS or conflicting in ClinVar in January 2020 or January 2023 
   and have a clear label now? And how does this compare to to the other tools?
3. Dated VUS recheck. The VUS scores committed to GitHub on November 5, 2026 were compared with ClinVar on _____ to check how
   the model did on potential reclassifications.
"""
import os
import time
import numpy as np
import pandas as pd
import requests

import Revel_cadd_alphamissense_bench as bench

# Pathways, with the current location, the data being 
HERE = os.path.dirname(os.path.abspath(__file__))
EXTERNAL = os.path.join(HERE, "..", "data", "external")
ARCHIVE = os.path.join(EXTERNAL, "clinvar_archive")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


# Returns the unique ID for each variant, including chromosome number, position, reference base, and alternate base.
def keys_of(df):
    return df["chrom"].astype(str) + ":" + df["pos"].astype(int).astype(str) + df["ref"] + ">" + df["alt"]

# Assigns the cross-validation benignity and pathogenic prediction scores for each variant in a gene-consequence group.\
# Returns 1 minus the probability of being benign because the saved cv explanations show the chance of benigity.
def cv_rows(cv_preds):
    return cv_preds.assign(model = 1 - cv_preds["p_benign"], cell = cv_preds["gene"] + "/" + cv_preds["consequence"])

# Compares the model to the PSI scores by itself, using AUC-ROC scores to create a verdict
def compare(rows, score_a, score_b, title, by_cell = False, n_boot = 1000):

     # Calculates the AUC-ROC score for each variant group consisting of the same gene and consequence, using fold_auc().
     a, b = bench.fold_auc(rows, score_a, by_cell = by_cell), bench.fold_auc(rows, score_b, by_cell = by_cell)

     # 95% confidence interval through paired_bootstrap(), and creates a verdict to determine if the model did better or 
     # worse than the PSI scores 
     low, high = bench.paired_bootstrap(rows, score_a, score_b, by_cell = by_cell, n_boot = n_boot)
     verdict = "model is better" if low > 0 else ("model is worse" if high < 0 else "model ties")

     # Finds the number of pathogenic and benign variants, dropping duplicate variants, and taking the rows.
     n_path = rows.loc[rows["label"] != "Benign", ["chrom", "pos", "ref", "alt"]].drop_duplicates().shape[0]
     n_ben = rows.loc[rows["label"] == "Benign", ["chrom", "pos", "ref", "alt"]].drop_duplicates().shape[0]

     print(f"{title}:{score_a} {a:.3f} vs {score_b}{b:.3f}, difference {a-b:+.3f} [{low:+.3f}, {high:+.3f}], meaning that the {verdict}.")
     print(f"There was {n_path} pathogenic variants and {n_ben} benign variants.")

     # Returns important data
     return {"comparison": title, "a": score_a, "auc_a": a, "b": score_b, "auc_b": b, "diff": a - b,
            "ci_low": low, "ci_high": high, "verdict": verdict, "pathogenic": n_path, "benign": n_ben}

def ttn_vs_curator_rule(matched, setup, cv_preds, n_boot = 1000):

    # Nonsense TTN variants specifically to compare my model with PSI on these TTN stop codons
    rows = cv_rows(cv_preds)
    rows = rows[rows["cell"] == "TTN/nonsense"].copy()

    # Gets the 49 raw feature values
    raw = setup["raw"](matched)

    # Uses PSI scores and checks if A-band is the TTN region (most stop codons are there)
    lookup = pd.DataFrame({"key": keys_of(matched), "psi": raw["ttn_psi_dcm"].to_numpy(), "a_band": raw["ttn_A-band"].to_numpy()})

    # Ensures there is one lookup per row/variant
    rows = rows.assign(key = keys_of(rows)).merge(lookup.drop_duplicates("key"), on = "key", how = "left")
    assert rows["psi"].notna().all(), "a TTN stop codon is missing its exon usage"

    # Table that will be compared with my model, "1" if PSI score >= 0.9 and "1" if in a_band
    rows["rule"] = ((rows["psi"] >= 0.9) | (rows["a_band"] > 0)).astype(float)


    out = [compare(rows, "model", "psi", "TTN stop codons, model vs exon usage (PSI)", n_boot = n_boot),
           compare(rows, "model", "rule", "TTN stop codons, model vs PSI (>0.9) and A-band location", n_boot = n_boot)]
    return pd.DataFrame(out)