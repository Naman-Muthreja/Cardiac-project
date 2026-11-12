"""
addons.py includes 3 addons that are used as additional benchmarks for my model. 
1. TTN vs the curator's rule. Are TTN stop codons ranked better than simple exon usage?
2. Hard variants. How does the model do on hard variants that were VUS or conflicting in ClinVar in January 2020 or January 2023 
   and have a clear label now? And how does this compare to to the other tools?
3. Dated VUS recheck. The VUS scores committed to GitHub  are compared with ClinVar today.
"""
import os
import time
import numpy as np
import pandas as pd
import requests

import Revel_cadd_alphamissense_bench as bench

HERE = os.path.dirname(os.path.abspath(__file__))
EXTERNAL = os.path.join(HERE, "..", "data", "external")
ARCHIVE = os.path.join(EXTERNAL, "clinvar_archive")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


# Returns the unique ID for each variant, including chromosome number, position, reference base, and alternate base.
def keys_of(df):
    return df["chrom"].astype(str) + ":" + df["pos"].astype(int).astype(str) + df["ref"] + ">" + df["alt"]

# Assigns the cross-validation benignity and pathogenic prediction scores for each variant in a gene-consequence group.
def cv_rows(cv_preds):
    return cv_preds.assign(model = 1 - cv_preds["p_benign"], cell = cv_preds["gene"] + "/" + cv_preds["consequence"])

