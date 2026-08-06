#imports, includes re(reguular expression lib), time(pause between API calls),
#pandas for viewing data, and requests to download data
import re
import time
import pandas as pd
import requests

# Explicit columns actually needed(avoids RAM exhaustion)
CLINVAR_USECOLS = [
    "#AlleleID", "AlleleID", "Type", "GeneSymbol", "ClinicalSignificance",
    "Name", "Assembly", "Chromosome", "PositionVCF",
    "ReferenceAlleleVCF", "AlternateAlleleVCF", "ReviewStatus",
]

# Length of DNA sequences, includes center base and 100 bases to the left, 100 to the right
WINDOW = 201

# // forces the 100.5 to floor down to 100
HALF = WINDOW // 2

GENE_TO_CLASS = {
    "MYH7": "HCM",
    "MYBPC3": "HCM",
    "TTN": "DCM",
}

# If no assertation is provided, it isn't good data
LOW_CONFIDENCE_REVIEW = {"no assertion criteria provided", "no assertion provided"}

# Filters ClinVar for the needed data, and Verbose = false means that function 
# runs quietly in the background without unneccesary print calls(verbose = false)
# Chunksize is present because reading the whole variant_summary file in one go 
# can cause crashes
def load_clinvar(path, verbose = False, chunksize = 200_000):
    header = pd.read_csv(path, sep = "\t", nrows = 0)
    # Only keeps the necessary data
    keep = [c for c in CLINVAR_USECOLS if c in header]
    if verbose:
        print("Columns actually used:", keep)

    # Initializing an empty list later to be used as a df of filtered data
    kept_chunks = []

    for chunk in pd.read_csv(
        path, sep="\t", usecols=keep, chunksize=chunksize, low_memory=False):
        # Apply all filters to the current chunk only
        # Renaming #AlleleID is done to prevent ambiguity, because sometimes ClinVar uses
        # the name "AlleleID", and other times it is "#AlleleID"
        chunk = chunk.rename(columns={"#AlleleID": "AlleleID"})
        chunk = chunk[chunk["Assembly"] == "GRCh38"]
        chunk = chunk[chunk["Type"] == "single nucleotide variant"]
        chunk = chunk[chunk["GeneSymbol"].isin(["MYH7", "MYBPC3", "TTN"])]
        # If the chunk surrived, add it to the kept list
        if len(chunk) > 0:
            kept_chunks.append(chunk)
    # Combines all the chunks into one DataFrame, with the row numbers reset(ignore index)
    if kept_chunks:
        df = pd.concat(kept_chunks, ignore_index= True)
    else:
        # Empty df, if nothing matches
        df = pd.DataFrame(columns = keep)

    # Filtering out low confidence variants or duplicates
    df = df[~df["ReviewStatus"].str.lower().isin(LOW_CONFIDENCE_REVIEW)]
    df = df.drop_duplicates(subset=["AlleleID"])
    return df


# Returns false for a nonstring input, and detects ClinVar's stop codon notation, to search for DCM
def is_nonsense_snv(clinvar_name):
    if not isinstance(clinvar_name, str):
        return False
    return re.search(r"p\.[A-Za-z]{3}\d+Ter", clinvar_name) is not None

# This sees whether a variant is pathogenic or benign using keywords in ClinVar, then
# returns the appropriate name. For example, a benign variant returns benign.
def label_row(gene, clinsig, name ):
    sig = clinsig.lower()
    is_pathogenic = "pathogenic" in sig and "conflicting" not in sig
    is_benign = "benign" in sig and "conflicting" not in sig 
    if is_pathogenic:
        if gene == "TTN":
            # We don't need to do this for MYH7, or MYBPC3, because missense mutations can be 
            # pathogenic in those genes, but TTN missense mutations are healthy, so we
            # filter TTN for only nonsense substitutions. If is_nonsense_snv is true, DCM is outputted
            return "DCM" if is_nonsense_snv(name) else None
        # If pathogenic but not TTN, return what gene caused it to be pathogenic (missense mutations)
        return GENE_TO_CLASS[gene]
    if is_benign:
        return "Benign"
    return None 

def fetch_sequence(chrom, pos):
    start = pos - HALF
    end = pos + HALF
    # Uses the Ensembl REST API to download coordinate windows for chromosomes
    url = (
        f"https://rest.ensembl.org/sequence/region/human/"
        f"{chrom}:{start}..{end}?content-type=text/plain"
    )
    # Stores the server's response into a variable called 'r'
    r =  requests.get(url)
    # If something like a 404 error comes up, return None
    if r.status_code != 200:
        return None
        # Returns the ACTG string from the Ensembl REST API, formatted properly.
    else: 
        return r.text.strip.upper()
# Checks the center(the mutation site) and returns the DNA strand for the model to analyze,
# given that ENSEMBL REST API matches ClinVar
def apply_variant(seq, ref, alt):
    center = HALF
    if seq[center] != ref:
        return None 
    else: 
        return seq[:center] + alt + seq[center+1:] 
    
def build_dataset(clinvar_path, out_path):
    df = load_clinvar(clinvar_path)
    rows = []
    

    