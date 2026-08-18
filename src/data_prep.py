# data_prep.py filters for the specific data needed and exports it to one dataframe

# #imports, includes re(regular expression lib), time(pause between API calls),
#pandas for viewing data, and requests to download data
import re
import time
import pandas as pd
import requests

# Explicit columns actually needed(avoids RAM exhaustion)
ClinVar_USECOLS = [
    "#AlleleID", "AlleleID", "Type", "GeneSymbol", "ClinicalSignificance",
    "Name", "Assembly", "Chromosome", "PositionVCF",
    "ReferenceAlleleVCF", "AlternateAlleleVCF", "ReviewStatus",
]

# Length of DNA sequences, includes center base (mutation site) and 100 bases to the left, 100 to the right
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
# runs quietly in the background without uneccesary print calls(verbose = false)
# Chunksize is present because reading the whole variant_summary file in one go 
# can cause crashes
def load_ClinVar(path, verbose = False, chunksize = 200_000):

    header = pd.read_csv(path, sep = "\t", nrows = 0)

    # Only keeps the necessary data
    keep = [c for c in ClinVar_USECOLS if c in header]

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

        # If the chunk survived, add it to the kept list
        if len(chunk) > 0:
            kept_chunks.append(chunk)

    # Combines all the chunks into one DataFrame, with the row index numbers reset(ignore index)
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
def is_nonsense_snv(ClinVar_name):
    if not isinstance(ClinVar_name, str):
        return False
    return re.search(r"p\.[A-Za-z]{3}\d+Ter", ClinVar_name) is not None

# This sees whether a variant is pathogenic or benign using keywords in ClinVar, then
# returns the appropriate name. For example, a benign variant returns benign.
def label_row(gene, clinsig, name ):

    sig = clinsig.lower()
    is_pathogenic = "pathogenic" in sig and "conflicting" not in sig
    is_benign = "benign" in sig and "conflicting" not in sig 

    if is_pathogenic:
        
        if gene == "TTN":
            # I don't need to do this for MYH7, or MYBPC3, because missense mutations can be 
            # pathogenic in those genes, but TTN missense mutations are healthy, so I
            # filter TTN for only nonsense substitutions. If is_nonsense_snv is true, DCM is outputted
            return "DCM" if is_nonsense_snv(name) else None
        
        # If pathogenic but not TTN, return what gene caused it to be pathogenic (missense mutations)
        return GENE_TO_CLASS[gene]
    if is_benign:
        return "Benign"
    return None 

def fetch_sequence(chrom, pos, timeout=8, max_retries=2):

    # Makes sure that the position of the variant is exactly the half way point
    start = pos - HALF
    end = pos + HALF

    # Uses the Ensembl REST API to download coordinate windows for chromosomes
    url = (
        f"https://rest.ensembl.org/sequence/region/human/"
        f"{chrom}:{start}..{end}?content-type=text/plain"
    )
    r = None
    # Try 2 times to acess the ENSEMBL REST API, with a timeout of 15 seconds
    for attempt in range(max_retries):
        try: 
             r = requests.get(url, timeout=timeout)
        # If the ENSEMBL REST API does give a cooldown/error, try again after a few seconds
        # Give up on the variant if ENSEMBL REST API keeps giving cooldowns
        except (requests.exceptions.RequestException, OSError):
            time.sleep(1+attempt)
            continue

    # If the status code is specifically too many requests, wait for a few seconds
        if r.status_code == 429:
            time.sleep(2+attempt*2)
            continue 

        break 

    # If something like a 404 error comes up, return None.
    if r is None or r.status_code != 200:
        return None
    
        # Returns the ACTG string from the Ensembl REST API, formatted properly.
    else: 
        return r.text.strip().upper()
    
# Checks the center(the mutation site) and returns the DNA strand for the model to analyze,
# given that ENSEMBL REST API matches ClinVar. It's like a sanity check.
def apply_variant(seq, ref, alt):
    center = HALF
    if seq[center] != ref:
        return None 
    else: 
        return seq[:center] + alt + seq[center+1:] 

# Defines key information, like pos, chrom, ref, alt, etc, and then outputs it. It uses many
# previously defined functions to accomplish this (Ex: fetch_sequence to get the DNA sequence).
def build_ClinVar_dataset(ClinVar_path, out_path, limit=None, old_frac=0.50, new_frac=1.0, seed=42):

    df = load_ClinVar(ClinVar_path)

    # Because seeds include reproducible data, this checks the old seed, sees what should have
    # been saved, and skips it, so that it only adds the 50% not saved in the first pass
    # into the dataset.
    new_sample = df.sample(frac=new_frac, random_state=seed)
    if old_frac > 0:
        old_sample = df.sample(frac=old_frac, random_state=seed)
        df = new_sample[~new_sample.index.isin(old_sample.index)]
    else:
        df = new_sample
    rows = []

    # This for loop iterates every row (rather than column headers like normal)  using iterrows, but the _ makes it so that 
    # it so that index numbers are discarded. Also adds a progress tracker
    for i, (_, row) in enumerate(df.iterrows()):
        if limit is not None and i >= limit:
            break

        # Displays every 100 rows
        if i % 100 == 0:
            print(f"...processed {i}/{len(df)} rows, {len(rows)} kept so far")

        # Creates an autosaving mechanism so all progress isn't loss if downloading stops.
        if i % 500 == 0 and i > 0:
            pd.DataFrame(rows).to_csv(out_path + ".partial.csv", index=False)
            print(f"...checkpoint saved at row {i} ({len(rows)} kept so far)")

        # Calls the label_row function from earlier, and assigns the clinical significance
        # (e.g. pathogenic) to "label", as a string for the model. Remember, because I'm putting the three
        # parameters wrapped inside label_row, row["Gene Symbol"] becomes the gene parameter; 
        # str(row(["Clinical Significance"]) becomes clinsig, and the name is str(row["Name"]).
        label = label_row(row["GeneSymbol"], str(row["ClinicalSignificance"]), str(row["Name"]))

        # If the label does not have the needed data, it goes to the next iteration, since this
        # one is not needed.
        if label is None:
            continue 

        # Takes the chromosome identifier, with string format
        chrom = str(row["Chromosome"])

        # Tries to cast the Position into an integer format. 
        # Position VCF is used for ClinVar's column names rather than Position, which is 
        # used for gnomAD.
        try: 
            pos = int(row["PositionVCF"])
        # If an error is thrown, iterate to the next row, since this one is not needed.
        except(ValueError, TypeError):
            continue

        # Initializes the ref and alt variables to their respective positions
        ref = str(row["ReferenceAlleleVCF"])
        alt = str(row["AlternateAlleleVCF"])

        # Makes sure that only SNVs and not insertions are used
        VALID_BASES = {"A", "C", "G", "T"}
        if ref not in VALID_BASES or alt not in VALID_BASES:
            continue

        # Fetch the DNA from fetch_sequence
        seq = fetch_sequence(chrom, pos)

        # Sleep for 1/10 second to prevent overwhelming ENSEMBL
        time.sleep(0.1)

        # Makes sure that a value is actually outputted, otherwise moves on
        if seq is None or len(seq) != WINDOW:
            continue 

       # Fetches the variant position, defines the mutant/the point of mutation
        mutant = apply_variant(seq, ref, alt)

        # Makes sure that a value is actually outputted, otherwise moves on
        if mutant is None:
            continue

        # Appends all the data needed about the variant, which will be later merged with gnomAD
        rows.append(
            {
                "sequence": mutant, "label": label, "gene": row["GeneSymbol"],
                "pos": pos, "chrom": chrom, "ref": ref, "alt": alt,
                "name": row["Name"],
            }
        )

    # Creates a data saving mechanism so that I can run the file in two 10-hour parts instead
    # of one 20-hour data extraction process. Makes sure to save the dataset without indexes being
    # scattered, using ignore_index = True. 
    out = pd.DataFrame(rows)
    if old_frac > 0:
        old_out = pd.read_csv(out_path)
        out = pd.concat([old_out, out], ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} sequences to {out_path}")
    print(out["label"].value_counts())
    return out

# Makes a function similar to build_ClinVar_dataset used for ClinVar data, but with benign varaiants
# from gnomAD to balance data. The faf_threshold (Filtering allele frequency) of 0.001  means
# that the mutation is relatively common (1 in 1k), indicating benignity.
def build_gnomAD_benign(gnomAD_csv_path, gene, faf_threshold  = 0.001):

    df = pd.read_csv(gnomAD_csv_path)
    # If a gene passed quality control (qc) in the exome or whole genome, it is okay to move on.
    # A NaN value gets filled with an empty string.

    passes_qc = (df["Filters - exomes"].fillna("").eq("PASS") | df["Filters - genomes"].fillna("").eq("PASS"))

    # Only rows that evaluate "PASS" as True will move on.
    df = df[passes_qc]

    # If the Filtering allele frequency is greater than 0.001, keep the variant (benign needed only)
    # Fills missing values with 0(which won't work) with fillna. 
    df = df[df["GroupMax FAF frequency"].fillna(0) > faf_threshold]

    # This code is very similar to build_ClinVar_dataset, and is still needed, for filtering for 
    # only needed data.
    rows = []
    # Makes the for loop, with iterrows for index, and a progress tracker
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100 == 0:
            print(f"...processed {i}/{len(df)} rows, {len(rows)} kept so far")
        # Creates an autosaving mechanism so all progress isn't loss if downloading stops
        if i % 500 == 0 and i > 0:
            pd.DataFrame(rows).to_csv(f"gnomad_{gene}_partial.csv", index=False)
            print(f"...{gene} checkpoint saved at row {i} ({len(rows)} kept so far)")               

        # Intializes chromosome identifier, and maps the Position to an integer format
        # Unless an error occurs, where it goes to the next row.
        chrom = str(row["Chromosome"])
        try:
            pos = int(row["Position"])
        except (ValueError, TypeError):
            continue

        # Intializes ref and alt to their respective positions
        ref = str(row["Reference"])
        alt = str(row["Alternate"])

        # Filters for only SNVs, skips things like "CT" in one space
        VALID_BASES = {"A", "C", "G", "T"} 
        if ref not in VALID_BASES or alt not in VALID_BASES:
            continue

        # Fetches DNA Sequence
        seq = fetch_sequence(chrom, pos)
        time.sleep(0.1)
        if seq is None or len(seq) != WINDOW:
            continue

        # Fetches the variant position, defines the mutant/the point of mutation
        mutant = apply_variant(seq, ref, alt)
        if mutant is None:
            continue

        # Outputs the information about the variant
        # Note: gnomAD does not have clinical labeling, but I assume benignity, due
        # to the filtering I conducted.

        rows.append(
            {
                "sequence": mutant, "label": "Benign", "gene": gene,
                "pos": pos, "chrom": chrom, "ref": ref, "alt": alt,
                "name": "",
            }
        )
    # returns the df
    return pd.DataFrame(rows)

# Builds the final dataset combining both the gnomAD and ClinVar outputs.
def build_full_dataset(ClinVar_path, gnomAD_csv_paths, out_path, faf_threshold = 0.001):

    # Stores the resulting Dataframe from build_ClinVar_dataset to ClinVar_rows
    ClinVar_rows = build_ClinVar_dataset(ClinVar_path, out_path)

    # Stores the resulting DataFrame from build_gnomAD_benign to gnomAD_frames,
    # for each of the gene-path pairs (MYH7, MYBPC3, and TTN). 
    gnomAD_frames = [
        build_gnomAD_benign(path, gene, faf_threshold)
        for gene, path in gnomAD_csv_paths.items()
    ]

    # This finalizes a df for the gnomAD data, by combining data for each gene's CSV file
    # IgnoreIndex makes sure to reindex the variants instead of keeping the old indexes.
    # Also creates an empty df instead of crashing.

    gnomAD_rows = pd.concat(gnomAD_frames, ignore_index= True ) if gnomAD_frames else pd.DataFrame()

    # Merging all the data into one final df

    combined = pd.concat([ClinVar_rows, gnomAD_rows], ignore_index= True)
    combined.to_csv(out_path, index = False)

    # Prints how many times each label appears
    print(f"Combined Dataset: {len(combined)} total sequences")
    print(combined["label"].value_counts())

    return combined