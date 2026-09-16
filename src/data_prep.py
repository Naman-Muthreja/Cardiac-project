"""
data_prep.py filters for the specific data needed and exports it to one dataframe
"""
# #imports, includes re(regular expression lib), time(pause between API calls),
#pandas for viewing data, and requests to download data
import re
import time
from consequence import parse_consequence, VEP_TO_CONSEQUENCE
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

def fetch_sequence(chrom, pos, timeout=12, max_retries=3):

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
            r = None
            continue

    # If the status code is specifically too many requests, wait for a few seconds
        if r.status_code == 429:
            time.sleep(2+attempt*2)
            continue 

        # Terminates the for loop after running these checks
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
def build_ClinVar_dataset(ClinVar_path, out_path, limit=None, old_frac=0.0, new_frac=1.0, seed=42):

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

        # Appends all the data needed about the variant, which will be later merged with gnomAD. 
        rows.append(
            {
                "sequence": mutant, "label": label, "gene": row["GeneSymbol"],
                "pos": pos, "consequence": parse_consequence(str(row["Name"])), "chrom": chrom, "ref": ref, "alt": alt,
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

    # Takes the length of the output before duplicate variants are taken out
    before = len(out)

    # Drops the duplicate variants
    new_out = out.drop_duplicates(subset=["chrom","pos","ref","alt"], keep="first")

    # Prints the amount of rows dropped
    print(f"Dropped {before - len(new_out)} duplicate (chrom, pos, ref, alt) rows")

    # Makes the output (with dropped duplicates) a csv.
    new_out.to_csv(out_path, index=False)

    print(f"Saved {len(new_out)} sequences to {out_path}")
    print(new_out["label"].value_counts())
    return new_out

def consequence_targets(ClinVar_df, gene, ratio = 1.0):

    # Checks for a true/false match per row for each gene. Takes the data from the built ClinVar dataframe, including the same properties,
    # like gene, label, consequence, etc.
    g = ClinVar_df[ClinVar_df["gene"] == gene]

    # Defines pathogenic and benign using label from g, operating on the built g df.
    pathogenic = g[g["label"] != "Benign"]
    benign = g[g["label"] == "Benign"]

    targets = {}

    # Finds the amount of pathogenic variants per consequence, later used with a ratio to find the amount of benign variants needed per consequence
    for cons, sub in pathogenic.groupby("consequence"):

        # Counts the variant amounts wanted (the total of sub) and the amount I already have, to see how much gnomAD benign variants is needed
        want = int(round(ratio *len(sub)))
        have = len(benign[benign["consequence"] == cons])

        # The amount of variants needed is the amount wanted - the amount I already have
        need = max(0, want-have)
        if need > 0:
            targets[cons] = need
    return targets
    
# Makes a function similar to build_ClinVar_dataset used for ClinVar data, but with benign varaiants
# from gnomAD to balance data. Targets is used to provide how much of each variant is needed.
def build_gnomAD_benign(gnomAD_csv_path, gene, targets, exclude_keys = None, faf_threshold = 1e-6, seed = 42, limit = None):

    df = pd.read_csv(gnomAD_csv_path, low_memory=False)
    # If a gene passed quality control (qc) in the exome or whole genome, it is okay to move on.
    # A NaN value gets filled with an empty string.
    passes_qc = (df["Filters - exomes"].fillna("").eq("PASS") | df["Filters - genomes"].fillna("").eq("PASS"))

    # Only rows that evaluate "PASS" as True will move on.
    df = df[passes_qc]

    # Finds important information from ClinVar, drops gnomAD variants that completely match.
    if exclude_keys is not None:
        before = len(df)

        # Zip converts different datatypes (like strings and values) into a single list of tuples
        keys = zip(
            df["Chromosome"].astype(str),
             # The position must be numerical, so I use pandas to convert to an integer, and fill out all errors/crashes with -1 
             # instead of NaN, which will still not work.
            pd.to_numeric(df["Position"], errors="coerce").fillna(-1).astype(int),
            df["Reference"].astype(str),
            df["Alternate"].astype(str),
        )
        df = df[[k not in exclude_keys for k in keys]]
        print(f"[{gene}] dropped {before - len(df)} rows already in ClinVar")

    
    # If the Filtering allele frequency is greater than 1e-6, keep the variant 
    # Fills missing values with 0(which won't work) with fillna. Also converts text to numbers
    df["Allele Frequency"] = pd.to_numeric(df["Allele Frequency"], errors="coerce")
    df = df[df["Allele Frequency"].fillna(0) > faf_threshold]
    print(f"[{gene}] {len(df)} after AF > {faf_threshold}")

    # Applies the dictionary lookup VEP_TO_CONSEQUENCE to each "VEP Annotation" in the dataframe, drops na, prints total
    df["consequence"] = df["VEP Annotation"].map(VEP_TO_CONSEQUENCE)
    df = df.dropna(subset=["consequence"])
    print(df["consequence"].value_counts().to_string())

    picked = []
    # Makes a for loop that iterates for every consequence and the amount of that consequence that I actually want, 
    # using targets as a dictionary. Because of .items(), cons is a string, representing the consequence, like "missense", while
    # want is the amount of variants wanted in that class.
    for cons, want in targets.items():

            # For each consequence in the previously defined df from the last function, check if the consequence column matches one of the cons
            sub = df[df["consequence"] == cons]

            #  I define “take” to be the minimum of the amount of variants that I want for a specific consequence, and the 
            # total number of variants in that consequence. 
            take = min(want, len(sub))

            # If the amount of variants taken is lower than wanted, give a print message
            if take < want:
                print(f"Wanted {want}, but have {len(sub)} for {gene}/{cons}")

            if take > 0:
                picked.append(sub.sample(n=take, random_state = seed))

    # I return an empty Pandas dataframe rather than "None" so later functions dont crash.
    if not picked:
         print("0 variants for 'picked'")
         return pd.DataFrame()
                 
    df = pd.concat(picked, ignore_index=True)

    # If there is a limit used for verification, mention it.
    if limit is not None:
         df = df.head(limit)
    


    # This code is very similar to build_ClinVar_dataset, and is still needed, for filtering for only the needed data
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
                "pos": pos, "consequence": row["consequence"], "af": float(row["Allele Frequency"]), "chrom": chrom, "ref": ref, "ref_sequence": seq, "alt": alt,
                "name": "",
            }
        )
    # returns the df
    return pd.DataFrame(rows)

# Builds the final dataset combining both the gnomAD and ClinVar outputs. Note: the second time I used this function, to make the data for benign variants
# more spread out, I define one of the parameters as ClinVar_dataset_path instead of ClinVar_path, to prevent rebuilding the same ClinVar data again.
# Builds the final dataset combining both the gnomAD and ClinVar outputs.
def build_full_dataset(clinvar_dataset_path, clinvar_raw_path, gnomAD_csv_paths, out_path, faf_threshold = 1e-6, ratio = 1.0, seed = 42, limit = None):

    # Reads the rows from the built dataset from calling the function last time
    ClinVar_rows = pd.read_csv(clinvar_dataset_path)

    # Makes the source of ClinVar_rows "ClinVar", that way, I can look at the final dataset to see if a variant originates from ClinVar.
    ClinVar_rows["source"] = "ClinVar"
    print(f"Loaded {len(ClinVar_rows)} ClinVar rows from {clinvar_dataset_path}")

    # If the consequence column is impresent in the clinvar columns, apply the parse consequence function from consequence.py
    if "consequence" not in ClinVar_rows.columns:
        ClinVar_rows["consequence"] = ClinVar_rows["name"].apply(parse_consequence)

    clinvar_all = load_ClinVar(clinvar_raw_path)

    # Very similar to "keys" in build. Used later when calling build_gnomAD_benign
    clinvar_keys = set(zip(
        clinvar_all["Chromosome"].astype(str),
        # The position must be numerical, so I use pandas to convert to an integer, and fill out all errors/crashes with -1 
        # instead of NaN, which will still not work.
        pd.to_numeric(clinvar_all["PositionVCF"], errors="coerce").fillna(-1).astype(int),
        clinvar_all["ReferenceAlleleVCF"].astype(str),
        clinvar_all["AlternateAlleleVCF"].astype(str)))
    print(f"Exclusion set holds {len(clinvar_keys)} ClinVar variants")

    # This adds the ref_sequence to the ClinVar data as well as the gnomAD data. Note that the ClinVar data was already built at the time of 
    # writing this line, which is why a simple append does not work, unlike gnomAD. It returns the refercence sequence as the bps before
    # the variant, the SNV, and the bps after.
    if "ref_sequence" not in ClinVar_rows.columns:
        ClinVar_rows["ref_sequence"] = (ClinVar_rows["sequence"].str[:HALF] +
                                         ClinVar_rows["ref"].astype(str) + ClinVar_rows["sequence"].str[HALF + 1:])

        # Returns how many ClinVar rows this function built ref_sequence for, by summing up the amount of reference sequences that did not 
        # match up with the window of base pairs.
        irregular = int((ClinVar_rows["ref_sequence"].str.len() != WINDOW).sum())
        print(f"Rebuilt ref_sequence for ClinVar rows ({irregular} wrong length)")

    gnomAD_frames = []
    # .items() returns a key-value pair, which respectively gets defined as gene and path. Path is each of the three gnomAD files.
    for gene, path in gnomAD_csv_paths.items():

        # Calls the consequence_targets function to find out how many of a variant type is needed per gene, then prints the values.
        targets = consequence_targets(ClinVar_rows, gene, ratio=ratio)
        print(f"\n{gene} targets: {targets}")

        # Appends what returns after calling build_gnomAD_benign to gnomAD_frames. Uses the "targets" parameter to return the correct amount
        # for each consequence. Path is defined from the for loop, and runs iteratively through all 3 file paths.
        gnomAD_frames.append(build_gnomAD_benign(path, gene, targets, exclude_keys = clinvar_keys,
                                                   faf_threshold = faf_threshold, seed = seed, limit = limit))
        
    # Redefining gnomAD_frames to make sure that there are no empty data frames
    gnomAD_frames = [f for f in gnomAD_frames if len(f) > 0]

    # Concatenates the threee tables in gnomAD_frames into a single data frame with variants from all 3 genes
    gnomAD_rows = pd.concat(gnomAD_frames, ignore_index=True) if gnomAD_frames else pd.DataFrame()

    # Makes the source of ClinVar_rows "ClinVar", that way, I can look at the final dataset to see if a variant originates from ClinVar.
    if len(gnomAD_rows) > 0:
        gnomAD_rows["source"] = "gnomAD"

    # Combines the ClinVar dataset and gnomAD dataset into one
    combined = pd.concat([ClinVar_rows, gnomAD_rows], ignore_index= True)

    # I make sure to convert the ClinVar chrom, ref, alt to a string so it matches gnomAD's format. 
    # For example, the number 14 becomes the string "14".
    for c in ["chrom", "ref", "alt"]:
        combined[c] = combined[c].astype(str)

    # To ensure the two datasets can match in format when checking for duplicates, I ensure “pos” is an integer. This makes sure
    # that 235 and 235.0 are still regarded as the same position. Na is -1 to prevent errors from occuring (like with NaN).
    combined["pos"] = pd.to_numeric(combined["pos"], errors = "coerce").fillna(-1).astype(int)

    # Removes matchups between gnomAD and ClinVar of the built dataset, using drop_duplicates(), keeping only the first of each duplicate.
    before = len(combined)
    combined = combined.drop_duplicates(subset=["chrom", "pos", "ref", "alt"], keep ="first")

    # Saves the data, prints the amount of duplicate variants and variants saved.
    print(f"\nDropped {before - len(combined)} duplicate variants")
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(combined)} rows to {out_path}")

    # Prints the amount of variants per label, and a crosstab.
    print(combined["label"].value_counts())
    print(pd.crosstab([combined["gene"], combined["consequence"]], combined["label"]).to_string())


    return combined

# LEGACY build_full_dataset, which was used the first time
# def build_full_dataset(clinvar_dataset_path, gnomAD_csv_paths, out_path, faf_threshold = 1e-6):
#  clinvar_all = load_ClinVar(clinvar_dataset_path)
#   clinvar_keys = set(zip(
#        clinvar_all["Chromosome"].astype(str),
#        pd.to_numeric(clinvar_all["PositionVCF"], errors="coerce").fillna(-1).astype(int),
#        clinvar_all["ReferenceAlleleVCF"].astype(str),
#        clinvar_all["AlternateAlleleVCF"].astype(str)))
#   ClinVar_rows = pd.read_csv(clinvar_dataset_path)
#   gnomAD_frames = [
#        build_gnomAD_benign(path, gene, targets, exclude_keys = None, faf_threshold, seed = 42, limit = None)
#       for gene, path in gnomAD_csv_paths.items()
#   ]
#   gnomAD_rows = pd.concat(gnomAD_frames, ignore_index= True ) if gnomAD_frames else pd.DataFrame()
#  combined = pd.concat([ClinVar_rows, gnomAD_rows], ignore_index= True)
#  combined.to_csv(out_path, index = False)
#  print(f"Combined Dataset: {len(combined)} total sequences")
#  print(combined["label"].value_counts())
#  return combined