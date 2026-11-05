"""
vus.py downloads variants of uncertain significance (VUS) from ClinVar, with the kinds of variants my model was
trained on. It builds each one's 201-letter windows from the GRCh38 reference genome and classifies the variant as
"HCM", "DCM", or "Benign", and then uses report.py to create a clinical report, explaining the classifications. These VUS
variants will be later checked in a few months to look for reclassifications, and if there are reclassifications, I compare
them to what my model predicted.
"""
import os
import time
import numpy as np
import pandas as pd
import requests
import explain
import report
import Revel_cadd_alphamissense_bench as bench
from model_extra_features import annotate

# File paths
HERE = os.path.dirname(os.path.abspath(__file__))
EXTERNAL = os.path.join(HERE, "..", "data", "external")

# Entrez utilities that includes ClinVar (searches for the VUS variants)
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# Defines the gene regions, chromosome number and genomic positions
REGIONS = {"MYH7": ("14", 23380000, 23440000), "MYBPC3": ("11", 47330000, 47360000), "TTN": ("2", 178520000, 178820000)}

# What to search ClinVar for
QUERIES = [("MYH7", "missense variant"), ("MYBPC3", "missense variant"), ("MYBPC3", "intron variant"),
           ("MYBPC3", "splice donor variant"), ("MYBPC3", "splice acceptor variant"), ("TTN", "nonsense")]

# Uses a method like "GET" to search up data from ClinVar, with a timeout of 300 seconds and **kwargs (keyword arguments) 
# to receive extra information
def ask(method, url, **kwargs):
    for attempt in range(4):
        r = requests.request(method, url, timeout = 300, **kwargs)

        # Returns the answer r, if the error isn't 429 (too many requests) or 500 or higher (the server itself broke)
        if r.status_code != 429 and r.status_code < 500:
            r.raise_for_status()
            return r    

        # Wait time before trying again in case of 429 or 500 or higher; wait time increases 
        time.sleep(3 * (attempt + 1) ** 2)
    r.raise_for_status()

def region_sequence(gene):

    # Finds the chromosome number, start, and end of gene
    chrom, start, end = REGIONS[gene]

    # Builds the file path for that DNA stretch
    path = os.path.join(EXTERNAL, f"region_{gene}_{chrom}_{start}_{end}.txt")

    # If the DNA sequence data with the reference letters are not in my data, use ENSEMBL to get that data.
    if not os.path.exists(path):
        url = f"https://rest.ensembl.org/sequence/region/human/{chrom}:{start}..{end}:1?content-type=text/plain"
        r = ask("GET", url)

        # Makes the directory, writes down that data without spaces at the two ends, using "w".
        os.makedirs(EXTERNAL, exist_ok = True)
        with open(path, "w") as fh:
            fh.write(r.text.strip())

    # Reads the saved file back, stops the code if “seq” isn’t the right size, and returns seq.
    seq = open(path).read().strip()
    assert len(seq) == end - start + 1, f"{gene} region has {len(seq)} letters, expected {end - start + 1}"
    return seq

def search_vus(gene, consequence):

    # Filter by gene, VUS properties, SNV only, and consequence
    term = (f'{gene}[gene] AND "clinsig vus"[Properties] AND "single nucleotide variant"[Type of variation] '
            f'AND "{consequence}"[molecular consequence]')

    # Uses esearch to get the VUS ID numbers, **kwargs is taken advantage of because params are passed. Retmax increases
    # the limit of how many IDs are returned. 
    r = ask("GET", EUTILS + "esearch.fcgi", params = {"db": "clinvar", "term": term, "retmode": "json", "retmax": 10000})

    # Sleeps to prevent overwhelming ClinVar
    time.sleep(0.5)                            

    # Uses json to turn the reply into a dictionary, and idlist pulls out a list of ID
    return r.json()["esearchresult"]["idlist"]

def summaries(ids, gene):
    rows = []

    # Runs in batches of 300 IDs looked up at a time
    for start in range(0, len(ids), 300):

        # Stores each batch into a chunk
        chunk = ids[start:start + 300]

        # Joins the 300 IDs in this batch with a comma, and plugs it into ask(), searching for a summary of the VUS variant,
        # including information like gene and molecular consequence
        r = ask("POST", EUTILS + "esummary.fcgi", data = {"db": "clinvar", "id": ",".join(chunk), "retmode": "json"})

        # Uses json to convert the results into python-friendly dictionaries
        result = r.json()["result"]

        # For each ClinVar ID in the chunk, get the results of that chunk and store it as "u", {} is a fallback
        for uid in chunk:
            u = result.get(uid, {})

            # Uses variation_set and gets "canonical_spdi" to store the data in SPDI (Sequence, Position, Deleted, Inserted)
            # format, with a fallback if it doesn't work.
            spdi = (u.get("variation_set") or [{}])[0].get("canonical_spdi", "")

            # Splits the SPDI format based on ":", meaning multiple parts are made
            parts = spdi.split(":")

            # The amount of parts by the split should be 4, and the deletion and insertion should be 1 letter each
            if len(parts) != 4 or len(parts[2]) != 1 or len(parts[3]) != 1:
                continue

            # Drops "NC", the version, and leading zeroes
            chrom = str(int(parts[0].split(".")[0].replace("NC_", "")))

            # Appends data to “rows”, such as the accessionID (to be used in a few months to check for reclassification updates
            #, position, clinvar_consequence, review status, and when it was last evaluated. 
            rows.append({"clinvar_id": u.get("accession", uid), "name": u.get("title", ""), "gene": gene, "chrom": chrom,
                         "pos": int(parts[1]) + 1, "ref": parts[2], "alt": parts[3],                 # SPDI counts from 0
                         "clinvar_consequence": ";".join(u.get("molecular_consequence_list", [])),
                         "review": (u.get("germline_classification") or {}).get("review_status", ""),
                         "last_evaluated": (u.get("germline_classification") or {}).get("last_evaluated", "")})
        time.sleep(0.5)
    return rows

# Fetches the VUS data
def fetch_vus(path = os.path.join(EXTERNAL, "clinvar_vus.csv"), refresh = False):

    # Reads the CSV if the file is already saved and refresh is off
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, dtype = {"chrom": str})
    
    rows = []

    # Iterates over gene and consequence in QUERIES
    for gene, consequence in QUERIES:

        # Searches for the IDs of the variants
        ids = search_vus(gene, consequence)

        # Adds the row summaries to the rows
        rows += summaries(ids, gene)

        # Prints the gene, consequence, and the number of VUS variants
        print(f"{gene} {consequence}: {len(ids)} VUS")

    # Drops any duplicate rows, and resets the indices
    vus = pd.DataFrame(rows).drop_duplicates(subset = ["chrom", "pos", "ref", "alt"]).reset_index(drop = True)

    # Makes the directory, converts to a CSV, and returns it
    os.makedirs(EXTERNAL, exist_ok = True)
    vus.to_csv(path, index = False)
    return vus

# Returns the VUS dataframe with reference sequence and sequence columns
def make_windows(vus):

    # Makes a copy of the VUS dataframe, to be altered
    vus = vus.copy()

    ref_seqs, mut_seqs = [], []

    for gene, pos, ref, alt in zip(vus["gene"], vus["pos"], vus["ref"], vus["alt"]):

        # Chromosome number, start, and end of the gene is defined, then region_sequence is called to return the sequence
        chrom, start, end = REGIONS[gene]
        seq = region_sequence(gene)

        # Turns the genome position into an integer position inside of seq, and stops the code if that position is
        # within 100 bases of the edge, since that would mean the variant couldn't be in the middle position of 201 bp
        i = int(pos) - start
        assert 100 <= i <= len(seq) - 101, f"{gene} {pos} is too close to the edge of the downloaded region"

        # Creates the DNA window, a:b excludes the final part of b, the variant is in the middle, and letters become uppercase
        window = seq[i - 100:i + 101].upper()

        # Checks the middle sequence letter to make sure it matches the reference 
        assert window[100] == ref, f"{gene} {pos}: genome has {window[100]}, ClinVar says {ref}"

        # Adds the window to the empty ref_seqs list
        ref_seqs.append(window)

        # Makes sure the window is 100 letters to each side, with the mutation in the middle
        mut_seqs.append(window[:100] + alt + window[101:])

    vus["ref_sequence"], vus["sequence"] = ref_seqs, mut_seqs

    return vus

# Determines whether a ClinVar variant is missense, nonsense, or noncoding with my own code, and then checks 
# with ClinVar to see if that is true. This validates that the features code from the CNN is reading the new variants correctly.
def assign_consequence(vus, check = True):

    # Calls the annotate function from model_extra_features, determining details like the amino acid for ref and alt, and 
    # if the exon is coding or not.
    ann = annotate(vus)

    kind = []

    # For whether the variant is in the coding sequence, reference amino acid, and alternate amino acid, determine the kind of variant,
    # using if statements to determine the consequence of the variant (like a * indicating nonsense).
    for in_cds, a, b in zip(ann["in_cds"], ann["ref_aa"], ann["alt_aa"]):
        if in_cds and isinstance(a, str) and isinstance(b, str):
            kind.append("nonsense" if b == "*" else ("synonymous" if a == b else "missense"))
        elif in_cds:
            kind.append("unknown")                    
        else:
            kind.append("noncoding")
    vus = vus.assign(consequence = kind)

    # If check is True, define expected to be the two types of expected consequences, and checkable to be those consequences
    # that can be checked (not unknown)
    if check:
        expected = {"missense": "missense variant", "nonsense": "nonsense"}
        checkable = vus["consequence"].isin(expected)


        # Finds the number of variants that agree with ClinVar based on consequence, prints it, and stops the code 
        # if that number's mean is not > 0.98
        agree = [expected[k] in c for k, c in zip(vus.loc[checkable, "consequence"], vus.loc[checkable, "clinvar_consequence"])]
        print(f"Consequence check: my call matches ClinVar for {np.mean(agree):.1%} of {int(checkable.sum())} coding VUS")
        assert np.mean(agree) > 0.98, "my consequence calls disagree with ClinVar"

    return vus

def load_vus(full_dataset, keep_cells = ("MYH7/missense", "MYBPC3/missense", "MYBPC3/noncoding", "TTN/nonsense")):

    # Fetches the VUS data, makes the genomic windows, and assigns the consequences
    vus = assign_consequence(make_windows(fetch_vus()))

    # A lambda function that builds a unique ID text for every row of the table.
    key = lambda d: d["chrom"].astype(str) + ":" + d["pos"].astype(str) + d["ref"] + ">" + d["alt"]

    # Gets all the IDs of my full dataset and put it into a set called "known"
    known = set(key(full_dataset))

    # To prevent data leakage, remove all VUS variants that were in the built dataset (could have been reclassified).
    # Then, reset the index for the remaining VUS variants
    vus = vus[~key(vus).isin(known)]
    vus = vus[(vus["gene"] + "/" + vus["consequence"]).isin(keep_cells)].reset_index(drop = True)

    # Describe the label column as "VUS" and "chrom" as the integer chromosome number
    vus["label"] = "VUS"
    vus["chrom"] = vus["chrom"].astype(int)

    # Print out the amount of VUS variants ready
    print(f"{len(vus)} VUS ready:", (vus["gene"] + "/" + vus["consequence"]).value_counts().to_dict())
    return vus

# Scores and creates a report for every VUS variant in the data, adds REVEL/CADD/AlphaMissense, checks IG for completeness, and
# batch_size keeps memory low
def score_vus(vus, matched, models, cv_preds, results_dir, steps = 128, batch_size = 64):

    # Feature scaling and typical control as every other explanation, with the VUS added as "extra"
    setup = explain.final_setup(matched, extra = vus)
    names = list(setup["raw"](setup["test_df"]).columns)
    typical = explain.typical_benign(setup["F_fit"], setup["fit_df"])
    ig = explain.explain_saved(models, setup["extra_df"], setup["F_extra"], typical, names,
                               os.path.join(results_dir, "ig_vus.csv"), steps = steps, batch_size = batch_size)

    # The VUS scores are downloaded once to data/external, and REVEL/CADD/AlphaMissense are called on my data.
    vus_tools = bench.tool_scores_for(vus, os.path.join(EXTERNAL, "tool_scores_vus.csv"))
    known_tools = bench.check_tool_scores(matched, bench.fetch_tool_scores(matched))

    # Drops duplicates between VUS and my data, and builds the context (all features)
    all_tools = pd.concat([known_tools, vus_tools], ignore_index = True).drop_duplicates(subset = ["chrom", "pos", "ref", "alt"])
    ctx = report.build_context(matched, setup, cv_preds, all_tools)

    # Adds warnings to the report by using flags()
    warnings = report.warning_table(vus, ig, setup, ctx)

    # Concatenates variant information, prediction scores, tool scores, and warnings to one Pandas df
    table = pd.concat([vus[["clinvar_id", "name", "gene", "consequence", "chrom", "pos", "ref", "alt", "review",
                            "last_evaluated"]].reset_index(drop = True),
                       ig[["score", "baseline_score", "p_HCM", "p_DCM", "p_Benign"]].reset_index(drop = True),
                       vus_tools[bench.TOOLS].reset_index(drop = True), warnings], axis = 1)

    # The number of variants in the table should be the number of VUS
    assert len(table) == len(vus), "one row per VUS"

    # Converts the df to a CSV file, and adds the number of variants saved
    table.to_csv(os.path.join(results_dir, "vus_scores.csv"), index = False)
    print(f"Saved {len(table):,} VUS to {os.path.join(results_dir, 'vus_scores.csv')}")

    return table, ig, setup, ctx