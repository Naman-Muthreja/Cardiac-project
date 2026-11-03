"""
vus.py downloads variants of uncertain significance (VUS) from ClinVar, with the kinds of variants my model was
trained on. It builds each one's 201-letter windows from the GRCh38 reference genome and classifies the variant as
"HCM", "DCM", or "Benign", and then uses report.py to create a clinical report, explaining the classifications.
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

# Defines the gene regions,chromosone number and genomic positions
REGIONS = {"MYH7": ("14", 23380000, 23440000), "MYBPC3": ("11", 47330000, 47360000), "TTN": ("2", 178520000, 178820000)}

# What to search ClinVar for
QUERIES = [("MYH7", "missense variant"), ("MYBPC3", "missense variant"), ("MYBPC3", "intron variant"),
           ("MYBPC3", "splice donor variant"), ("MYBPC3", "splice acceptor variant"), ("TTN", "nonsense")]

# Uses a method like "GET" to search up data from ClinVar, with a timeout of 300 seconds and **kwargs (keyword arguments) 
# to receive extra information
def ask(method, url, **kwargs):
    for attempt in range(4):
        r = requests.request(method, url, timeout = 300, **kwargs)

        # Returns the answer r, if the error isn't 429 (too many requests) or 500 (the server itself broke)
        if r.status_code != 429 and r.status_code < 500:
            r.raise_for_status()
            return r    

        # Wait time before trying again in case of 429 or 500; wait time increases 
        time.sleep(3 * (attempt + 1) ** 2)
    r.raise_for_status()

def region_sequence(gene):

    # Finds the chromosome number, start, and end of gene's regionFor HTTP status codes below 500, excluding 429 (Too Many Requests), raise_for_status() raises an exception for unsuccessful responses, while successful responses are returned using return r. 
    chrom, start, end = REGIONS[gene]

    # Makes the variant names using that data
    path = os.path.join(EXTERNAL, f"region_{gene}_{chrom}_{start}_{end}.txt")

    # If the DNA sequence data with the reference letters are not in my data, so use ENSEMBL to get that data.
    if not os.path.exists(path):
        url = f"https://rest.ensembl.org/sequence/region/human/{chrom}:{start}..{end}:1?content-type=text/plain"
        r = ask("GET", url)

        # Makes the directory, writes down that data without spaces at the two ends, using "w".
        os.makedirs(EXTERNAL, exist_ok = True)
        with open(path, "w") as fh:
            fh.write(r.text.strip())

    # Writes down the sequence as “seq”, stops the code if “seq” isn’t the right size, and returns seq.
    seq = open(path).read().strip()
    assert len(seq) == end - start + 1, f"{gene} region has {len(seq)} letters, expected {end - start + 1}"
    return seq

def search_vus(gene, consequence):

    # Filter by gene, VUS properties, SNV only, and consequence
    term = (f'{gene}[gene] AND "clinsig vus"[Properties] AND "single nucleotide variant"[Type of variation] '
            f'AND "{consequence}"[molecular consequence]')

    # Uses esearch to get the VUS variants info, **kwargs is taken advantage of because params are passed. Retxmax increases
    # the limit of how many IDs are returned. 
    r = ask("GET", EUTILS + "esearch.fcgi", params = {"db": "clinvar", "term": term, "retmode": "json", "retmax": 10000})

    # Sleeps to prevent overwhelming ClinVar
    time.sleep(0.5)                            

    # Converts the text into a python dictionary
    return r.json()["esearchresult"]["idlist"]

def summaries(ids, gene):
    rows = []

    # Runs in batches of 300 IDs looked up at a time
    for start in range(0, len(ids), 300):

        # Stores each batch into a chunk
        chunk = ids[start:start + 300]

        # Combines all of those chunks and plugs it into ask(), searching fo rthe 
        r = ask("POST", EUTILS + "esummary.fcgi", data = {"db": "clinvar", "id": ",".join(chunk), "retmode": "json"})

        # Uses json to convert the results into python-friendly dictionaries
        result = r.json()["result"]

        # For each userID in the chunk, get the results of that chunk and store it as "u", {} is a fallback
        for uid in chunk:
            u = result.get(uid, {})

            # Uses variation_set and gets "canonical_spdi" to store the data in SPDI (Sequence, Position, Deleted, Inserted)
            # format, with a fallback if it doesn't work.
            spdi = (u.get("variation_set") or [{}])[0].get("canonical_spdi", "")

            # Splits the SPDI format based on ":", meaning multiple parts are maade
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

            # Pause and hand back
            time.sleep(0.5)
            return rows

# Fetches the VUS data
def fetch_vus(path = os.path.join(EXTERNAL, "clinvar_vus.csv"), refresh = False):

    # Reads the CSV
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, dtype = {"chrom": str})
    
    rows = []

    # Iterates over gene and conseqeunce in QUERIES
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