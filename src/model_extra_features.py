"""
Model_extra_features is responsible for providing data on the exon the variant is in,
how far along the protein it is, evolutionary data, hydropathy, volume, charge, Grantham distances,
and how close to a splice site the codon is. 
It also converts data from get_annotations, such as PSI scores, to a clean table format for the model to use.
Information about source, like gnomAD or ClinVar, is never given to the model, to prevent data leakage.
"""

import json
import os
import numpy as np
import pandas as pd

# Defines the annotation directory, using os and file to find the correct path/directory and properly name it, going up one
# file folder at a time.
ANN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "annotations")

# Defines the names of important features later to be used
ALL_GROUPS = ("base", "tx", "psi", "aa")

# Defines the dictionary lookups, of the gene name, chromosome number and clinical reference transcript
MANE ={"MYH7": ("chr14", "NM_000257.4"), "MYBPC3": ("chr11", "NM_000256.3"), "TTN": ("chr2", "NM_001267550.2")}

# A pairs with T, C pairs with G, and this constructs that translation table (complement)
COMPLEMENT = str.maketrans("ACGT", "TGCA")

# Base string and Amino string (NCBI published)
_BASES = "TCAG"
_AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"

# Dictionary lookup, enumerates bases.
CODON_TABLE = {

# 'a' is the first base of the codon, 'b' is the second, and 'c' is the third
a + b + c: _AMINO[16 *i + 4 * j + k]

# i, j, k are positions (0 to 3) in "TCAG"; a, b, c are the letters at those positions.
# 16*i + 4*j + k is the codon's place in _AMINO
# For each, looks at the index of a specific base, and assigns it to a base letter, making a lookup table for an amino acid.
for i, a in enumerate(_BASES)
for j, b in enumerate(_BASES)
for k, c in enumerate(_BASES)
}

# Provides the blueprint for taking a transcript from UCSC, and uses it to build 
# an ordered map of exons and coding DNA, in the correct direction of the way the gene is actually read.
class Transcript:

    # Runs automatically when creating a transcript, record contains data like exonStarts and exonEnds in the UCSC record
    def __init__(self, record):

        # strip(",") removes that end comma, split(",") cuts the string at every comma, and int() turns each piece into a number
        # For example, "100,300,500," -> ["100", "300", "500",] -> [100, 300, 500]
        starts = [int(x) for x in record["exonStarts"].strip(",").split(",")]
        ends = [int(x) for x in record["exonEnds"].strip(",").split(",")]

        # I make exons equal to the zip of the starts and ends sorted by a specific key that 
        # sorts the values in descending order, by converting the values to negative numbers. The order is made by checking
        # the negative of each exon's start, so the highest start comes up first.
        self.exons = sorted(zip(starts, ends), key = lambda e: -e[0])
        self.n_exons = len(self.exons)

        # Defines cds_start and cds_end using portions of "record"
        cds_start = record["cdsStart"]
        cds_end = record["cdsEnd"]

        coding = []

        # For loop that takes the starts and ends of self.exon
        for start, end in self.exons:

            # Walks down, one step at a time, from the highest number (the end) to the lowest number (the start),
            # which is the minus strand reading direction.
            for p in range(end,start,-1):

                # Keep only positions that fall within the coding sequence (CDS).
                if cds_start < p <= cds_end:
                    coding.append(p)

        # Converts to a NumPy array, defines the index for each position
        self.coding = np.array(coding)
        self.coding_index = {p: i for i, p in enumerate(coding)}

        # In gene-reading order, finds the last exon (smallest position).
        last_exon_top = self.exons[-1][1]

        # Gets the coding positions of everything over the coding position of the last exon.
        # Then takes the max of index to find the last junction, or the last coding position before the final exon. 
        self.last_junction = max((i for i, p in enumerate(coding) if p > last_exon_top), default = len(coding))

# Is the variant's position in between the start and end of an exon? If yes, it is an exon, and return the exon index.
    def locate(self, pos):
        
        for exon_number, (start, end) in enumerate(self.exons):
            if start < pos <= end:
                return exon_number, 0

        # Sets these values as placeholders later to be changed
        best_exon_number, best_distance = 0, None

        # For loop that calculates the distance to each exon for each exon_number
        for exon_number, (start, end) in enumerate(self.exons):
            # For every edge, the distance to the exon is the postion minus the edge
            # Start +1 is used because UCSC’s half-open interval system makes the start one before the first base
            for edge in (start + 1 , end):
                distance = abs(pos - edge)

                # If distance is smaller then best_distance, or best_d is currently None,
                # that distance and its exon number should respecitvely be set as best_distance and best_exon_number
                if best_distance is None or distance < best_distance:
                    best_exon_number, best_distance = exon_number, distance

        # After checking all cases, return best_exon_number and best_distance
        return best_exon_number, best_distance
    
def load_transcripts(ann_dir = ANN_DIR):
    out = {}

    for gene, (chrom, refseq_id) in MANE.items():

        # uses json to open each tx_{chromosome}.json file (made in get_annotations.py, used to download exon maps),
        # and filters for [“ncbiRefSeq”] to keep only the list of transcripts.
        records = json.load(open(os.path.join(ann_dir, f"tx_{chrom}.json")))["ncbiRefSeq"]

        # The out dictionary for each gene is constructed, checking if the transcript resulting from "records" matches the refseq_id. 
        # If it does match, it plugs that match into the Transcript class, resulting in the NumPy coding array and last_junction.
        out[gene] = Transcript([r for r in records if r["name"] == refseq_id][0])

    return out

def load_ttn_exons(ann_dir = ANN_DIR):

    # uses json to open each ttn_exons.json file (made in get_annotations.py, used to download exon maps)
    raw = json.load(open(os.path.join(ann_dir, "ttn_exons.json")))
    rows = []

    for v in raw.values():

        # Defines low as the smaller coordinate, high as the higher coordinate, useful for genomic overlap calculations
        low, high = sorted((v["s38"], v["e38"]))

        # Defines iso, which stands for "isoforms" of TTN, and has different isoforms as columns with a "-" if absent.
        iso = v["tx"] or ["-"] * 7

        # Appends useful information for the model to "rows", going from messy data into a clean, usable table.
        rows.append({
        "low": low,
        "high": high,
        "psi_dcm": (v["psi_dcm"] or 0) / 100,    # Percent Spliced In (How important it is for the heart)
        "psi_gtex": (v["psi_gtex"] or 0) / 100,  # Normal hearts to serve as comparison
        "region": v["region"] or "none",
        "in_n2ba": float(iso[1] != "-"),  # The != checks as True return a value of 1.0
        "in_n2b": float(iso[2] != "-"),
        "in_n2a": float(iso[3] != "-"),
        "in_novex3": float(iso[6] != "-")
        })

    # Returns dataframe, sorts by low, from smallest to largest, resets indexes
    return pd.DataFrame(rows).sort_values("low").reset_index(drop = True)

# Hydropathy is the water loving/hating metric, which can alter protein interactions
HYDROPATHY = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
                  L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)   

# Volume dictates how much space an amino acid takes up, and large substitutions can cause destabilizing cavities, while
# small substitutions may leave gaps
VOLUME = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7,
              L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)

# Charges hold proteins together, and a change in charge can result in differences in protein structure, folding, and stability.
CHARGE = dict(R=1, K=1, H=0.5, D=-1, E=-1)

# Grantham distnaces computes on composition, polarity, volume for each amino acid
GRANTHAM = {"C": (2.75, 5.5, 55), "S": (1.42, 9.2, 32), "T": (0.71, 8.6, 61), "P": (0.39, 8.0, 32.5),
             "A": (0, 8.1, 31), "G": (0.74, 9.0, 3), "N": (1.33, 11.6, 56), "D": (1.38, 13.0, 54),
             "E": (0.92, 12.3, 83), "Q": (0.89, 10.5, 85), "H": (0.58, 10.4, 96), "R": (0.65, 10.5, 124),
             "K": (0.33, 11.3, 119), "M": (0, 5.7, 105), "I": (0, 5.2, 111), "L": (0, 4.9, 111),
             "V": (0, 5.9, 84), "F": (0, 5.2, 132), "Y": (0.20, 6.2, 136), "W": (0.13, 5.4, 170)}

# Finds the grantham distnace between two amino acids
def grantham(a,b):
    
    c1, p1, v1 = GRANTHAM[a]
    c2, p2, v2 = GRANTHAM[b]
    
    # Each weight = 1 / (average difference of that property over all 190 amino acid pairs)^2
    #   composition: avg 0.7394 -> 1/0.7394^2 = 1.829 
    #   polarity:    avg 3.134  -> 1/3.134^2  = 0.1018
    #   volume:      avg 50.06  -> 1/50.06^2  = 0.000399
    total = 1.833 * (c1 - c2) ** 2 + 0.1018 * (p1 - p2) ** 2 + 0.000399 * (v1 - v2) ** 2

    # Square root turns the sum back into a distance (3D Pythagoras).
    # 50.723 scales it so the average distance over all 190 pairs is 100.
    return 50.723 * total ** 0.5


# Evolutionary data, logarithimic format where positive means evolutionarily favorable, and negative means 
# evolutionarily unfavorable
BLOSUM_TEXT = """A 4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
R -1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
N -2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
D -2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
C 0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
E -1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
G 0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
H -2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
L -1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
K -1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
M -1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
F -2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
S 1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
T 0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
V 0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4"""


# splitlines() makes a new line for each row, line.split() sepereates the string at whitespace
ROWS = [line.split() for line in BLOSUM_TEXT.splitlines()]

# Sorts the column labels in order, giving indexes
ORDER = [r[0] for r in ROWS]

# Creates the actual dictionary, checking each j (index), v(value), storing as an integer for each value.
# It goes from 1 because index 0 is "A", which is just a label that has no numerical value.
BLOSUM62 = {(r[0], ORDER[j]): int(v) for r in ROWS for j, v in enumerate(r[1:])}


# Runs the tools and table constructors on my actual data, writing down useful information about each variant.
def annotate(df, ann_dir = ANN_DIR):

    # Load_transcripts and load_ttn_exons is called on my data
    transcripts = load_transcripts(ann_dir)
    ttn = load_ttn_exons(ann_dir)   

    rows = []
    # Loops through each variant in the dataframe, resulting in lightweight tuples. 
    for r in df.itertuples():

        # Defines transcript and pos by looking at the rows. Transcript includes important information including but not limited to
        # the list of exons, the position, the number of exons, the last junction, and the coding index.
        t= transcripts[r.gene]
        pos = int(r.pos)

        o = {} # Output

        # Where the variant sits in the exon map, uses locate() to calculate distance to the exon
        exon, dist = t.locate(pos)

        # Uses a percentage to show how far along a gene exon, based on the exon number. I divide by t.n_exons - 1 because exon number starts at 0, an the final 
        # exon must be 100% of the way through.
        o["exon_frac"] = exon / (t.n_exons - 1)

        # Finds if the variant is in the last exon. 
        o["last_exon"] = float(exon == t.n_exons -1)

        # Checks to see if the variant is intronic using the "distance" marker, since a positive distance implies it is at least some positions away from an exon.
        # The column "intron_dist" measures far into the intron the variant is. 
        o["intronic"] = float(dist > 0)
        o["intron_dist"] = float(dist)
        
        # Uses the start and ends to check the distance to an exon edge (assuming the variant is an exon, else it just returns 0.0).
        # It checks distance to edge by finding the minimum of distance to the low edge and high edge.
        s, e = t.exons[exon] 
        o["exon_edge_dist"] = float(min(pos -s -1, e - pos)) if dist == 0 else 0.0

        # First checks if the genomic position even exists in the coding base, by using the coding_index from “t”,
        # which is the load_transcripts() function of each gene
        in_cds = pos in t.coding_index
        o["in_cds"] = float(in_cds)

        # If in_cds yields "True", that means that the genomic position correlates to an exon, and I retrieve that position and store it
        # as "ci" (shorthand for coding index). If in_cds is "False", find the nearest coding position, and return its index using argmin(). 
        ci = t.coding_index[pos] if in_cds else int(np.argmin(np.abs(t.coding - pos)))
        o["protein_frac"] = ci/len(t.coding)

        # NMD (Nonsense Mediated Decay) is escaped when a (coding) stop codon occurs within 50 bases of the final junction. Instead, a short protein is made.
        # This checks that rule for every variant.
        o["nmd_escape"] = float(in_cds and ci >= t.last_junction - 50)

        # Starts off with "None" as reference amino acid and alternate amino acid values, then finds the remainder of dividing 
        # the coding index by 3 (because a codon includes 3 bases).
        # It uses this dividing by 3 to find the position of the codon the variant is at. 
        ref_aa = alt_aa = None
        if in_cds:
            frame = ci % 3

            # Codon_position is the coding index minus the frame to that value + 3. Checks the codon_pos of each variant one at a time through this for loop.
            codon_pos = t.coding[ci - frame: ci - frame + 3]

            # Only read a given codon if all 3 letters are inside the 201-letter window
            if len(codon_pos) == 3 and all (abs(p-pos) <= 100 for p in codon_pos):

                # Joins the each base of the variant's codon located in codon_pos together.
                # Translates the DNA sequence because the stored window is the plus end, but the gene is actually read from the minus strand.
                codon = "".join(r.ref_sequence[p - pos + 100] for p in codon_pos).translate(COMPLEMENT)

                # This changes the codon to include the alternate base, also known as the mutation. 
                # It finds the correct position to keep the alternate base by using frame (variant position), and then translates the regular sequence to the 
                # alternate, mutation-including one.
                mutant = list(codon)
                mutant[frame] = r.alt.translate(COMPLEMENT)

                # Translates the variant codons to a reference amino acid table, and the mutated codons to alternate amino acid table.
                ref_aa, alt_aa = CODON_TABLE.get(codon), CODON_TABLE.get("".join(mutant))

        o["ref_aa"], o["alt_aa"] = ref_aa, alt_aa

        # If the gene is TTN, find the row whose exon starts before the variant and ends after the variant.
        if r.gene == "TTN":
                hit = ttn[(ttn["low"] <= pos) & (ttn["high"] >= pos)]

                # If the variant is intronic, then find the nearest exon instead, by taking the minimum of the distance from the high edge and 
                # low edge. Uses iloc to extract the specific row by filtering for index.
                if len(hit) == 0:
                     gap = np.minimum(np.abs(ttn["low"] - pos), np.abs(ttn["high"] - pos) )
                     hit = ttn.iloc[[int(np.argmin(gap))]]

                # Returns PSI (Percent spliced in) and TTN isoforms
                h = hit.iloc[0]
                # Copies the data to the outputs ("o")
                for c in ["psi_dcm", "psi_gtex", "region", "in_n2ba", "in_n2b", "in_n2a", "in_novex3"]:
                    o[c] = h[c]

        # Appends the dictionary values of "o" to the list of rows
        rows.append(o)

    # Full dataframe of rows
    return pd.DataFrame(rows, index = df.index)

# Takes the clean output from annotate, and turns it into a matrix/table that the model can actually use
def build(df, groups = ALL_GROUPS, ann = None, gene_span = None):

    # If "ann" does not have a value, calculate its correct value
    if ann is None:
        ann = annotate(df)

    # Defines an empty dataframe called "X" and attributes the label is_ttn to all genes that are TTN.
    X = pd.DataFrame(index = df.index)
    is_ttn = (df["gene"] == "TTN").to_numpy()

    # Finds gene and consequence
    if "base" in groups:
        # Checks if a variant corresponds to a gene, and does so for all 3 genes, outputting "1.0" for a truth
        # statement match (one truth value match per row)
        for g in ["MYH7", "MYBPC3", "TTN"]: X[f"g_{g}"] = (df["gene"] == g).astype(float)

        # Similar encoding process, but for conseqeunces
        for c in ["missense", "nonsense", "noncoding"]:
            X[f"c_{c}"] = (df["consequence"]  == c).astype(float)

        # Using the gene name and the dataframe containing the variants, find the minimum and maximum variant positions.
        if gene_span is None:
            gene_span = {g:(d["pos"].min(), d["pos"].max()) for g, d in df.groupby("gene")}

        # First output of "g" is the minimum value, second output is the maximum value
        low = df["gene"].map(lambda g: gene_span[g][0])
        high = df["gene"].map(lambda g: gene_span[g][1])

        # Normalized position of the variant from 0 to 1. Does so by dividing how far the variant is from the end of the gene
        # by the total length of the gene.
        relative = ((high - df["pos"]) / (high-low)).to_numpy()
        X["rel_pos"] = relative
        for g in ["MYH7", "MYBPC3", "TTN"]:
            X[f"rel_{g}"] = np.where(df["gene"] == g, relative, -1.0)

        # For each DNA base, makes a reference and alternate column for it, using the dataframe.
        for b in "ACGT":
            X[f"ref_{b}"] = (df["ref"] == b).astype(float)
            X[f"alt_{b}"] = (df["alt"] == b).astype(float)

        # “Transitions” is defined as all possible transition substitutions, like “A” to “G”. Transversions may still occur.
        #  Then, the “transition” column is added, which converts these values to floats for the table.
        transitions = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
        X["transition"] = [float((a, b) in transitions) for a, b in zip(df["ref"], df["alt"])]

        # CpG variants are 10x more likely to be in a population. They occur when G follows C, and are 10 times more likely in a population.
        X["cpg"] = [float((s[100] == "C" and s[101] == "G") or (s[99] == "C" and s[100] == "G"))
                    for s in df["ref_sequence"]]
        
    # Finds details about the exon and position
    if "tx" in groups:
        # Takes important data from annotate() and makes it a NumPy table, with label tx (transcript)
        for c in ["exon_frac", "last_exon", "intronic", "in_cds", "protein_frac", "nmd_escape"]:
            X[f"tx_{c}"] = ann[c].to_numpy()

        # Converts the intron_dist and exon_edge_dist from the annotate function into columns of the table
        # Takes the logarithm to compress these values, because the values are very spread out (can be 1bp or 10,000 bp)
        # log1p adds 1 inside of the parenthesis, so that the log(0) is never taken
        X["tx_intron_dist"] = np.log1p(ann["intron_dist"].to_numpy())
        X["tx_exon_edge_dist"] = np.log1p(ann["exon_edge_dist"].to_numpy())

        # Returns a normalized score of how far a variant is located in a gene, with one truth value match
        # per row (corresponding to one particular gene). For a gene that doesn't match the variant, adds "-1.0"
        # to the table.
        for g in ["MYH7", "MYBPC3", "TTN"]:
            X[f"tx_prot_{g}"] = np.where(df["gene"] == g, ann["protein_frac"], -1.0)

    # Finds isoforms and PSI data
    if "psi" in groups:
            # Takes the Percent Spliced In scores and TTN isoforms from annotate, otherwise returning a value of "0.0"
            for c in ["psi_dcm", "psi_gtex", "in_n2ba", "in_n2b", "in_n2a", "in_novex3"]:
                col = ann[c] if c in ann else pd.Series(0.0, index = df.index)

                # makes sure Non-TTN variants get “-1.0”, NaN values are filled with “0.0”, and TTN variants
                # return their respective PSI score to indicate how important they are to the heart, with high scores indicating an increased chance of
                # pathogenicity because an important exon has been disrupted.
                X[f"ttn_{c}"] = np.where(is_ttn, col.fillna(0).astype(float), -1.0)

                # TTN is a big gene that can be in a variety of regions of the muscle sarcomere, from the Z-disk to the M-band,
                # so I return the region for each TTN variant's exon.
            region = ann["region"] if "region" in ann else pd.Series(None, index = df.index)
            for reg in ["A-band", "I-band", "Z-disk", "M-band"]:
                X[f"ttn_{reg}"] = ((region == reg).to_numpy() & is_ttn).astype(float)

    # Finds detail about the amino acid swap, such as hydropathy and evolutionary data
    if "aa" in groups:
        ra, aa = ann["ref_aa"], ann["alt_aa"]

        # Checks if the variant is “ok”, making sure that the reference amino acid and alternate amino acids aren’t 100% equal, a stop codon, or an invalid format.
        ok = [isinstance(a, str) and isinstance(b, str) and "*" not in (a,b) and a != b for a, b in zip(ra, aa)]

        # Converts all biological data from the annotate function, if matching the "ok" criteria
        # For each column, uses an approximate scaling factor to keep the data in similar scales.
        X["aa_grantham"] = [ grantham(a, b) / 215 if k else 0.0  for a, b, k in zip(ra, aa, ok)]
        X["aa_blosum"] = [BLOSUM62[(a, b)] / 4 if k else 0.0 for a, b, k in zip(ra, aa, ok)]
        X["aa_hydro"] = [(HYDROPATHY[b] - HYDROPATHY[a]) / 9 if k else 0.0 for a, b, k in zip(ra, aa, ok)]
        X["aa_volume"] = [(VOLUME[b] - VOLUME[a]) / 170 if k else 0.0 for a, b, k in zip(ra, aa, ok)]
        X["aa_charge"] = [CHARGE.get(b, 0) - CHARGE.get(a, 0) if k else 0.0 for a, b, k in zip(ra, aa, ok)] # Charge specifically needs .get() because some amino acids have a neutral charge. 
        X["aa_to_pro"] = [float(k and b == "P") for b, k in zip(aa, ok)] # Flags proline for the model due to its unusal ring structure
        X["aa_gly"] = [float(k and "G" in (a, b)) for a, b, k in zip(ra, aa, ok)] # Flags glycine for the model due to its unusually small side chain
        X["aa_cys"] = [float(k and "C" in (a, b)) for a, b, k in zip(ra, aa, ok)] # Flags cysteine because cysteine can form disulfide

    # Returns all of that data
    return X