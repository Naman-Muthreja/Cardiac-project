"""
Model_extra_features is responsible for providing data on the exon the variant is in,
how far along the protein it is, the codon and therefore amino acid changed, and how close to a splice site the codon is.
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
        # For example, "100,300,500" -> ["100", "300", "500",] -> [100, 300, 500]
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
        # Then takes the max of index to find the last junction 
        self.last_junction = max((i for i, p in enumerate(coding) if p > last_exon_top), default = len(coding))

# Is the variant's position in between the start and end of an exon? If yes, it is an exon, and return the exon number.
    def locate(self, pos):
        
        for exon_number, (start, end) in enumerate(self.exons):
            if start < pos <= end:
                return exon_number, 0

        # Sets these values as placeholders later to be changed
        best_exon_number, best_distance = 0, None

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

