"""
Report.py will generate clean clinical reports using the information collected by Integrated Gradients, providing an interpretable mechanism for doctors to use,
with clear statistics and lines formatted like a real report.
"""
import html
import os
import numpy as np
import pandas as pd

import Revel_cadd_alphamissense_bench as bench
from explain import GROUPS, groups_for
from model_extra_features import load_transcripts, load_ttn_exons, MANE, COMPLEMENT, CODON_TABLE

# Converts each one letter abbreviation for an amino acid to a three letter short form
THREE = dict(A="Ala", R="Arg", N="Asn", D="Asp", C="Cys", Q="Gln", E="Glu", G="Gly", H="His", I="Ile", L="Leu",
             K="Lys", M="Met", F="Phe", P="Pro", S="Ser", T="Thr", W="Trp", Y="Tyr", V="Val")

# Converts "Ter" to *, to signify a stop codon
THREE["*"] = "Ter"

# Finds the start and end amino acid positions of several important regions for the reports
DOMAINS = {
    "MYH7": [(32, 81, "SH3"), (85, 778, "Motor domain"), (781, 810, "IQ"), (839, 1935, "Coiled-coil rod")],
    "MYBPC3": [(153, 256, "C1"), (362, 452, "C2"), (453, 543, "C3"), (544, 633, "C4"), (645, 771, "C5"),
               (774, 870, "C6"), (872, 967, "C7"), (971, 1065, "C8"), (1068, 1163, "C9"), (1181, 1274, "C10")],}

# Generates the genome positions to go from high to low, like the minus strand
def reading_order(t):

    # Sorts in descending order
    in_order = [q for start, end in t.exons for q in range(end, start, -1)]

    # Returns the sorted order, with indices matching in order, and where the coding sequence starts and ends
    return {q: k for k, q in enumerate(in_order)}, in_order.index(t.coding[0]), in_order.index(t.coding[-1])

# Converts an exonic genomic position into its coding-DNA component
def c_coordinate(t, order, p):

    where, first, last = order

    # Looks up the coding variant reading-order index of a genomic position
    k = where[p]

    # If the genomic position is before the first base, return the distance with "-"
    if k < first:
        return f"-{first - k}"

    # If the genomic position is after the first base, return the distance with "*"
    if k > last:
        return f"*{k - last}"

    # Returns the coding dna portion, adds 1 to match up with HGVS
    return str(t.coding_index[p] + 1)
