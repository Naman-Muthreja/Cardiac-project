"""
Model_extra_features is responsible by providing data on the exon the variant is in,
how far along the protein it is, the codon and therefore amino acid changed, and how close to a splice site the codon is.
This gives useful information about the CODING part of the DNA sequence.

Information about source, like gnomAD or CLinVar, is never given to the model, to prevent data leakage.
"""

import json
import os
import numpy as np
import pandas as pd

# Defines the annotation directory, using os and file to find the correct path/directory and properly name it, going up one
# file from "data" at a time.
ANN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "annotations")

# Defines the names of important features later to be used
ALL_GROUPS = ("base", "tx", "psi", "aa")

# Defines the dictionary lookups, of the gene name, chromosome number and clinical reference transcript
MANE ={"MYH7": ("chr14", "NM_000257.4"), "MYBPC3": ("chr11", "NM_000256.3"), "TTN": ("chr2", "NM_001267550.2")}

# A pairs with G, C pairs with T, and this constructs that translation table (complement)
COMPLEMENT = str.maketrans("ACGT", "TGCA")

# Base string and Amino string (NCBI published)
_BASES = "TCAG"
_AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"

# Dictionary lookup, enumaterates bases.
CODON_TABLE = {

# 'a' is the first base of the codon, 'b' is the second, and 'c' is the third
a + b + c: _AMINO[16 *i + 4 * j + k]

# Sets i equal to base A, j equal to base T, and k equal to base G. 'i' is first base index, 'b' is second, 'c' is third.
# For each, looks at the index of a specific base, and assigns it to each letter, constructing an amino sequence.
for i, a in enumerate(_BASES)
for j, b in enumerate(_BASES)
for k, c in enumerate(_BASES)
}