"""
This function works out what a variant actually does to the protein, using the HGVS protein notation that ClinVar stores at the end of its Name
column. After realizing that 0 DCM variants were actually able to be tested by REVEL, I wrote this function to see if there was something improper
about the data. Running this function helped me conclude the two main issues in my code: Most benign variants were synonymous, and my HCM and DCM
code functioned as a simple gene classifier rather than learning real biological data.
"""

import re

# Matches the HGVS notation for a protein change, followed by 3 letters which encode for an amino acid, then the position number, and then the new amino acid.
# Ex: for p.Arg663His (meaning that there was an amino acid change from Arginine to Histidine), it will capture "Arg", "663" and "His".
# The last part means read 3 letters or an equal to sign, which is two ways ClinVar writes the amino acid. 
HGVS_PROTEIN = re.compile(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|=)")

VEP_TO_CONSEQUENCE = {
    "missense_variant":         "missense",
    "stop_gained":              "nonsense",
    "synonymous_variant":       "synonymous",
    "splice_donor_variant":     "noncoding",
    "splice_acceptor_variant":  "noncoding",
    "splice_region_variant":    "noncoding",
    "intron_variant":           "noncoding",
    "5_prime_UTR_variant":      "noncoding",
    "3_prime_UTR_variant":      "noncoding",
}


def parse_consequence(name):

    # Returns unknown if the data is not a string
    if not isinstance(name, str):
        return "unknown"

    # Defines m to be the search of the name parameter inputted
    m = HGVS_PROTEIN.search(name)

    # If the protein searched is none, that means the variant never made it as far as to make a protein, in other words, it was noncoding.
    if m is None:
        return "noncoding"

    # Uses tuple unpacking to use HGVS_PROTEIN and seperate it into groups (reference amino acid, codon number, and the alternate amino acid)
    ref_aa, codon_number, alt_aa = m.groups()

    # If the alternate amino acid is the same thing as the reference amino acid, that means it is synonymous (substitution did not change the amino acid)
    if alt_aa == "=" or alt_aa == ref_aa:
        return "synonymous" 

    # Ter is the termination codon, indicating a premature stop substitution has occured, a strong indicator of DCM.

    if alt_aa == "Ter":
        return "nonsense"

    return "missense"
    