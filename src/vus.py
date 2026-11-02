"""
vus.py downloads variants of uncertain significance (VUS) from ClinVar, with the genes of variants my model was
trained on It builds each one's 201-letter windows from the GRCh38 reference genome and classifies the variant as
"HCM", "DCM", or "Benign", and then creates a clinical report explaining the classifications.
"""

import os
import time
import numpy as np
import pandas as pd
import requests

from model_extra_features import annotate
import os
import time
import numpy as np
import pandas as pd
import requests
from model_extra_features import annotate

# File paths
HERE = os.path.dirname(os.path.abspath(__file__))
EXTERNAL = os.path.join(HERE, "..", "data", "external")

# Electronic utilities that includes ClinVar (searches for the VUS variants)
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# Defines the gene regions, with codon number and genomic positions
REGIONS = {"MYH7": ("14", 23380000, 23440000), "MYBPC3": ("11", 47330000, 47360000), "TTN": ("2", 178520000, 178820000)}

# What to search ClinVar for
QUERIES = [("MYH7", "missense variant"), ("MYBPC3", "missense variant"), ("MYBPC3", "intron variant"),
           ("MYBPC3", "splice donor variant"), ("MYBPC3", "splice acceptor variant"), ("TTN", "nonsense")]


