import re
import time
import pandas as pd
import requests

WINDOW = 201
HALF = WINDOW // 2

GENE_TO_CLASS = {
    "MYH7": "HCM",
    "MYBPC3": "HCM",
    "TTN": "DCM",
}

LOW_CONFIDENCE_REVIEW = {"no assertion criteria provided", "no assertion provided"}