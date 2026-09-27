"""
get_annotations.py  takes in two tables that give the model biological knowledge to be able to make better predictions, 
specifically by downloading ex
"""
import json
import os
import re
import time
import requests

# Defines the gene regions for each gene (chrosomome and pos)
GENE_REGIONS = {
    "chr14": (23380000, 23440000),     # MYH7
    "chr11": (47330000, 47360000),     # MYBPC3
    "chr2":  (178520000, 178820000),   # TTN
}

# Loops through each gene region, downloads the UCSC exon/transcript annotation JSON for that genomic interval, 
# and saves each response.
def download_exon_maps(out_dir):
    for chrom, (start, end) in GENE_REGIONS.items():
        url = (f"https://api.genome.ucsc.edu/getData/track?genome=hg38;track=ncbiRefSeq;"
               f"chrom={chrom};start={start};end={end}")

        # Gets the url, raises a status error in case something goes wrong
        r = requests.get(url, timeout = 60)
        r.raise_for_status()

        # Makes a custom file path, writes it down, prints the amount of characters saved
        with open(os.path.join(out_dir, f"tx_{chrom}.json"), "w") as f:
            f.write(r.text)
        print(f"saved tx_{chrom}.json ({len(r.text)} characters)")

# Turns a messay html response into a clean usable dictionary 
def parse_exon_page(html):

    # Removes HTML tags and replaces anything like <> with a space. 
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)

    # Searches meta-transcript for 4 different coordinate systems (Hg19, Hg38,LRG, Meta-transcript)
    coords = re.search(r"Meta-Transcript Start End Start End Start End Start End "
                       r"([\d,]+) ([\d,]+) ([\d,]+) ([\d,]+)", text)
    if coords is None:
        return None

    # Searches for psi, region, and isforms using re.search(), which searches the text, using *? for minimum characters and
    # \d+ for whole numbers to return matching text values. Captures different values by looking at what the text says. 
    # Ex: capturing X (\d+) in X DCM percent.
    psi = re.search(r"is (\d+)% in DCM .*? and (\d+)% in GTEx", text) # DCM first group, GTEX second group
    region = re.search(r"occurs in the ([\w\-/ ]+?) region of the titin", text)
    isoforms = re.search(r"META N2BA N2B N2A NOVEX-1 NOVEX-2 NOVEX-3 (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)", text)

    # Returns data read into a clean dictionary
    return {
        "s38": int(coords.group(3).replace(",", "")),
        "e38": int(coords.group(4).replace(",", "")),
        "psi_dcm": int(psi.group(1)) if psi else None,
        "psi_gtex": int(psi.group(2)) if psi else None,
        "region": region.group(1) if region else None,
        "tx": list(isoforms.groups()) if isoforms else None,
    }

# Downloads the ttn exons and returns it to the out directory
def download_ttn_exons(out_dir, first_exon = 1, last_exon = 364, pause = 0.15):
    exons = {}

    # +1 in used here so that last_exon is also included.
    for i in range(first_exon, last_exon + 1):

        # Reads the html data from cardiob to cleanly parse the TTN data.
        html = requests.get(f"https://www.cardiodb.org/titin/titin_exon.php?id={i}", timeout = 20).text
        info = parse_exon_page(html)

        # If info has a value, set it to be the correct index, and run this for each one. This makes sure
        # that each dictionary is labeled with an exon number.
        if info is not None:
            exons[i] = info
        time.sleep(pause)

    # w = write, saves all the collected exon information into a singular file
    with open(os.path.join(out_dir, "ttn_exons.json"), "w") as f:

        # Converts python dict into JSON format and writes it, printing #variants saved
        json.dump(exons, f)
        print
        return exons

# Makes the file path, and calls the download_exon_maps and download_ttn_maps functions
# Runs only if executed directly, to prevent time wastage in imports.
if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "annotations")
    os.makedirs(out, exist_ok = True)
    download_exon_maps(out)
    download_ttn_exons(out)
