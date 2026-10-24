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
from train import make_regions_split, block_folds
from model_extra_features import load_transcripts, load_ttn_exons, MANE, COMPLEMENT, CODON_TABLE


# Converts each one letter abbreviation for an amino acid to a three letter short form
THREE = dict(A="Ala", R="Arg", N="Asn", D="Asp", C="Cys", Q="Gln", E="Glu", G="Gly", H="His", I="Ile", L="Leu",
             K="Lys", M="Met", F="Phe", P="Pro", S="Ser", T="Thr", W="Trp", Y="Tyr", V="Val")

# Converts "*" to Ter, to signify a stop codon
THREE["*"] = "Ter"

# Start and end amino acid positions of several important regions for the reports
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

    # If the genomic position is after the last coding letter, return the distance with "*"
    if k > last:
        return f"*{k - last}"

    # Returns the coding dna portion, adds 1 to match up with HGVS
    return str(t.coding_index[p] + 1)

# Builds each variant's names from my transcript map
def hgvs_names(df, transcripts = None):  

    # Finds "transcripts" if defined, or calls the function that loads it, and orders the reading frame
    transcripts = transcripts or load_transcripts()
    orders = {g: reading_order(t) for g, t in transcripts.items()}

    out = []

    for r in df.itertuples():

        # Finds the transcript, order, and position
        t, order = transcripts[r.gene], orders[r.gene]
        pos = int(r.pos)

        # Finds the exon number and distance to the exon (zero if exonic, non-zero if intronic) and the start and
        # end of the exons.
        exon, dist = t.locate(pos)
        start, end = t.exons[exon]

        p_text = ""

        # Translates to the complement base of the reference and alternate bases
        ref_c, alt_c = r.ref.translate(COMPLEMENT), r.alt.translate(COMPLEMENT)    

        # If the variant is an exon, return the coding coordinates in HGVS format
        if dist == 0:
            c_text = f"c.{c_coordinate(t, order, pos)}{ref_c}>{alt_c}"

        # Intron below the exon in genome numbers means that it is after it in reading order:
        elif pos <= start:                 
            c_text = f"c.{c_coordinate(t, order, start + 1)}+{dist}{ref_c}>{alt_c}"

        # Intron above the exon means that it is before it in reading order
        else:                              
            c_text = f"c.{c_coordinate(t, order, end)}-{dist}{ref_c}>{alt_c}"

        # If the position is in the coding sequence, get its index
        if pos in t.coding_index:
            ci = t.coding_index[pos]   
        
            # Frame is one of the three positions in a codon that contains the variant, and codon number is the index for codons
            frame, codon_no = ci % 3, ci // 3 + 1

            # Codon position is the position of each base in the 3-base codon, subtracts by frame to get the start of the codon
            codon_pos = t.coding[ci - frame: ci - frame + 3]

            # If the codon length is 3 and the distance to variant is <= 100, as it should be, reconstruct the codon and join
            # the codon_pos values together, with p - pos + 100 to center it on the variant. 
            if len(codon_pos) == 3 and all(abs(p - pos) <= 100 for p in codon_pos):
                codon = "".join(r.ref_sequence[p - pos + 100] for p in codon_pos).translate(COMPLEMENT)

                # Mutant is defined as the altered codon, with the alternate base at position "frame"
                mutant = codon[:frame] + alt_c + codon[frame + 1:]

                a, b = CODON_TABLE[codon], CODON_TABLE[mutant]

                # Converts the amino acid swap to HGVS format, with the codon number in the middle and p. to represent a protein-level
                # change.
                p_text = f"p.{THREE[a]}{codon_no}{'=' if a == b else THREE[b]}"

        # Tries to get the "name" field of the current row
        official = getattr(r, "name", None)

        # Extracts the official ClinVar protein information in case the HGVS protein information process above did not work
        if not p_text and isinstance(official, str) and "(p." in official: 
             p_text = "p." + official.split("(p.")[1].rstrip(")")

        # Simplified description if the protein description still could not be found, and the consequence was nonsense
        if not p_text and getattr(r, "consequence", "") == "nonsense" and pos in t.coding_index:
             p_text = f"p.(codon {t.coding_index[pos] // 3 + 1} to stop)"

        # Stores the results and returns the table
        out.append({ "c": c_text,"p": p_text,"exon": exon + 1,"n_exons": t.n_exons})
    return pd.DataFrame(out, index = df.index)

# Typical control and where the training variants are
def build_context(matched, setup, cv_preds, tools):

    # Gets all the training variants, and recalculates the raw features
    fit_df = setup["fit_df"]
    raw_fit = setup["raw"](fit_df)

    # Gets the gene and consequence, and filters for benign variants for "benign_fit"
    cell_fit = fit_df["gene"] + "/" + fit_df["consequence"]
    benign_fit = fit_df["label"].eq("Benign")

    # “Ctx” (context) includes the average of each feature column in benign training variants, sorted via gene and consequence
    ctx = {"typical_raw": {c: raw_fit[(cell_fit == c) & benign_fit].mean() for c in cell_fit.unique()},
           "fit_counts": cell_fit.value_counts().to_dict(),
           "fit_pos": {c: np.sort(part["pos"].to_numpy()) for c, part in fit_df.groupby(cell_fit)}}

    # Assigns pathogenicity prediction scores for each gene-consequence pair ("cell")
    rows = cv_preds.assign(model = 1 - cv_preds["p_benign"], cell = cv_preds["gene"] + "/" + cv_preds["consequence"])

    # Calculates the cross validation AUC-ROC score (with 5 folds) for each "cell"
    ctx["cell_auc"] = {c: bench.fold_auc(part, "model") for c, part in rows.groupby("cell")}

    # Defines the cross-validation rows to be "rows", then calculates near_far_auc
    ctx["cv_rows"] = rows
    ctx["near_auc"], ctx["far_auc"] = near_far_auc(matched, rows)

    # Finds raw features of all 2090 variants, needed for TTN's PSI.
    raw_features = setup["raw"](matched)

    # Finds all TTN and nonsense variants
    is_ttn_nonsense = matched["gene"].eq("TTN") & matched["consequence"].eq("nonsense")

    # "High" is when the TTN nonsense variant has a high PSI score, meaning the heart largely uses it
    high = is_ttn_nonsense & (raw_features["ttn_psi_dcm"] >= 0.9)

    # “ttn_high_controls” is defined to be these benign TTN nonsense variants, with "psi_dcm" > 0.9
    #  while "ttn_controls" includes all benign TTN nonsense variants, without needing to have "psi_dcm" > 0.9.
    ctx["ttn_high_controls"] = int((high & matched["label"].eq("Benign")).sum())
    ctx["ttn_controls"] = int((is_ttn_nonsense & matched["label"].eq("Benign")).sum())

    # Keys and rk (cross validation rows) are defined using HGVS notation, and high_keys are the keys with label "high"
    keys = matched["chrom"].astype(str) + ":" + matched["pos"].astype(str) + matched["ref"] + ">" + matched["alt"]
    high_keys = set(keys[high])
    rk = rows["chrom"].astype(str) + ":" + rows["pos"].astype(str) + rows["ref"] + ">" + rows["alt"]

    # Cross-validation on "high-keys", returns the fold AUC-ROC score
    ctx["ttn_high_auc"] = bench.fold_auc(rows[rk.isin(high_keys)], "model")

    # IDs for the tool table
    key_t = tools["chrom"].astype(str) + ":" + tools["pos"].astype(str) + tools["ref"] + ">" + tools["alt"]

    # Uses the other tools (REVEl, CADD, AlphaMissense) and sets indices
    ctx["tools"] = tools.assign(key = key_t).set_index("key")[bench.TOOLS]

    # Flags if a variant appears twice in the tool table
    assert ctx["tools"].index.is_unique, "a variant appears twice in the tool table"

    # Finds the regions of TTN and then returns ctx
    ctx["ttn_regions"] = ttn_protein_regions()
    return ctx

# Near_far_auc measures how well the model ranks variants near its training data vs far from it, where far means more than
# 3000 bases away from the nearest training variant of the same gene-consequence kind or outside the data entirely.
def near_far_auc(matched, cv_rows, far = 3000):

    # Splits the dataframe into train, val, test, and demo, and concats all dfs that aren't demo (to be used later)
    train_df, val_df, test_df, demo_df = make_regions_split(matched, seed = 49, verbose = False)
    data = pd.concat([train_df, val_df, test_df], ignore_index = True)          

    # Gene-consequence pairs ("cell"), position, gene, and keys are defined
    cell = (data["gene"] + "/" + data["consequence"]).to_numpy()
    pos, gene = data["pos"].to_numpy(), data["gene"].to_numpy()
    keys = (data["chrom"].astype(str) + ":" + data["pos"].astype(str) + data["ref"] + ">" + data["alt"]).to_numpy()

    is_far = {}

    # For each cross-validation seed, build the folds
    for cv_seed in sorted(cv_rows["cv_seed"].unique()):
         fold, near_edge = block_folds(data, n_folds = 5, seed = cv_seed)

         # Loops over folds
         for f in range(5):
            fit = fold != f

            # For each fold (not near edge), returns the positions of fold's training variants of the same gene
            for i in np.where((fold == f) & ~near_edge)[0]:
                same_kind = pos[fit & (cell == cell[i])]

                # The span of the gene the model saw
                span = pos[fit & (gene == gene[i])]

                # The distance to the same near-kind training variant
                distance = np.abs(same_kind - pos[i]).min() if len(same_kind) else np.inf

                # "Is_far" determines if the variant is > 3000 bases away from the nearest training variant or outside the data
                is_far[(cv_seed, f, keys[i])] = bool(distance > far or pos[i] < span.min() or pos[i] > span.max())

    # IDs for the cross-validation rows
    row_keys = cv_rows["chrom"].astype(str) + ":" + cv_rows["pos"].astype(str) + cv_rows["ref"] + ">" + cv_rows["alt"]

    # Constructs an array that flags "far" vairnats, given cross-valdation seed, fold, and keys
    flag = np.array([is_far[(cv_seed, f, k)] for cv_seed, f, k in zip(cv_rows["cv_seed"], cv_rows["fold"], row_keys)])

    # Finds the AUC-ROC score for near_auc and far_auc (>3000 bases away). Does this by cell, meaning by groups of the
    # same gene-consequence training variant
    near_auc = bench.fold_auc(cv_rows[~flag], "model", by_cell = True, min_rows = 5, min_class = 2)
    far_auc = bench.fold_auc(cv_rows[flag], "model", by_cell = True, min_rows = 5, min_class = 2)


    return near_auc, far_auc

# Finds how trustable a clinical report is for a variant by comparing the saved cross-validation predictions against
# the true labels
def calibration(ctx, cell, score):

      # Keeps only rows of the same kind (gnee and consequence)
      rows = ctx["cv_rows"][ctx["cv_rows"]["cell"] == cell]

      # Puts variants into 5 different bins, using the floor function. The bins range from a range from 0% - 20%
      # pathogenicity scores to 80% - 100%
      low = min(np.floor(score*5) / 5, 0.8)

      # Selects predictions based on the score ranges assigned by "low", and places them into variable "near"
      near = rows[(rows["model"] >= low) & (rows["model"] < low + 0.2 + 1e-9)]

      # If sufficient evidence is not there, return None
      if len(near) < 15:
        return None

      variants = (near["chrom"].astype(str) + ":" + near["pos"].astype(str) + near["ref"] + ">" + near["alt"]).nunique()

      # Returns the score range, the proportion of pathogenic variants, the number of prediction rows selected, and unique # of variants
      return low, low + 0.2, float((near["label"] != "Benign").mean()), len(near), variants

# Converts titin’s annotations (like “A-band” or “Z-disk”) into ranges of amino acid sequences that 
# belong to particular protein regions
def ttn_protein_regions():

    # Loads the transcripts and exons
    t = load_transcripts()["TTN"]
    exons = load_ttn_exons()

    base_region = []

    for start, end in t.exons:     

        # Reorders the positions from highest to lowest (negative gene strand)
        coding_here = sum(1 for p in range(end, start, -1) if p in t.coding_index)

        # Uses cardiodb to find the exons at the exact same place
        hit = exons[(exons["low"] <= end) & (exons["high"] > start)]

        # Finds the per-region label, and puts it on each coding base
        base_region += [hit["region"].iloc[0] if len(hit) else "none"] * coding_here

        # Splits into codon regions (by 3)
        codon_region = base_region[::3]

        # Important TTN regions are found, and then matched with the codons
        keep = {"Z-disk": "Z-disk", "near Z-disk": "Z-disk", "I-band": "I-band", "A-band": "A-band", "M-band": "M-band"}
        named = [keep.get(r) for r in codon_region]

        runs, first = [], 0

        # Runs for each variant until the very end
        for k in range(1, len(named) + 1):

            # The range stops running at the end (checks to see if it is at the end)
            if k == len(named) or named[k] != named[first]:

                # If the finished range has a real label, save it
                if named[first]:
                    runs.append((first + 1, k, named[first]))

                # [explain here tmrw]
                first = k

            # Retunrs the appended data with the range of amino acid sequences
            return runs