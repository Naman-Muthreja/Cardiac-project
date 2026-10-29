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
    "MYBPC3": [(153, 256, "C1"), (257, 361, "M-motif"), (362, 452, "C2"), (453, 543, "C3"), (544, 633, "C4"), (645, 771, "C5"),
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

    # “ttn_high_controls” is defined to be these benign TTN nonsense variants, with "psi_dcm" >= 0.9
    #  while "ttn_controls" includes all benign TTN nonsense variants, without needing to have "psi_dcm" > 0.9.
    ctx["ttn_high_controls"] = int((high & matched["label"].eq("Benign")).sum())
    ctx["ttn_controls"] = int((is_ttn_nonsense & matched["label"].eq("Benign")).sum())

    # Keys and rk (cross validation rows) are defined using variant IDs , and high_keys are the keys with label "high"
    keys = matched["chrom"].astype(str) + ":" + matched["pos"].astype(str) + matched["ref"] + ">" + matched["alt"]
    high_keys = set(keys[high])
    rk = rows["chrom"].astype(str) + ":" + rows["pos"].astype(str) + rows["ref"] + ">" + rows["alt"]

    # AUC-ROC score of the saved cross-validation predictions for TTN stop codons in high use exons
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

            # For each fold (not near edge), returns the positions of fold's training variants of the same gene-consequence pair
            for i in np.where((fold == f) & ~near_edge)[0]:
                same_kind = pos[fit & (cell == cell[i])]

                # The span of the gene the model saw
                span = pos[fit & (gene == gene[i])]

                # The distance to the gene-consequence matching training variant
                distance = np.abs(same_kind - pos[i]).min() if len(same_kind) else np.inf

                # "Is_far" determines if the variant is > 3000 bases away from the nearest training variant or outside the data
                is_far[(cv_seed, f, keys[i])] = bool(distance > far or pos[i] < span.min() or pos[i] > span.max())

    # IDs for the cross-validation rows
    row_keys = cv_rows["chrom"].astype(str) + ":" + cv_rows["pos"].astype(str) + cv_rows["ref"] + ">" + cv_rows["alt"]

    # Constructs an array that flags "far" variants, given cross-valdation seed, fold, and keys
    flag = np.array([is_far[(cv_seed, f, k)] for cv_seed, f, k in zip(cv_rows["cv_seed"], cv_rows["fold"], row_keys)])

    # Finds the AUC-ROC score for near_auc and far_auc (>3000 bases away). Does this by cell, meaning by groups of the
    # same gene-consequence training variant
    near_auc = bench.fold_auc(cv_rows[~flag], "model", by_cell = True, min_rows = 5, min_class = 2)
    far_auc = bench.fold_auc(cv_rows[flag], "model", by_cell = True, min_rows = 5, min_class = 2)


    return near_auc, far_auc

# Finds how trustable a clinical report is for a variant by comparing the saved cross-validation predictions against
# the true labels
def calibration(ctx, cell, score):

    # Keeps only rows of the same kind (gene and consequence)
    rows = ctx["cv_rows"][ctx["cv_rows"]["cell"] == cell]

    # Puts variants into 5 different bins, using the floor function. The bins range from a range from 0% - 20%
    # pathogenicity scores to 80% - 100%, and selects the bin for the current variant
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

        # Counts the amount of coding variants in the exon
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

    # Iterates through the codons, checking to see if its the end, and +1 is added because range() excludes stopping values
    for k in range(1, len(named) + 1):

        # Detects when the range (including the TTN regions) ends
        if k == len(named) or named[k] != named[first]:

            # If the finished range has a real label, save it
            if named[first]:
                runs.append((first + 1, k, named[first]))
            
            # Tells the code when the next range begins
            first = k
            
    # Returns the appended data with the range of amino acid sequences
    return runs

# Converts a number to percentage text
def pct(v):

    # Rounds extreme predictions that would round to 100 percent or 0 percent normally instead of claiming certainity
    if v >= 0.990:
        return ">99%"
    if v <= 0.01:
        return "<1%"
    return f"{100 * v:.0f}%"

# Turns each group of data into text to be used for the clinical report
def describe(group, raw, typ, row, names):
    gene = row["gene"]

    if group == "Which TTN exon this is":

        # Uses recursion to call back on itself and find additional groups
        parts = [describe(g, raw, typ, row, names) for g in ["Heart use of this TTN exon", "Titin region", "TTN isoforms", "Exon number and NMD"]]

        # Undoes the natural log and addition of 1 with e^x - 1, to return the original splice-site distance values
        # > 0.5 is so that the intron dist is non-zero (rounding errors can make it close to 0)
        if np.expm1(raw["tx_intron_dist"]) > 0.5:
            parts.append(describe("Splice-site distance", raw, typ, row, names))
        return ". ".join(parts)

    # Returns the intron distance if applicable 
    if group == "Splice-site distance":
        d, e = np.expm1(raw["tx_intron_dist"]), np.expm1(raw["tx_exon_edge_dist"])
        if d > 0.5 and gene == "TTN":
            return (f"Not inside an exon of the main TTN transcript ({d:,.0f} letters from the nearest one)")

        # Converts the splice donor side to clinical text and transcribes the distance to the exon
        if d > 0:
            side = " (splice donor side)" if "+" in names["c"] else (" (splice acceptor side)" if "-" in names["c"] else "")
            return f"In an intron, {d:,.0f} letters from the edge of exon {names['exon']}{side}"

        # Coverts the case where the exon distance to the edge is exactly 0 into clinical text
        if e == 0:                                                                          
            return f"At the very edge of exon {names['exon']} (its first or last letter), right next to the splice site"
        return f"Inside exon {names['exon']}, {e:,.0f} letters from its edge"

    # Returns the position in the protein, and compares it to the typical benign control
    if group == "Position in the protein":
        return f"{pct(raw['tx_protein_frac'])} of the way along the protein (typical control: {pct(typ['tx_protein_frac'])})" 

    if group == "Position along the gene":
        # If the relative position is outside of the scale, that means the variant has a stretch of the gene that
        # is outside what the training variants cover 
        if raw["rel_pos"] < 0 or raw["rel_pos"] > 1:
            return (f"Outside the stretch of the gene that the training variants cover, so the model has never seen a "
                    f"variant here. The typical control: {pct(typ['rel_pos'])}")

        # Returns the relative position along that the variants cover, and compares it to the typical benign variant
        return (f"{pct(raw['rel_pos'])} of the way along the stretch of the gene that the training variants cover"
                f" (typical control: {pct(typ['rel_pos'])})")

    if group == "Exon number and NMD":
        # Finds the # exon out of total exons (like Exon 242 out of 324)
        text = f"Exon {names['exon']} of {names['n_exons']}"

        # Adds a last exon tag for the last exon
        if raw["tx_last_exon"] > 0:
            text += ", the last exon"

        # Shows if NMD was escaped, which results in a shortened protein
        if raw["tx_nmd_escape"] > 0:
            text += "; a stop codon here escapes nonsense-mediated decay"
        return text

    # Includes how used the TTN exon is for the variant (using PSI scores)
    if group == "Heart use of this TTN exon":
        return (f"Heart TTN messages include this exon {pct(raw['ttn_psi_dcm'])} of the time in DCM hearts and "
                f"{pct(raw['ttn_psi_gtex'])} in healthy hearts (typical control: {pct(typ['ttn_psi_dcm'])})")

    # Returns the specific TTN isoform if the variant is in group "TTN_isoforms"
    if group == "TTN isoforms":
        names_in = [n for n, c in [("N2BA", "ttn_in_n2ba"), ("N2B", "ttn_in_n2b"), ("N2A", "ttn_in_n2a"), ("novex-3", "ttn_in_novex3")] if raw[c] > 0]
        return "Exon is part of: " + (", ".join(names_in) if names_in else "none of the four tracked isoforms")

    # Returns the Titin region of the variant
    if group == "Titin region":
        region = [n for n in ["A-band", "I-band", "Z-disk", "M-band"] if raw[f"ttn_{n}"] > 0]
        return f"Codes for the {region[0]} part of titin" if region else "Outside the four main titin regions"

    # Finds the amino-acid change properties, by comparing the reference amino acid with the alternate amino acid
    if group == "Amino-acid change":
        a, b = row.get("ref_aa"), row.get("alt_aa")

        # If a value has a mismatch
        if not isinstance(a, str) or not isinstance(b, str) or a == b or "*" in (a, b):


            # If it's a mismatched value and the consequence is missense, the codon is split across an exon edge,
            # meaning the amino acid features remain as 0 because they could not be read.
            if row.get("consequence") == "missense":                                       
                return (f"{names['p'] or 'Missense'}: the codon is split across an exon edge, so the amino-acid "
                        f"features could not be read and were left at 0")
            return "Not a missense change"

        # Returns amino acid properties, multiplication is to undo the reduction scales conducted in model_extra_features
        return (f"{THREE[a]} to {THREE[b]}: Grantham {215 * raw['aa_grantham']:.0f} of 215, BLOSUM62 {4 * raw['aa_blosum']:+.0f}, "
                f"hydropathy {9 * raw['aa_hydro']:+.1f}, volume {170 * raw['aa_volume']:+.0f} cubic angstroms")

    if group == "DNA letters":

        # Translates the reference and alternate letters to the negative strand
        ref_c, alt_c = row["ref"].translate(COMPLEMENT), row["alt"].translate(COMPLEMENT)

        # Finds and rounds the intron distance
        d = round(float(np.expm1(raw["tx_intron_dist"])))

        # Finds the donor side, since the donor side marks the start of an intron
        side = "+" if "+" in names["c"] else "-"   

        # +1 means the G of GT, +2 means T of GT, -2 means A of AG, -1 means G of AG for negative strand
        consensus = {("+", 1): ("G", "GT donor"), ("+", 2): ("T", "GT donor"), ("-", 1): ("G", "AG acceptor"), ("-", 2): ("A", "AG acceptor")}


        if d in (1, 2) and consensus[(side, d)][0] == ref_c:
            letter, signal = consensus[(side, d)]

            # Returns the reference letter, alternate letter, and the signal that indicates the end of the exon
            return (f"{row['ref']}>{row['alt']} on the genome, {ref_c}>{alt_c} on the gene's own strand (complemented). It replaces the {letter} of "
                    f"the {signal} signal that marks the exon edge")

        return (f"{row['ref']}>{row['alt']} on the genome, {ref_c}>{alt_c} on the gene's own strand (complemented). Next to a splice site the "
                f"exact letters matter, while elsewhere they mostly reflect which changes are common")

    if group == "Mutation-rate signals":

        # Transitions are between the same chemical family (two purines or two pyrimidines) while transversions are between
        # different chemical families (a purine and pyrimidine)
        kind = "A transition" if raw["transition"] > 0 else "A transversion"

        # A CpG site is 10x more likely to cause a mutation, so it is marked if present
        kind += " at a CpG site" if raw["cpg"] > 0 else ", not at a CpG site"

        # Cautionary message
        return (f"{kind}. These describe how often a change like this happens, not necessarily how harmful it is")

    if group == "DNA sequence (CNN branch)":
        return "Pattern the convolutional branch read in the 201-letter DNA window around the mutation"
    return "DNA Sequence group not founded"

# Flags variants based on rules, limits, or cautions
def flags(row, raw, ig, ctx):
    cell = f"{row['gene']}/{row['consequence']}"
    out = []

    # Flags if the variant is nonsense and TTN with high heart usage
    if row["gene"] == "TTN" and row["consequence"] == "nonsense" and raw["ttn_psi_dcm"] >= 0.9:

        # Finds the number of benign TTN nonsense variants that also have PSI score > 0.9 to serve as a comparison group
        # for TTN nonsense variants that have PSI score > 0.9.
        out.append(("limit", "Known TTN limit",
                    f"Known limit. {ctx['ttn_high_controls']} of {ctx['ttn_controls']} benign controls in my data "
                    f"also carry a stop codon in a TTN exon the heart uses at least 90% of the time. Inside that "
                    f"group the model separates patients from controls only slightly better than chance "
                    f"(AUC {ctx['ttn_high_auc']:.2f})"))

    # Finds the distance to the intron. expm1 is to undo the ln + 1 used for data compression
    d = round(float(np.expm1(raw["tx_intron_dist"])))

    # Flags if the distance to the exon is 1 or 2 (Canonical splice site). ACMG cites it as very strong evidence for pathogenicity,
    # because it changes the boundary between an intron and exon, resulting in LOF (Loss of Function) often through NMD
    if row["gene"] != "TTN" and 0 < d <= 2:
        out.append(("rule", "Canonical splice site",
                    "Canonical splice site (1 or 2 letters from the exon). Loss of function here is a standard "
                    "ACMG rule (PVS1)."))

    # Gets the positions of the training variants
    positions = ctx["fit_pos"].get(cell, np.array([]))

    # Finds the nearest position for all positions, and takes the minimum.
    nearest = np.abs(positions - int(row["pos"])).min() if len(positions) else np.inf

    # Flags if the variant is outside of the range of training variants, or > 3000 bases away from the nearest training variant
    # of the same gene and consequence
    if raw["rel_pos"] < 0 or raw["rel_pos"] > 1 or nearest > 3000:
        where = ("Beyond the range of training variants" if (raw["rel_pos"] < 0 or raw["rel_pos"] > 1)
                 else f"{nearest:,.0f} letters from the nearest training variant of this kind")

        # Returns the unfamiliar region and how well the cross-validation ranked variants that were this far, compared to variants
        # that were close
        out.append(("limit", "Unfamiliar region",
                f"Unfamiliar region: this variant is {where}. In cross-validation the model ranked variants this far "
                f"from its training data with AUC {ctx['far_auc']:.2f}, against {ctx['near_auc']:.2f} near it"))

    # Finds the number of fit variants inside each group
    n_fit = ctx["fit_counts"].get(cell,0)

    # If the variant being examined has less than 30 training variants of the same gene-consequence group, flag it with
    # a caution message.
    if n_fit < 30:
        out.append(("caution", "Few training variants",
                    f"Caution: Only {n_fit} training variants of this kind ({cell})."))

    # Gets the pathogenicity scores for each of the 3 models trained
    each = [float(ig[f"score_model_{k}"]) for k in (1, 2, 3)]

    # Flags if the most confident model and least confident model differ by a pathogenicity score of 0.2, since
    # they have the same NN architecture, so a big disagreement can mean uncertainty in predictions
    if max(each) - min(each) > 0.2:
        out.append(("caution", "Models disagree",
                    f"The three models disagree by {max(each) - min(each):.2f}."))

    # Add up the IG credits for each feature group, and finds the highest singular reason for the attribution score
    groups = {g: float(sum(ig[c] for c in cols)) for g, cols in groups_for(row["gene"]).items()}
    top = max(groups, key = lambda g: abs(groups[g]))

    # If the highest attribution score is significant and is "Mutation-rate signals", I flag it and return the score
    # without that feature because it reflects how often changes like this happened in the data, depending on other variants,
    # instead of using pathogenicity/benignity signals in the variant.
    if top == "Mutation-rate signals" and abs(groups[top]) > 0.05:
        without = float(ig["score"]) - groups[top]
        out.append(("caution", "Mutation-rate shortcut",
                 f"The biggest reason is a mutation-rate signal ({100 * groups[top]:+.0f} points). It reflects how often "
                 f"a change like this happens and how variants were collected, not harm. If this reason was removed, the score"
                 f" would be about {pct(min(max(without, 0), 1))}."))

    #  Looks up other tools' scores
    key = f"{row['chrom']}:{int(row['pos'])}{row['ref']}>{row['alt']}"
    if key in ctx["tools"].index and row["consequence"] == "missense":
        score = float(ig["score"])

        # Defines "tools" (AlphaMissense, REVEL, CADD)
        tools = {"AlphaMissense": ("alphamissense", ctx["tools"].loc[key, "alphamissense"]),
                 "REVEL": ("revel", ctx["tools"].loc[key, "revel"])}

        # If my model confidently flags a variant as pathogenic, try to find a tool that says it is benign
        if score >= 0.7:
            against = [f"{name} {v:.2f} ({bench.describe(t, v)})" for name, (t, v) in tools.items()
                       if v == v and bench.describe(t, v).startswith(("BP4", "likely benign"))]

            # If a tool does indeed find a variant as benign and my model finds it to be pathogenic, write a cautionary message
            if against:

                # "Leans" fixes the grammar if one tool disagrees, otherwise it lists all the tools that disagree
                out.append(("caution", "Other tools disagree",
                            f"Other tools disagree: {' and '.join(against)} {'lean' if len(against) > 1 else 'leans'} "
                            f"benign, while this model scores {pct(score)}."))

        # If my model confidently flags a variant as benign, try to find a tool that says it is pathogenic
        if score <= 0.3:
            against = [f"{name} {v:.2f} ({bench.describe(t, v)})" for name, (t, v) in tools.items()
                       if v == v and bench.describe(t, v).startswith(("PP3", "likely pathogenic"))]

            # If a tool does indeed find a variant as pathogenic and my model finds it to be benign, write a cautionary message
            if against:

                # "Leans" fixes the grammar if one tool disagrees, otherwise it lists all the tools that disagree
                out.append(("caution", "Other tools disagree",
                            f"Other tools disagree: {' and '.join(against)} {'lean' if len(against) > 1 else 'leans'} "
                            f"pathogenic, while this model scores {pct(score)}."))

    # If the variant is in a group of similar variants that have a within_stratum AUC-ROC score of less than 0.7 (like ‘MYBPC3 missense”), 
    # flag that the model has low reliability for those kinds of variants.
    auc = ctx["cell_auc"].get(cell, np.nan)
    if not np.isnan(auc) and auc < 0.7:
        out.append(("caution", "Low reliability for this kind",
                    f"For {cell} variants the model's cross-validated AUC is only {auc:.2f}."))

    # Returns "out", which includes all flags
    return out

# A color pallete for different parts of the picture being made using SVG, with red to indicate disease, blue
# to indicate benignity, the normal variant being colored in black, the typical control being in dark gray, and background
# marks in light gray.
DISEASE, BENIGN, INK, MUTED, RULE = "#B3261E", "#1A5FB4", "#16213A", "#5B6577", "#D9DEE5"

# Makes a bar graph from 0 to 100
def scale_svg(score, baseline, width = 640):

     # Converts a score to a x position on the canvas, with 20 units of margin on the left, and 6 positions to the right for every  
     # 0.01 increase in "v"
     x = lambda v: 20 + v * (width - 40)

     # Score on a scale of 1 to 100, a rounded rectangle with gradient color from benign to pathogenic, a vertical line to show
     # the control (benign) score, labels with clamps that make the middle at least 70 and at most 570,
     # a dot of where the current variant sits on that scale, and labels for the axis.
     return f"""<svg viewBox="0 0 {width} 74" width="100%" role="img" aria-label="Score {pct(score)} on a 0 to 100% scale">
      <defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="{BENIGN}"/><stop offset="1" stop-color="{DISEASE}"/></linearGradient></defs>
      <rect x="20" y="30" width="{width - 40}" height="10" rx="5" fill="url(#g)" opacity="0.28"/>
      <line x1="{x(baseline):.1f}" x2="{x(baseline):.1f}" y1="24" y2="46" stroke="{MUTED}" stroke-width="2" stroke-dasharray="3 3"/>
      <text x="{min(max(x(baseline), 70), width - 70):.1f}" y="62" text-anchor="middle" class="svg-small">typical control {pct(baseline)}</text>
      <circle cx="{x(score):.1f}" cy="35" r="9" fill="{INK}"/>
      <text x="{min(max(x(score), 70), width - 70):.1f}" y="17" text-anchor="middle" class="svg-strong">this variant {pct(score)}</text>
      <text x="20" y="62" class="svg-small">0%</text><text x="{width - 20}" y="62" text-anchor="end" class="svg-small">100%</text>
      </svg>"""

# Shows a column chart of different IG credits, with scores for each feature to compare how important different features were
def ledger_svg(baseline, items, final, width = 640):

        # Leaves an empty strip at the top, defines the area of the plot, and space for the left for arrow marks
        top, plot_h, left = 24, 150, 100

        # Tallest red and tallest blue columns are found, with defaults as fallbacks
        up = max([v for _, v in items if v > 0], default = 0)
        down = max([-v for _, v in items if v < 0], default = 0)

        # Defines the scale and the zero point
        scale = plot_h / (up + down)
        zero = top + up * scale

        # Sets equally sized slots for each feature being analyzed, and the height of the canvas
        slot = (width - left) / len(items)
        height = top + plot_h + 86

        # Adds a label, flatline, and arrows above and below that line to represent "toward disease" and "toward benign".
        parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="How much each reason moves the score">',
             f'<line x1="0" x2="{width}" y1="{zero:.1f}" y2="{zero:.1f}" stroke="{INK}" stroke-width="1"/>',
             f'<text x="0" y="{zero - 6:.1f}" class="svg-small">↑ toward disease</text>',
             f'<text x="0" y="{zero + 14:.1f}" class="svg-small">↓ toward benign</text>']

        # Iterates over each index and name-value pair
        for k, (name, value) in enumerate(items):

            # find the middle, and define the width of each column to be 55% of each slot, but never more than 46, so it still
            # looks like a column
            middle = left + (k + 0.5) * slot
            bar_w = min(slot * 0.55, 46)

            # Scales the height of each bar proportionate to its magnitude, with a minimum height to prevent
            # some smaller bars from hiding.
            bar_h = max(abs(value) * scale, 1.5)

            # If the attribution score is > 0, then the bar should extend above midline, and below 0 it should have
            # its peak be the midline, and extend downwards
            bar_y = zero - bar_h if value > 0 else zero

            # Color fills in the graph with gray, and is red if the value is > 0, otherwise benign for value < 0
            color = "#999999" if name == "Everything else" else (DISEASE if value > 0 else BENIGN)

            # Labels are 6 above the top edge for pathogenic, and 14 below the bottom edge for benign
            label_y = bar_y - 6 if value > 0 else bar_y + bar_h + 14

            # Positive if attribution score is positive, else negative
            sign = "+" if value > 0 else "−"

            # Splits the feature name based on each space into a list of words, to split each feature name into 2 lines
            # so it doesn't bleed into the next group
            words = name.split()
            half = (len(words) + 1) // 2

            # Makes the bars, using left edge (half the column's width subtracted from middle), height, label, color, and values
            parts.append(f'<rect x="{middle - bar_w / 2:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>'
                     f'<text x="{middle:.1f}" y="{label_y:.1f}" text-anchor="middle" class="svg-strong">{sign}{abs(100 * value):.0f}</text>'
                     f'<text x="{middle:.1f}" y="{top + plot_h + 30}" text-anchor="middle" class="svg-small">{html.escape(" ".join(words[:half]))}</text>'
                     f'<text x="{middle:.1f}" y="{top + plot_h + 44}" text-anchor="middle" class="svg-small">{html.escape(" ".join(words[half:]))}</text>')

        # Adds up all of the bars' values
        total = sum(v for _, v in items)
        parts.append(f'<text x="{(left + width) / 2:.1f}" y="{height - 8}" text-anchor="middle" class="svg-strong">'
                 f'Typical control {pct(baseline)}   +   all reasons {"+" if total > 0 else chr(8722)}{abs(100 * total):.0f} points   =   this variant {pct(final)}</text>')

        # End tag, and joins the string into one
        parts.append("</svg>")
        return "".join(parts)

# Adds how far along the protein a variant is, with the positions of important domains as well
def protein_svg(gene, codon, n_codons, ctx, width = 640):

    # Finds the TTN regions, if gene is TTN, otherwise finds the other domains
    track = ctx["ttn_regions"] if gene == "TTN" else DOMAINS[gene]

    # Finds the fraction of the way from the first codon to last codon the variant's codon is, and turns that value
    # into an x poisition on the canvas
    x = lambda c: 20 + (c - 1) / max(n_codons - 1, 1) * (width - 40)

    # Shades of colors for each of TTN's regions
    shades = {"Z-disk": "#C9D3E0", "I-band": "#E2E8EF", "A-band": "#B7C4D4", "M-band": "#9FB0C5"}

    # Makes a thin light-grey bar for the whole chains, including all codons
    parts = [f'<svg viewBox="0 0 {width} 86" width="100%" role="img" aria-label="Where the variant sits in the {gene} protein">',
             f'<rect x="20" y="40" width="{width - 40}" height="8" fill="{RULE}" rx="4"/>']

    labelled = set()

    # For the first codon, last codon, and name, fill with the correct shades of colors, and seperates the regions into 
    # positions
    for a, b, name in track:
        fill = shades.get(name, "#C3CEDC")
        parts.append(f'<rect x="{x(a):.1f}" y="34" width="{max(x(b) - x(a), 1):.1f}" height="20" fill="{fill}" rx="3"/>')

        # Checks if the bar is greater than 34 (otherwise the label is bigger), and if the gene isn't labeled yet If so,
        # appends the label to the middle position of the bar, and adds the name to labelled
        if x(b) - x(a) > 34 and name not in labelled:
            parts.append(f'<text x="{(x(a) + x(b)) / 2:.1f}" y="70" text-anchor="middle" class="svg-small">{name}</text>')
            labelled.add(name)

    # Adds a pin to the correct position on the map, and the number of codons at the end of the protein map too. Also adds the labels
    # labels off the codon numbers, with atleast 60 units away from the left edge and 60 from the right edge, to prevent things 
    # from getting cut out
    parts.append(f'<line x1="{x(codon):.1f}" x2="{x(codon):.1f}" y1="14" y2="58" stroke="{INK}" stroke-width="2"/>'
                 f'<circle cx="{x(codon):.1f}" cy="12" r="5" fill="{INK}"/>'
                 f'<text x="{min(max(x(codon), 60), width - 60):.1f}" y="8" text-anchor="middle" class="svg-small svg-bold">codon {codon}</text>'
                 f'<text x="20" y="84" class="svg-small">1</text><text x="{width - 20}" y="84" text-anchor="end" class="svg-small">{n_codons}</text></svg>')

    # Glues the parts and returns it
    return "".join(parts)

# CSS block responsible for taking the information in the clinical report and changing how it looks, including 
# font, font size, coloring, padding, page height, italicization, bolding, outlines, and more
CSS = """
:root { --ink:#16213A; --muted:#5B6577; --rule:#D9DEE5; --panel:#F4F6F9; --disease:#B3261E; --benign:#1A5FB4; --amber:#8A5A12; }
* { box-sizing: border-box; }
body { margin: 0; background: #E9EDF2; color: var(--ink); font: 14px/1.5 "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif; }
.page { width: 8.5in; min-height: 11in; margin: 24px auto; background: #fff; padding: 0.55in 0.6in; box-shadow: 0 1px 3px rgba(22,33,58,.18); }
h1, h2 { font-family: Georgia, "Iowan Old Style", "Times New Roman", serif; font-weight: 600; letter-spacing: 0; }
h1 { font-size: 26px; line-height: 1.2; margin: 0; }
h1 .gene { font-style: italic; }
h2 { font-size: 16px; margin: 22px 0 6px; padding-top: 10px; border-top: 1px solid var(--rule); }
.top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 18px; color: var(--muted); font-size: 12px; }
.top strong { color: var(--ink); font-size: 13px; }
.notice { border: 1px solid var(--amber); color: var(--amber); padding: 2px 8px; border-radius: 3px; font-weight: 600; }
.ids { color: var(--muted); margin: 4px 0 0; }
.call { display: grid; grid-template-columns: 1.1fr 1fr; gap: 22px; margin-top: 16px; align-items: start; }
.big { font-family: Georgia, serif; font-size: 44px; line-height: 1; font-variant-numeric: tabular-nums; }
.verdict { font-size: 15px; margin: 6px 0 4px; }
.classes { display: flex; height: 10px; border-radius: 5px; overflow: hidden; margin: 8px 0 4px; }
.legend { font-size: 12px; color: var(--muted); }
.panel { background: var(--panel); border-radius: 6px; padding: 10px 14px; font-size: 13px; }
.panel p { margin: 4px 0; }
.svg-small { font: 11px "Segoe UI", Helvetica, Arial, sans-serif; fill: var(--muted); }
.svg-strong { font: 600 12px "Segoe UI", Helvetica, Arial, sans-serif; fill: var(--ink); }
.svg-small, .svg-strong { paint-order: stroke; stroke: #fff; stroke-width: 3px; stroke-linejoin: round; }
.svg-bold { font-weight: 700; }
table { width: 100%; border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; }
th { text-align: left; color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--rule); padding: 4px 6px; }
td { border-bottom: 1px solid var(--rule); padding: 5px 6px; vertical-align: top; }
td.num { text-align: right; white-space: nowrap; }
.up { color: var(--disease); font-weight: 600; } .down { color: var(--benign); font-weight: 600; }
.flag { border-left: 2px solid var(--ink); padding: 2px 10px; margin: 8px 0; font-size: 13px; }
.flag strong { display: block; }
a { color: var(--ink); }
.foot { margin-top: 22px; color: var(--muted); font-size: 11px; border-top: 1px solid var(--rule); padding-top: 8px; }
@media print { body { background: #fff; } .page { margin: 0; box-shadow: none; width: auto; min-height: 0; } h2, .call, table { break-inside: avoid; } }
"""

# Uses HTML to work out the values from the data and then place it into one finished HTML page
def render(row, ig, raw, names, ctx, warnings, truth = None, report_id = "", today = ""):

    # Finds the type of variant and its typical control, and the associated scores
    gene, cons = row["gene"], row["consequence"]
    cell = f"{gene}/{cons}"
    typ = ctx["typical_raw"][cell]
    score, base = float(ig["score"]), float(ig["baseline_score"])

    # Uses groups_for from explain.py to categorize the 49 features into 12 distinct categories, then adds the IG credits for each
    groups = {g: float(sum(ig[c] for c in cols)) for g, cols in groups_for(gene).items()}

    # Sorts the data from highest to lowest to find the six highest attribution scores
    shown = [(g, v) for g, v in sorted(groups.items(), key = lambda kv: -abs(kv[1])) if abs(v) >= 0.005][:6]

    # Takes everything that is not in the 6 highest categories, and places it into category "Everything else", then finds
    # the IG scores for that
    rest = sum(groups.values()) - sum(v for _, v in shown)
    items = shown + ([("Everything else", rest)] if abs(rest) >= 0.005 else [])

    # Finds the gap between what the real attribution scores add up to and what the Completeness Axiom says it should add up to.
    gap = abs(sum(groups.values()) - (score - base))

    # The verdict of what the model predicts, like "HCM", as well as each individual probability
    p_hcm, p_dcm, p_ben = float(ig["p_HCM"]), float(ig["p_DCM"]), float(ig["p_Benign"])
    disease = "HCM" if p_hcm >= p_dcm else "DCM"

    # Translates the pathogenicity scores into words
    if score >= 0.8:
        verdict = f"Leans disease-causing ({disease})"
    elif score >= 0.6:
        verdict = f"Leans toward disease-causing ({disease}), with uncertainty"
    elif score > 0.4:
        verdict = "Uncertain: the model cannot call this one"
    else:
        verdict = "Leans benign"

    # Looks up the cross-validated AUC-ROC score and calls the calibration function to place the score in a range
    auc = ctx["cell_auc"].get(cell, np.nan)
    cal = calibration(ctx, cell, score) 

    # Calibration text saying among these scores, this many were pathogenic
    cal_text = (f"Among held-out cross-validation predictions for this kind of variant that scored {100 * cal[0]:.0f}% to "
                f"{100 * cal[1]:.0f}%, {pct(cal[2])} were pathogenic ({cal[3]:,} predictions of {cal[4]:,} variants, with "
                f"the benchmark being half pathogenic by design).") if cal else ""

    # Defines the key and then finds the tool scores using that key
    key = f"{row['chrom']}:{int(row['pos'])}{row['ref']}>{row['alt']}"
    tool_rows = ""
    tools = ctx["tools"].loc[key] if key in ctx["tools"].index else pd.Series({t: np.nan for t in bench.TOOLS})

    # For each tool, get the score, show "none" if the value is N/A, and also find the description of the tool's score,
    # such as "Likely pathogenic"
    for tool, label, fmt in [("revel", "REVEL", "{:.3f}"), ("alphamissense", "AlphaMissense", "{:.3f}"), ("cadd", "CADD (PHRED)", "{:.1f}")]:
        v = tools.get(tool, np.nan)
        shown_v = "none" if pd.isna(v) else fmt.format(v)
        tool_rows += f"<tr><td>{label}</td><td class='num'>{shown_v}</td><td>{html.escape(bench.describe(tool, v))}</td></tr>"

    detail_rows = ""

    # For each score and feature, if the feature is everything else, move on, otherwise find the correct color and 
    # display its attribution score and description. 
    # For example, “Splice-site letters”, “In an intron, two letters from the edge of exon 4”, “+32 pts” is a possible row.
    for g, v in items:
        if g == "Everything else":
            continue
        cls = "up" if v > 0 else "down"
        detail_rows += (f"<tr><td>{html.escape(g)}</td><td>{html.escape(describe(g, raw, typ, row, names))}</td>"
                        f"<td class='num {cls}'>{'+' if v > 0 else '-'}{abs(100 * v):.1f} pts</td></tr>")

    # Flags any concerns, via the previously defined flag() function, making sure to bold the reason, like “Canonical splice site"
    kind_name = {"limit": "Known limit", "rule": "Existing rule", "caution": "Caution"}
    flag_html = "".join(f"<div class='flag'><strong>{kind_name[kind]}: {html.escape(short)}</strong>{html.escape(text)}</div>"
                        for kind, short, text in warnings)
    if not flag_html:
        flag_html = "<p>No automatic warnings for this variant.</p>"

    # If the label (like "HCM") is already known (like for demo variants), return it just in the report
    # to see how the model did
    truth_html = f"<p><strong>Label in my data (hidden from the model):</strong> {html.escape(truth)}</p>" if truth else ""

    # Gets the codon and how far along the protein the variant is, using protein_svg()
    codon = names.get("codon")
    protein = protein_svg(gene, codon, ctx["n_codons"][gene], ctx) if codon else ""

    # Finds the protein change if possible (intronic variants don't change the protein)
    p_part = f" ({html.escape(names['p'])})" if names["p"] else ""

    # Finds each of the 3 model's scores
    models_text = ", ".join(pct(float(ig[f"score_model_{k}"])) for k in (1, 2, 3))

    # Sets up the HTML page to make sure it is compatible, and also inserts the CSS block. html.escape() is used to properly
    # convert strings.
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(gene)} {html.escape(names['c'])} report</title><style>{CSS}</style></head><body><main class="page">

<!--Top part showing the report, ID, and that its for research use only-->
<div class="top"><span><strong>Cardiac variant explanation report</strong><br>Report {html.escape(report_id)}, {html.escape(today)}</span>
<span class="notice">Research use only.</span></div>

<!-- Gene and genomic information -->
<h1><span class="gene">{html.escape(gene)}</span> {html.escape(names['c'])}{p_part}</h1>
<p class="ids">{html.escape(MANE[gene][1])}, GRCh38 chr{row['chrom']}:{int(row['pos']):,} {row['ref']}&gt;{row['alt']}, {html.escape(cons)}, exon {names['exon']} of {names['n_exons']}</p>

<section class="call">
   
   <!-- Main pathogenicity score and verdict -->
  <div>
    <div class="big">{pct(score)}</div>
    <p class="verdict"><strong>{html.escape(verdict)}</strong></p>
    <p class="legend">Chance of being disease-causing (HCM or DCM), averaged over 3 models ({models_text}).</p>

    <div class="classes" aria-hidden="true"><div style="width:{100 * p_hcm:.1f}%;background:{DISEASE}"></div><div style="width:{100 * p_dcm:.1f}%;background:#6E2433"></div><div style="width:{100 * p_ben:.1f}%;background:{BENIGN}"></div></div>
    <p class="legend">HCM {pct(p_hcm)}, DCM {pct(p_dcm)}, benign {pct(p_ben)}</p>
  </div>    
  
  <!-- How reliable the model is for similar variants, and its cross-validation AUC-ROC score-->
  <div class="panel">
    <p><strong>How reliable is the model for {html.escape(cell)} variants?</strong></p>
    <p>Cross-validated AUC {auc:.2f} (0.50 = coin flip, 1.00 = perfect), from {ctx['fit_counts'].get(cell, 0):,} training variants of this kind.</p>
    <p>{html.escape(cal_text)}</p>
    {truth_html}
  </div>

  {scale_svg(score, base)}

   <!--Completeness axiom in words, to show how the score went from an average benign variant to its actual score->
   <h2>Why the model says this</h2>
   <p class="legend">Typical control = the average benign training variant of this gene and type, in this variant's own DNA window. Integrated Gradients splits the difference between it ({pct(base)}) and this variant ({pct(score)}) into one piece per reason; the pieces add up to the difference (gap {100 * gap:.2f} points). Each column is one reason in points: red columns push toward disease, blue columns toward benign.</p>
   {ledger_svg(base, items, score)}
   <table><tr><th>Reason</th><th>What is true for this variant</th><th class="num">Effect</th></tr>{detail_rows}</table>

   <!-- Where the model sits in the protein, how much to trust it, using flag(), and what other tools say.->
   <h2>Where it sits in the protein</h2>
{protein}
<h2>How much to trust this</h2>
{flag_html}
<h2>What other tools say</h2>
<table><tr><th>Tool</th><th class="num">Score</th><th>Published category</th></tr>{tool_rows}</table>
<p class="legend">REVEL, CADD, AlphaMissense categories: In cross-validation on missense variants my model tied all three; on TTN stop codons it beat CADD.</p>


</section>

"""