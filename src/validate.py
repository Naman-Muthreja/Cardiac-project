"""
validate.py is a validation script, conducting several checks, such as aggregate localization rate, which is how often
the variant's position is among the most credited DNA positions, and a Mann-Whitney U test (pass with p < 0.01) 
to see if the attribution is higher in a known protein domain like the MYH7 motor domain, than outside of it.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
import train
import explain
import report
from encoding import encode_dataset
from model_extra_features import load_transcripts, build

# File path for the current location, and where to save the results
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

def cv_explanations(matched, steps = 128, batch_size = 64, path = os.path.join(RESULTS, "h2_variants.csv")):

    # If the path already exists, reads the cross-validation explanations from it
    if os.path.exists(path):
        print(f"Read saved cross-validation explanations from {path}")
        return pd.read_csv(path)

    # Starts the time, calls make_regions_split to split the variants into train/validate/test 3000-letter genomic position blocks
    t0 = time.time()
    train_df, val_df, test_df, demo_df = train.make_regions_split(matched, seed = 49, verbose = False)

    # Concatenates the data for train, val, and test, all stacked
    data = pd.concat([train_df, val_df, test_df], ignore_index = True)

    # One hot encodes the data, X_all is the DNA windows, y_all is the labels
    X_all, y_all = train.prepare_tensors(data)
    pieces, worst = [], 0.0

    # For each cross_validation seed, calls on block_folds() to split the data into 5 distinct folds.
    for cv_seed in (0, 1, 2):
        fold, near_edge = train.block_folds(data, n_folds = 5, seed = cv_seed)

        # For each of the five folds, define fit variants to be all the variants not in the testing fold (roughly 4/5)
        for f in range(5):
            fit = np.where(fold!=f)[0]
            held = np.where((fold == f) & (~near_edge) & (data["consequence"] == "missense").to_numpy())[0]

            # Reset the indices of the data, and picks the training rows and held-out rows
            fit_df, held_df = data.iloc[fit].reset_index(drop = True), data.iloc[held].reset_index(drop = True)

            # Gets the features for the fit data and held data, and then trains the model with a new CV model
            F_fit, F_held = train.make_features_tensors(fit_df, [held_df])
            model = train.fit_one(X_all[fit], F_fit, y_all[fit], 80, 32, 7e-4, 3e-4, seed = cv_seed * 10 + f, device = "cpu")

            # Typical benign variant and baselines for the DNA and features
            typical = explain.typical_benign(F_fit, fit_df)
            x_base, f_base = explain.baselines(held_df, F_held, typical)

            # Calculates integrated gradients for the model
            dna, feat, p, p0, delta = explain.integrated_gradients([model], X_all[held], F_held, x_base,
                                                                    f_base,steps = steps, batch_size = batch_size)

           # Total credit (DNA + features) minus (score − baseline score) should be about 0. This keeps the worst gap.
            gap = (dna.sum(dim = (1, 2)) + feat.sum(dim = 1) - (p - p0)).abs().max().item()
            worst = max(worst, gap)

            # Gene_span includes the starting and ending position (lowest and highest)
            gene_span = {g: (d["pos"].min(), d["pos"].max()) for g, d in fit_df.groupby("gene")}
            names = list(build(held_df.head(2), gene_span = gene_span).columns)

            # Dataframe of the IG credit attributed to each feature
            credit = pd.DataFrame(feat.numpy(), columns = names)

            # Copys the important parts of the df into a new one to be adjustable
            out = held_df[["chrom", "pos", "ref", "alt", "gene", "label"]].copy()

            # Iterates over groups, and returns the IG credit for each one, converts to NumPy
            for group in ["Position in the protein", "Amino-acid change", "Position along the gene"]:
                out[group] = credit[explain.GROUPS[group]].sum(axis = 1).to_numpy()

            # Adds the score and cross-validation seed to output
            out["score"], out["cv_seed"] = p.numpy(), cv_seed
            pieces.append(out)

        # Print out the amount of variants saved, the worst completeness gap, and whether that gap passes
        print(f"cv seed {cv_seed} fold {f}: {len(held)} missense explained, completeness gap {gap:.4f}")
        print(f"Largest completeness gap over all folds: {worst:.4f}")
        assert worst < 0.01, "completeness failed"

        # Concatenates the rows together
        rows = pd.concat(pieces, ignore_index = True)

        # Defines the keys to read by and then groups the rows and averages the credits 
        # over the 3 cross-validations
        keys = ["chrom", "pos", "ref", "alt", "gene", "label"]
        v = rows.groupby(keys, as_index = False)[["Position in the protein", "Amino-acid change", "Position along the gene", "score"]].mean()

        # Adds two new columns row by row into a new column called "protein_level"
        v["protein_level"] = v["Position in the protein"] + v["Amino-acid change"]

        transcripts = load_transcripts()

        # Using coding indices, gets the position, the codon number, and uses it for the domain.
        def domain(transcripts, gene, pos):
            ci = transcripts[gene].coding_index.get(int(pos))
            if ci is None:
                return None
            codon = ci // 3 + 1  # Transcript map counts from 0, but codons count from 1

            return next((name for a, b, name in report.DOMAINS[gene] if a <= codon <= b), "outside")

        # Adds domain as a column by calling on it, saves v as a csv
        v["domain"] = [domain(g, p) for g, p in zip(v["gene"], v["pos"])]
        v.to_csv(path, index = False)

        print(f"Saved {len(v)} held-out missense variants to {path} in ({time.time() - t0:.0f} seconds)")
        return v
