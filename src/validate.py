"""
validate.py is a validation script, conducting several checks, such as aggregate localization rate, which is how often
the variant's position is among the most credited DNA positions, and a Mann-Whitney U test (pass with p < 0.01) comparing
variants inside of these important domains vs without to see if the attribution is higher in a known protein domain like
the MYH7 motor domain, than outside of it. Also returns several other important comparison metrics to test out hypothesis 2.
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
    v["domain"] = [domain(transcripts, g, p) for g, p in zip(v["gene"], v["pos"])]
    v.to_csv(path, index = False)

    print(f"Saved {len(v)} held-out missense variants to {path} in ({time.time() - t0:.0f} seconds)")
    return v
    
# Determines whether motor-domain variants get more protein-level credit than the other MYH7 variants, by comparing the 
# IG credits in pairs, one selected from each group
def mann_whitney(a, b, label):

    # 'a' is the group of motor-domain variants, and b is the array of non-motor-domain variants, and both are 
    # converted to an array of decimals of their credits
    a, b = np.asarray(a, dtype = float), np.asarray(b, dtype= float)

    # If there is less than 10 in a group, the amount of variants is too low and meaningless to test anyways
    if min(len(a), len(b)) < 10:
        print(f"{label}: too few to test ({len(a)} vs {len(b)})")
        return {"test": label, "n_a": len(a), "n_b": len(b), "p": np.nan}

    # Uses scipy's mannwhitneyu() function to conduct the test; "greater" is for calculating a one-sided test (which side is greater?)
    u, p = mannwhitneyu(a, b, alternative = "greater")

    # Prints and returns the length of motor-domain vs non-motor-domain variants, the median scores of the motor domain vs rest
    # to illustrate the difference in credits, the amount of pairs where motor-domain variants had a higher number of credits,
    #  and the probability that the test occured through random chance.
    print(f"{label}: n {len(a)} vs {len(b)}, median {np.median(a):+.3f} vs {np.median(b):+.3f}, U = {u:.0f}, one-sided p = {p:.2e}")
    return {"test": label, "n_a": len(a), "n_b": len(b), "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "U": float(u), "p": float(p)}

# tests out hypothesis 2 of my research plan, that the highest attribution scores by Integrated Gradients applied to the model
# will correlate with clinically important domains, such as the myosin motor domain in MYH7, rather than genomic noise.
def hypothesis_2(v, path = os.path.join(RESULTS, "h2_results.csv")):

    # Specifically focuses on MYH7 due to its prevalent motor domain
    myh7 = v[v["gene"] == "MYH7"]

    # Motor domain vs everything else
    motor, rest = myh7[myh7["domain"] == "Motor domain"], myh7[myh7["domain"] != "Motor domain"]

    # Prints out the MYH7 variants by domain (coiled rod, motor domain, etc)
    print("MYH7 missense by domain:", myh7["domain"].value_counts().to_dict())

    # Prints out the number of counts of each label (HCM, DCM, Benign) in the motor domain
    print("Label mix: motor", motor["label"].value_counts().to_dict(), "| rest", rest["label"].value_counts().to_dict())

    # Protein-level credit, amino acid change, and position in the protein, are compared between the motor variants 
    # and rest of the variants
    tests = [mann_whitney(motor["protein_level"], rest["protein_level"], "PRIMARY: protein-level credit, motor vs rest")]
    tests.append(mann_whitney(motor["Amino-acid change"], rest["Amino-acid change"], "amino-acid change alone"))
    tests.append(mann_whitney(motor["Position in the protein"], rest["Position in the protein"], "position in the protein alone"))


    # Pathogenic is all non-benign labels, and benign includes the benign labels
    pathogenic = lambda d: d[d["label"] != "Benign"]
    benign = lambda d: d[d["label"] == "Benign"]

    # Protein level for pathogenic and benign are both appended to tests
    tests.append(mann_whitney(pathogenic(motor)["protein_level"], pathogenic(rest)["protein_level"], "pathogenic only"))
    tests.append(mann_whitney(benign(motor)["protein_level"], benign(rest)["protein_level"], "benign only (control)"))

    # Filters for the MYBPC3 gene, and appends to "tests" the protein level between MYBPC3 C domain vs non C domain variants
    mybpc3 = v[v["gene"] == "MYBPC3"]
    tests.append(mann_whitney(mybpc3[mybpc3["domain"] != "outside"]["protein_level"],
                              mybpc3[mybpc3["domain"] == "outside"]["protein_level"], "MYBPC3 C-domains vs outside"))

    # All tests in one dataframe, the probability of results being chance has to be less than 0.01 to be considered "passed"
    table = pd.DataFrame(tests)
    table.to_csv(path, index = False)
    passed = table.loc[0, "p"] < 0.01
    print(f"Hypothesis 2 as pre-registered: {'PASSES' if passed else 'does not pass'} (p = {table.loc[0, 'p']:.2e}, needs < 0.01)")
    return table

# Returns an array of the per-position DNA credit, adding up the absolute value of the 4 channels
def position_credit(dna):
    return np.abs(np.asarray(dna)).sum(axis = 1)

def localization_rate(models, df, F, typical, top_k = 5, steps = 64, batch_size = 64):

    # Prepares the sequence of the mutant sequence and then the reference DNA window and a feature baseline
    X, _ = train.prepare_tensors(df)
    x_ref, f_base = explain.baselines(df, F, typical)
    out = {}

    # For each reference baseline (own window without the SNV, and all 0s)
    for name, x_base in [("reference", x_ref), ("zero", torch.zeros_like(X))]:

        # Calculates integrated gradients on the data and gets the per-position DNA credit
        dna, feat, p, p0, delta = explain.integrated_gradients(models, X, F, x_base, f_base, steps = steps, batch_size = batch_size)
        credit = position_credit(dna.numpy())

        # The number of positions that ranked higher than the variant, summing up the amount of "True" values returned
        rank = (credit > credit[:, [100]]).sum(axis = 1)            

        # Calculates the rate of variants ranking in the top 5 positions
        rate = float((rank < top_k).mean())

        # Finds the percent of IG credits made up by bases near the variant base swap position (within 5 bases)
        near = float((credit[:, 95:106].sum(axis = 1) / np.maximum(credit.sum(axis = 1), 1e-12)).mean())

        # Calculates how much attribution comes from the DNA branch vs the features (as a percent)
        dna_share = float((np.abs(dna.numpy()).sum(axis = (1, 2)) /
                           (np.abs(dna.numpy()).sum(axis = (1, 2)) + np.abs(feat.numpy()).sum(axis = 1))).mean())

        # Prints and returns the metrics found
        out[name] = {"rate": rate, "chance": top_k / 201, "share_within_5": near, "dna_share_of_credit": dna_share}
        print(f"Localization, {name} baseline: variant position in the top {top_k} of 201 for {rate:.1%} of {len(df)} "
              f"variants (chance {top_k / 201:.1%}); {near:.1%} of the DNA credit is within 5 letters of it; "
              f"the DNA branch holds {dna_share:.1%} of all credit")
    return out

# Runs the code's functions, saves the results
def run_all(matched, models, setup):
    v = cv_explanations(matched)
    h2 = hypothesis_2(v)
    typical = explain.typical_benign(setup["F_fit"], setup["fit_df"])
    loc = localization_rate(models, setup["test_df"], setup["F_test"], typical)
    gates_path = os.path.join(RESULTS, "ig_gates.json")
    gates = json.load(open(gates_path)) if os.path.exists(gates_path) else {}
    summary = {"h2_primary_p": float(h2.loc[0, "p"]), "h2_passes": bool(h2.loc[0, "p"] < 0.01),
               "localization": loc, "gates": gates}
    json.dump(summary, open(os.path.join(RESULTS, "validation.json"), "w"), indent = 2)
    print("Saved results/validation.json")
    return summary