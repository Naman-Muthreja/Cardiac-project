"""
Revel_cadd_alphamissense_bench fetches REVEL, CADD, and AlphaMissense tool scores for my variants, and then calculates data metrics for comparison with my model. 
Annotation in genomics means to attatch extra information to a variant previously unknown, hence the script's name.
"""
import os
import time                                         
import numpy as np                                   
import pandas as pd                                  
import requests                                      
from sklearn.metrics import roc_auc_score          

# Set names for the paths and tools used
HERE = os.path.dirname(os.path.abspath(__file__))
SCORES_PATH = os.path.join(HERE, "..", "data", "processed", "tool_scores.csv")
TOOLS = ["revel", "alphamissense", "cadd"]
MYVARIANT_FIELDS = {"revel": "dbnsfp.revel.score", "alphamissense": "dbnsfp.alphamissense.score", "cadd": "dbnsfp.cadd.phred"}


# Sets the ID in HGVS genomic notation, uses chr to refer to a chromosome (MyVariant.info needs it) and g. to refers to a genomic coordinate
# > refers to a change in DNA Sequence (like from G to C)
def variant_ids(df):
    return [f"chr{c}:g.{int(p)}{r}>{a}" for c, p, r, a in zip(df["chrom"], df["pos"], df["ref"], df["alt"])]

# Returns the highest score, which is the most pathogenic-looking, rather than the lowest score, which would bias towards
# benignity. Finds the most severe transcript for each list.
def highest(value):
    if isinstance(value, list):
        numbers = [v for v in value if isinstance(v, (int, float))]
        return max(numbers) if numbers else np.nan
    if isinstance(value, (int, float)):
        return float(value)
    return np.nan

# Digs through a dotted path, "like dbnsfp.revel.score"
def dig(record, dotted):
    for key in dotted.split("."):
        if not isinstance(record, dict):
            return None
        record = record.get(key)
    return record

# Fetches the scores for each tool
def fetch_tool_scores(df, path = SCORES_PATH, batch_size = 500, refresh = False):

    # HGVS genomic notation
    ids = variant_ids(df)

    # If the path exists, find the IDs of the variants saved vs missing
    if os.path.exists(path) and not refresh:
        saved = pd.read_csv(path)
        missing = set(ids) - set(saved["id"])
        assert not missing, f"{len(missing)} variants are not in {path}."
        print(f"Read saved tool scores from {path}")
        return saved.set_index("id").loc[ids].reset_index()

    downloaded = []

    # For all ids, goes batch_size at a time. Tries 3 times per batch, with a timeout. 
    for start in range(0, len(ids), batch_size):

        # A chunk starts at "start" and ends after adding "batch_size" number of ids.
        chunk = ids[start:start + batch_size]
        for attempt in range(3):
            try:
                r = requests.post("https://myvariant.info/v1/variant", timeout = 120,
                                  data = {"ids": ",".join(chunk), "assembly": "hg38",
                                          "fields": ",".join(MYVARIANT_FIELDS.values())})
                r.raise_for_status()

                # Turns the answer into Python objects
                hits = {h["query"]: h for h in r.json()}

                # Leaves loop if the attempt worked
                break
            # Except for these errors, check if the attempt is the third one (indexing at 0). If so, stop the program for that variant and wait for 5 seconds.
            except (requests.exceptions.RequestException, ValueError, KeyError):
                if attempt == 2:
                    raise
                time.sleep(5)

        # Flags the error of a variant being completely skipped (notfound doesn't count as skipped)
        assert set(chunk) <= set(hits), "MyVariant skipped some variants"

        # Appends the variant_id, score for a variant, and the tool-score pairs (Ex: revel : 0.71).
        # Includes each model's prediction score for a variant (higher meaning more pathogenic).
        for variant_id in chunk:
            downloaded.append({
                "id": variant_id,
                 **{
                 tool: highest(dig(hits[variant_id], field))
                 for tool, field in MYVARIANT_FIELDS.items()
                 }
             })
        time.sleep(1)

    # Scores is the DataFrame of all the rows, inside of the "path" file.
    scores = pd.DataFrame(downloaded)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    scores.to_csv(path, index = False)

    # Prints the count and returns the table
    print(f"Saved {len(scores)} rows to {path}")
    return scores

# Makes sure that the constructed data is numerically correct before finding the data metrics
def check_tool_scores(df, scores):

    # Checks to see if the IDs in the score table match in order to the IDs from the variant_ids function, so that the correct scores are assigned to the right variants.
    assert list(scores["id"]) == variant_ids(df),"score IDs are not in the same order as the variants"

     # Combines variant information and scores into one dataframe
    d = pd.concat([df[["chrom", "pos", "ref", "alt", "gene", "consequence", "label"]] .reset_index(drop=True), scores[TOOLS].reset_index(drop=True)], axis=1)

     # Finds the amount of variants that actually have scores, and finds that as a fraction.
    coverage = d.groupby(["gene", "consequence"])[TOOLS].agg(lambda s: f"{s.notna().sum()}/{len(s)}")
    print("Coverage (scored / total):")
    print(coverage.to_string())

     # Finds non-missense variants and calculates the amount of "blanked" (unfilled scores) variants for REVEL and Alphamissense, which only score on missense variants.
    for tool in ["revel", "alphamissense"]:
        other = (d["consequence"] != "missense") & d[tool].notna()
        if other.any():
            print(f"{tool}: blanked {int(other.sum())} score(s) on variants that are not missense in the MANE transcript:")

            # Prints the gene, consequence, and label for non-missense variants, and nan for the tools that don't score missense variants (about to be erased).
            print(d.loc[other, ["gene", "consequence", "label", tool]].to_string())
            d.loc[other, tool] = np.nan

     # Makes sure that each tool's scores are in the correct ranges (ex: REVEL having score 3 would mean a problem in the dataset)
    assert d["revel"].dropna().between(0, 1).all(), "REVEL outside 0 to 1"
    assert d["alphamissense"].dropna().between(0, 1).all(), "AlphaMissense outside 0 to 1"
    assert d["cadd"].dropna().between(0, 99).all(), "CADD outside 0 to 99"

    # Finds the AUC-ROC scores for each of the tools (missense variants work for all 3) as a direction check to make sure that the code is working.
    mis = d[d["consequence"] == "missense"]
    for tool in TOOLS:
        part = mis.dropna(subset = [tool])

        # Labels pathogenic scores as 1 and benign scores as 0, for each tool, then checks it with the answer key for the binary AUC-ROC
        auc = roc_auc_score((part["label"] != "Benign").astype(int), part[tool])
        print(f"Direction check, {tool} on {len(part)} missense: AUC {auc:.3f}")
        assert auc > 0.5, f"something is wrong"
    return d

# VUS version of check_tool_scores
def tool_scores_for(df, path):
    scores = fetch_tool_scores(df, path = path)
    assert list(scores["id"]) == variant_ids(df), "scores are not in the same order as the variants"
    d = pd.concat([df[["chrom", "pos", "ref", "alt", "gene", "consequence"]].reset_index(drop = True),
                   scores[TOOLS].reset_index(drop = True)], axis = 1)
    d.loc[d["consequence"] != "missense", ["revel", "alphamissense"]] = np.nan
    assert d["revel"].dropna().between(0, 1).all() and d["alphamissense"].dropna().between(0, 1).all()
    assert d["cadd"].dropna().between(0, 99).all()
    print("Coverage (scored / total):")
    print(d.groupby(["gene", "consequence"])[TOOLS].agg(lambda s: f"{s.notna().sum()}/{len(s)}").to_string())
    return d

# Quickly finds the weighted auc
def weighted_auc(true_labels, model_scores, weights):
        return float(roc_auc_score(true_labels, model_scores, sample_weight = weights))

def pieces(rows, by_cell = False):

    # Groups the rows cross validation seed and fold if by_cell is False, otherwise also includes gene and consequence, then returns the group
    keys = ["cv_seed", "fold"] + (["gene", "consequence"] if by_cell else [])
    return [(key[0], idx) for key, idx in rows.groupby(keys).indices.items()]

# Seperates the data into folds for cross-validation (to reduce variability in the data)
# One important thing to keep in mind is that the cross-validation does not need a redefinition of how to assign the variants to a fold, or how big a fold is,
# because I am using the cv_predictions csv built from my model in train.py as an input.
def fold_auc(rows, score, by_cell=False, weight=None, min_rows=10, min_class=3, parts=None):

    # Calls pieces if the grouping information is not already supplied.
    parts = pieces(rows, by_cell) if parts is None else parts

    # Finds the labels, prediction scores, and weights.
    y_all = (rows["label"].to_numpy() != "Benign").astype(int)
    s_all = rows[score].to_numpy(dtype=float)
    w_all = np.ones(len(rows)) if weight is None else weight

    totals = {}

    for seed, idx in parts:
        y, s, w = y_all[idx], s_all[idx], w_all[idx]

        # Makes sure the groups aren't too small (less than min_rows, pathogenic or benign less than min_class)
        if (w.sum() < min_rows or (w * y).sum() < min_class or (w * (1 - y)).sum() < min_class):
            continue

        # Retrieves running totals (starting at 0.0)
        total, n = totals.get(seed, (0.0, 0.0))

        # Calculates the weighted binary AUC-ROC for each seed
        totals[seed] = (total + weighted_auc(y, s, w) * w.sum(), n + w.sum())

# Returns a confidence interval 
def paired_bootstrap(rows, score_a, score_b, by_cell=False, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    parts = pieces(rows, by_cell)

    # Defines a genomic key for every variant used
    keys = (rows["chrom"].astype(str) + ":" +rows["pos"].astype(str) + rows["ref"] + ">" + rows["alt"] )
    codes, unique = pd.factorize(keys)
    diffs = []

    # Randomly picks variants n_boot amount of times, compares the two models provided by finding the difference using fold_auc, and returns a 95% confidence range
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(unique), len(unique)), minlength = len(unique))[codes].astype(float)
        diffs.append(fold_auc(rows, score_a, by_cell, w, parts = parts) - fold_auc(rows, score_b, by_cell, w, parts = parts))
    return np.nanpercentile(diffs, [2.5, 97.5])
    
def harness_self_test(preds):

    rng = np.random.default_rng(1)

    # Makes a copy of preds so that the original is not altered, assigns pathogenicity as 1 and benignity as 0
    t = preds.copy()
    y = (t["label"] != "Benign").astype(float)

    # Answer key, flipped (opposite of answer key), and constant guessing are created
    t["perfect"], t["flipped"], t["constant"] = y, -y, 0.0

    # Random guessing
    t["random"] = rng.random(len(t))

    # Lookup averages the amount of variants being pathogenic, and assigns that average as a score to be given to every single variant. 
    # For example, if there are 8 pathogenic variants and 12 benign variants, the average will be 8/20 = 0.4, and the AUC-ROC will test a score of 0.4 on all variants
    cell_rate = t.groupby(["gene", "consequence"])["label"].transform(lambda s: (s != "Benign").mean())
    t["lookup"] = cell_rate

    # Model's pathogenicity score is 1- [the probability of benign]
    t["model"] = 1 - t["p_benign"]

    # Answer key should have an AUC-ROC score of 1.0; flipped should have an AUC-ROC score of 0.0; constant guessing should have an AUC-ROC score of 0.5; random guessing
    # should have an AUC-ROC score of roughly 0.5, lookup should have an AUC-ROC score of 0.5.
    checks = [("perfect score", fold_auc(t, "perfect"), 1.0, 1e-9),
              ("flipped score", fold_auc(t, "flipped"), 0.0, 1e-9),
              ("constant score", fold_auc(t, "constant"), 0.5, 1e-9),
              ("random score", fold_auc(t, "random"), 0.5, 0.1),  # Allowed 0.1 error because random guessing can vary in correctness
              ("lookup, within stratum", fold_auc(t, "lookup", by_cell = True), 0.5, 1e-9)]

    # For these values in checks, print out the name, score retrieved, and wanted score.
    for name, got, want, tol in checks:
        print(f" {name:22s} AUC {got:.3f} (expect {want:.3f})")
        assert abs(got - want) <= tol, f"harness self-test failed: {name}" 

# Subsets to be used for comparison of my model to REVEL/CADD/AlphaMissense
SUBSETS = [
    ("Missense, MYH7 + MYBPC3", lambda d: d["consequence"] == "missense", False),
    ("Missense, MYH7 only", lambda d: (d["consequence"] == "missense") & (d["gene"] == "MYH7"), False),
    ("TTN nonsense", lambda d: (d["gene"] == "TTN") & (d["consequence"] == "nonsense"), False),
    ("MYBPC3 noncoding (splice)", lambda d: (d["gene"] == "MYBPC3") & (d["consequence"] == "noncoding"), False),
    ("All the tool scores, within cells", lambda d: d["gene"].notna(), True),
]

def benchmark(preds, tool_table, n_boot = 1000):

    # Unique key for each variant
    key = lambda d: d["chrom"].astype(str) + ":" + d["pos"].astype(str) + d["ref"] + ">" + d["alt"]

    # Uses key as an identifier for the variants, for each tool being evaluated, and also stops the function in case of a duplicate variant.
    tools = tool_table.assign(key=key(tool_table)).set_index("key")[TOOLS]
    assert tools.index.is_unique, "a variant appears twice in the tool table"

    # Uses key to make the rows, with the key column of rows being matched to the index of tools. I join from left to right.
    rows = preds.assign(key = key(preds))
    rows = rows.join(tools, on = "key", how = "left")

    # Makes sure that the number of rows did not change during the joining process
    assert len(rows) == len(preds), "the join changed the number of rows"
    missing = ~rows["key"].isin(tools.index)
    assert not missing.any(), f"{int(missing.sum())} predictions have no tool row"

    # Model pathogenicity scores = 1 - the probability of benignity
    rows["model"] = 1 - rows["p_benign"]

    results = []

    # Main loop that compares my model to REVEL/CADD/AlphaMissense
    for name, pick, by_cell in SUBSETS:
        sub = rows[pick(rows)]

        for tool in TOOLS:

            # Drops variants that don't have scores
            same = sub.dropna(subset=[tool])

            # Counts the number of pathogenic vs benign variants
            n_path = int((same.drop_duplicates("key")["label"] != "Benign").sum())
            n_ben = same["key"].nunique() - n_path

            # If the number of pathogenic or benign variants is 0, its due to the tool not giving scores for that type of class, or that there's too few pathogenic/benign variants
            if min(n_path, n_ben) < 10:
                verdict = "tool gives no scores here" if len(same) == 0 else "too few of one class"

                # Apppends these results before going to the next variant
                results.append({"subset": name, "tool": tool, "variants": same["key"].nunique(),
                    "pathogenic / benign": f"{n_path} / {n_ben}", "verdict": verdict})
                continue

            # Finds both the cross-validation AUC-ROC score for my model and the other tools
            tool_auc = fold_auc(same, tool, by_cell)
            model_auc = fold_auc(same, "model", by_cell)

            # Appends the results, with a flag, if the tool_auc or model_auc is NaN.
            if np.isnan(tool_auc) or np.isnan(model_auc):
                results.append({"subset": name, "tool": tool, "variants": same["key"].nunique(),       
                                "pathogenic / benign": f"{n_path} / {n_ben}", "verdict": "no AUC-ROC score was computed"})
                continue

            # Checks for statistical significance with a 95% confidence interval
            low, high = paired_bootstrap(same, "model", tool, by_cell, n_boot = n_boot)

            # If results aren’t NaN, I append the model AUC-ROC score, the other tools’ AUC-ROC score, the 95% confidence interval, pathogenicity divided by benign as a safety check,
            # the difference in AUC-ROC scores, and the verdict of whether my model did better or the tool, all to “results”, which I then return.
            results.append({"subset": name, "tool": tool, "variants": same["key"].nunique(),
                            "pathogenic / benign": f"{n_path} / {n_ben}", "tool AUC": round(tool_auc, 3), "model AUC": round(model_auc, 3),
                            "model - tool": round(model_auc - tool_auc, 3),
                            "95% CI": f"[{low:+.3f}, {high:+.3f}]",
                            "verdict": "model better" if low > 0 else ("tool better" if high < 0 else "tie")})
            
    return pd.DataFrame(results)

# Describes the variant for the clinical report
def describe(tool, score):
    if score is None or np.isnan(score):
        return "no score (only scores missense)" if tool != "cadd" else "no score"
    if tool == "revel":
        for cut, text in [(0.932, "PP3 strong"), (0.773, "PP3 moderate"), (0.644, "PP3 supporting")]:
            if score >= cut:
                return text
        for cut, text in [(0.003, "BP4 very strong"), (0.016, "BP4 strong"), (0.183, "BP4 moderate"), (0.290, "BP4 supporting")]:
            if score <= cut:
                return text
        return "indeterminate"
    if tool == "cadd":
        for cut, text in [(28.1, "PP3 moderate"), (25.3, "PP3 supporting")]:
            if score >= cut:
                return text
        for cut, text in [(0.15, "BP4 strong"), (17.3, "BP4 moderate"), (22.7, "BP4 supporting")]:
            if score <= cut:
                return text
        return "indeterminate"
    if score > 0.564:
        return "likely pathogenic"
    if score < 0.34:
        return "likely benign"
    return "ambiguous"