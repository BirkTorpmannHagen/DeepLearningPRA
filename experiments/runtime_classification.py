import itertools
import os.path

import numpy as np
import pandas
import pandas as pd
import seaborn
from matplotlib import pyplot
from matplotlib.lines import Line2D
from multiprocessing import Pool, cpu_count
from matplotlib.patches import Patch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import seaborn as sns
import matplotlib.pyplot as plt
from components import OODDetector
from itertools import product
from rateestimators import ErrorAdjustmentEstimator
from simulations import UniformBatchSimulator
from utils import load_pra_df, load_all, DSD_PRINT_LUT, DATASETWISE_RANDOM_LOSS, DATASETS, DSDS, THRESHOLD_METHODS, \
    SYNTHETIC_SHIFTS, BATCH_SIZES, load_all_biased, SAMPLERS, SAMPLER_LUT, DATASETWISE_RANDOM_CORRECTNESS


def cost_benefit_analysis():

    data = load_pra_df("Polyp", "knn", batch_size=1, samples=1000)
    oods = data[~data["shift"].isin(["ind_val", "ind_test", "train", "noise"])]["shift"].unique()
    cba_data = []
    print(oods)
    ood_val_set = "CVC-ClinicDB"
    ood_test_set = "EndoCV2020"
    with tqdm(total=11*2) as pbar:

        current_ood_val_acc = data[data["shift"]==ood_val_set]["correct_prediction"].mean()
        print(current_ood_val_acc)
        sim = UniformBatchSimulator(data, ood_test_shift=ood_test_set, ood_val_shift=ood_val_set, maximum_loss=0.5, estimator=ErrorAdjustmentEstimator, dsd_tnr=0.9, dsd_tpr=0.9)
        results = sim.sim(0.5, 600) #just to get the right parameters

        for acc in np.linspace(0, 1, 11):
            sim.detector_tree.ood_dsd_acc = acc
            sim.detector_tree.ood_ndsd_acc = acc
            sim.base_tree.update_tree()
            sim.detector_tree.update_tree()
            d_risk = sim.detector_tree.get_risk_estimate()
            ba = (sim.base_tree.dsd_tpr + sim.base_tree.dsd_tnr)/2
            cba_data.append({"Component":"Classifier", "Accuracy":acc, "Risk Estimate":d_risk})
            pbar.update(1)

        sim = UniformBatchSimulator(data, ood_test_shift=ood_test_set, ood_val_shift=ood_val_set, maximum_loss=0.5,
                                    estimator=ErrorAdjustmentEstimator, dsd_tnr=0.9, dsd_tpr=0.9)
        results = sim.sim(0.5, 600)  # just to get the right parameters
        for acc in np.linspace(0, 1, 11):
            sim.detector_tree.dsd_tnr = acc
            sim.detector_tree.dsd_tpr = acc
            sim.detector_tree.update_tree()
            d_risk = sim.detector_tree.get_risk_estimate()
            cba_data.append({"Component":"Event Detector", "Accuracy":acc, "Risk Estimate":d_risk})
            pbar.update(1)

    df = pd.DataFrame(cba_data).groupby(["Component", "Accuracy"]).mean().reset_index()

    sns.lineplot(df, x="Accuracy", y="Risk Estimate", hue="Component")
    plt.savefig("cba.pdf")
    plt.show()


def verdictwise_proportions(cal_idx=0, batch_size=1):
    df = load_all(batch_size, prefix="final_data")

    dfs_processed = []

    for dataset in df["Dataset"].unique():
        for feature in df["feature_name"].unique():
            if feature=="mahalanobis":
                continue
            df_dataset = df[(df["Dataset"]==dataset) & (df["feature_name"]==feature)].copy()
            try:
                ood_calibration_set = df_dataset[~df_dataset["shift"].isin(["ind_val", "train", "ind_test"])]["shift"].unique()[cal_idx%len(df_dataset[~df_dataset["shift"].isin(["ind_val", "train", "ind_test"])]["shift"].unique())]
            except:
                print(f"No OOD calibration set for {dataset} {feature}")
                continue
            train_data = df_dataset[(df_dataset["shift"]=="ind_val")|(df_dataset["shift"]==ood_calibration_set)]
            ood_detector = OODDetector(train_data, ood_val_shift=ood_calibration_set, threshold_method="val_optimal")
            print(f"Dataset: {dataset}, Feature: {feature}, OOD Calibration Set: {ood_calibration_set}, t={ood_detector.threshold}")
            df_dataset["verdict"] = df_dataset.apply(lambda row: ood_detector.predict(row), axis=1)
            df_dataset.replace(DSD_PRINT_LUT, inplace=True)
            df_dataset["verdict"] = df_dataset["verdict"].apply(lambda x: "OOD" if x==True else "IND")
            dfs_processed.append(df_dataset)
    df = pd.concat(dfs_processed)



    # Set bin edges globally

    g = sns.FacetGrid(df, row="Dataset", col="feature_name", sharex=False, sharey=False)

    # Plot manually in each facet
    def stacked_bar(data, color=None, **kwargs):
        # Compute bin edges locally for this subset of data
        bins = 20
        bin_edges = np.linspace(data["loss"].min(), data["loss"].quantile(0.99), bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        data["bin"] = pd.cut(data["loss"], bins=bin_edges, include_lowest=True)
        count_df = data.groupby(["bin", "verdict"]).size().unstack(fill_value=0)
        proportion_df = count_df.div(count_df.sum(axis=1), axis=0).fillna(0)

        bottom = np.zeros(len(proportion_df))
        for label in proportion_df.columns:
            plt.bar(bin_centers, proportion_df[label], bottom=bottom,
                    width=bin_edges[1] - bin_edges[0], label=label,
                    edgecolor='white', align='center', alpha=0.5)
            bottom += proportion_df[label].values

    def correctness_hist(data, color=None, ax=None, **kwargs):
        palette = sns.color_palette(n_colors=2)
        # Compute bin edges locally for this subset of data
        bins = 20
        bin_edges = np.linspace(data["loss"].min(), data["loss"].quantile(0.99), bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        data["bin"] = pd.cut(data["loss"], bins=bin_edges, include_lowest=True)
        count_df = data.groupby(["bin", "correct_prediction"]).size().unstack(fill_value=0)
        total_counts = count_df.sum(axis=1)/ count_df.sum(axis=1).sum()  # Normalize by total counts
        # total_counts = total_counts*2 # Scale to 2 for better visibility in the plot

        proportion_df = count_df.div(count_df.sum(axis=1), axis=0).fillna(0)
        if True in proportion_df.columns:
            y = proportion_df[True].values
        else:
            y = np.zeros(len(proportion_df))  # fallback if no correct predictions
        plt.plot(bin_edges[:-1], y, color="black", linestyle="--", label="Proportion correct")

    def verdict_hist(data, color=None, ax=None, **kwargs):
        palette = sns.color_palette(n_colors=2)
        # Compute bin edges locally for this subset of data
        bins = 20
        bin_edges = np.linspace(data["loss"].min(), data["loss"].quantile(0.99), bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        data["bin"] = pd.cut(data["loss"], bins=bin_edges, include_lowest=True)
        count_df = data.groupby(["bin", "correct_prediction"]).size().unstack(fill_value=0)
        y_true = count_df[True].values if True in count_df.columns else np.zeros(len(count_df))
        y_false = count_df[False].values if False in count_df.columns else np.zeros(len(count_df))
        # Normalize by total counts

        y_true = y_true / y_true.max()
        y_false = y_false / y_false.max()


        plt.step(bin_edges[:-1], y_true, where='post', color=palette[0], label="IND")
        plt.step(bin_edges[:-1], y_false, where='post', color=palette[1], label="OOD")


    def baseline(data, color=None, ax=None, **kwargs):
        dataset = data["Dataset"].unique()[0]
        if dataset!="Polyp":
            plt.axvline(DATASETWISE_RANDOM_LOSS[dataset], color="red", linestyle="--", label="Random Guessing")


    palette = sns.color_palette(n_colors=2)  # Set1 has two distinct colors
    g.map_dataframe(stacked_bar, palette="pastel")
    g.map_dataframe(correctness_hist)
    g.map_dataframe(verdict_hist)
    g.map_dataframe(baseline)
    # g.map_dataframe(sns.kdeplot, x="loss", color="black", legend=False, fill=False, clip = (0, 30))
    verdict_handles = [
        Patch(facecolor=palette[0], edgecolor='white', label="IND", alpha=0.5),
        Patch(facecolor=palette[1], edgecolor='white', label="OOD", alpha=0.5)
    ]
    hist_handles = [
        Line2D([0], [0], color=palette[0], lw=2, label="Proportion Correct "),
    ]
    all_handles = verdict_handles + hist_handles
    g.fig.legend(all_handles, [h.get_label() for h in all_handles], title="Legend", loc='lower center',
                 fontsize="large", ncol=4, bbox_to_anchor=(0.5, -0.01))

    # Combine and add
    all_handles = verdict_handles + hist_handles
    g.fig.legend(all_handles, [h.get_label() for h in all_handles], title="Legend", loc='lower center',
                 fontsize="large", ncol=4, bbox_to_anchor=(0.5, -0.01))

    g.set_axis_labels("loss", "Proportion")
    g.set_titles(col_template="{col_name}")

    for ax in g.axes.flat:
        dataset = ax.get_title().split(" = ")[-1].split("|")[0].strip()
        print(dataset)
        ax.set_xlim(0, df[df["Dataset"]==dataset]["loss"].quantile(0.99))
        ax.set_ylim(0, 1)


    # Check if the total number of facets is more than 3 and adjust accordingly
    plt.tight_layout()

    plt.savefig("proportions_all.pdf")
    plt.show()

def parallel_compute_ood_detector_prediction_accuracy(data_filtered, threshold_method, dataset, feature, ood_perf, perf_calibrated):
    data_dict = []
    for ind_val_fold in ["ind_val", "ind_test"]:
        for ood_val_fold in data_filtered["shift"].unique():
            data_copy = data_filtered.copy()
            if ood_val_fold in ["train", "ind_val", "ind_test"] or ood_val_fold in SYNTHETIC_SHIFTS:
                # dont calibrate on ind data or synthetic ood data
                continue
            data_train = data_copy[
                (data_copy["shift"] == ood_val_fold) | (data_copy["shift"] == ind_val_fold)]
            dsd = OODDetector(data_train, ood_val_fold, threshold_method=threshold_method)
            # dsd.kde()
            for ood_test_fold in data_filtered["fold"].unique():

                if ood_test_fold in ["train", "ind_val", "ind_test"]:
                    continue
                if perf_calibrated:
                    data_copy["ood"] = ~data_copy["correct_prediction"]
                ind_test_fold = "ind_test" if ind_val_fold == "ind_val" else "ind_val"
                data_test = data_copy[(data_copy["fold"] == ood_test_fold) | (data_copy["fold"] == ind_test_fold)]
                shift = ood_test_fold.split("_")[0]
                shift_intensity = ood_test_fold.split("_")[-1] if "_" in ood_test_fold else "Organic"
                if ood_perf and not perf_calibrated:
                    data_copy["ood"] = ~data_copy["correct_prediction"]
                tpr, tnr, ba = dsd.get_metrics(data_test)
                if np.isnan(ba):
                    continue
                data_dict.append({"Dataset": dataset, "feature_name": feature, "Threshold Method": threshold_method,
                     "OoD==f(x)=y": ood_perf, "Performance Calibrated": perf_calibrated,
                     "OoD Val Fold": ood_val_fold, "InD Val Fold": ind_val_fold,
                     "OoD Test Fold": ood_test_fold, "InD Test Fold": ind_test_fold, "Shift": shift,
                     "Shift Intensity": shift_intensity, "tpr": tpr, "tnr": tnr, "ba": ba})
    return data_dict


def _compute_one(args):
    (
        data_filtered, threshold_method, dataset, feature,
        ood_perf, perf_calibrated
    ) = args
    try:
        out = parallel_compute_ood_detector_prediction_accuracy(
            data_filtered, threshold_method, dataset, feature,
            ood_perf, perf_calibrated
        )
        return out  # may be None if function early-continues
    except Exception as e:
        # optional: return diagnostics instead of raising to avoid killing the pool
        return {
            "Dataset": dataset,
            "feature_name": feature,
            "Threshold Method": threshold_method,
            "OoD==f(x)=y": ood_perf,
            "Performance Calibrated": perf_calibrated,
            "error": str(e),
        }

def _make_jobs_for_feature(data_filtered, dataset, feature):
    # (ood_perf, perf_calibrated) pairs except the skipped case
    pairs = [(o, p) for o, p in product([True, False], [True, False]) if not (p and not o)]
    # cartesian product over threshold methods
    jobs = [
        (data_filtered, tm, dataset, feature, o, p)
        for (o, p) in pairs
        for tm in THRESHOLD_METHODS
    ]
    return jobs

# ---- main entry ----
def ood_detector_correctness_prediction_accuracy(batch_size, shift="normal"):
    df = load_all(prefix="fine_data", batch_size=batch_size, shift=shift, samples=100)
    df = df[df["fold"] != "train"]

    # Precompute total jobs for progress bar
    total_jobs = 0
    for dataset in DATASETS:
        data_dataset = df[df["Dataset"] == dataset]
        for feature in DSDS:
            data_filtered = data_dataset[data_dataset["feature_name"] == feature]
            if data_filtered.empty:
                continue
            total_jobs += len(_make_jobs_for_feature(data_filtered, dataset, feature))

    results_all = []
    # choose pool size
    n_procs = max(1, cpu_count() - 1)

    with Pool(processes=n_procs) as pool, tqdm(total=total_jobs, desc="Computing") as pbar:
        for dataset in DATASETS:
            data_dataset = df[df["Dataset"] == dataset]
            for feature in DSDS:
                data_filtered = data_dataset[data_dataset["feature_name"] == feature]
                if data_filtered.empty:
                    print(f"No data for {dataset} {feature}")
                    continue

                jobs = _make_jobs_for_feature(data_filtered, dataset, feature)

                # stream results as they complete; update pbar per completed task
                for out in pool.imap_unordered(_compute_one, jobs, chunksize=1):
                    pbar.update(1)
                    if out is None:
                        continue
                    results_all.append(out)

    # Collate + save per-dataset CSVs (to match original behavior)
    # Collate + save per-dataset CSVs (to match original behavior)
    if not results_all:
        print("No results produced.")
        return

    flat_success = [row for out in results_all if isinstance(out, list) for row in out]
    flat_errors = [out for out in results_all if isinstance(out, dict) and "error" in out]

    data = pd.DataFrame(flat_success)
    print(data)
    if not data.empty:
        data["feature_name"].replace(DSD_PRINT_LUT, inplace=True)

    if flat_errors:
        pd.DataFrame(flat_errors).to_csv(f"ood_detector_data/ood_detector_errors_{batch_size}.csv", index=False)

    for dataset in DATASETS:
        data_ds = data[data["Dataset"] == dataset]
        if data_ds.empty:
            continue
        data_ds.to_csv(f"ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv", index=False)

def get_all_ood_detector_data(batch_size, filter_thresholding_method=False, filter_ood_correctness=False, filter_correctness_calibration=False, filter_organic=False, filter_best=False):
    dfs = []
    for dataset, feature in itertools.product(DATASETS, DSDS):
        dfs.append(pd.read_csv(f"ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv"))
    df = pd.concat(dfs)
    if filter_thresholding_method:
        df = df[df["Threshold Method"] == "val_optimal"]
    if filter_ood_correctness:
        df = df[df["OoD==f(x)=y"] == False]
    if filter_correctness_calibration:
        df = df[df["Performance Calibrated"] == False]
    if filter_organic:
        df = df[~(df["OoD Val Fold"].isin(SYNTHETIC_SHIFTS) )&(~df["OoD Test Fold"].isin(SYNTHETIC_SHIFTS))]
    if filter_best:
        meaned_ba = df.groupby(["Dataset", "feature_name"])["ba"].mean().reset_index()
        best_ba = meaned_ba.loc[meaned_ba.groupby("Dataset")["ba"].idxmax()]
        df = df.merge(best_ba[["Dataset", "feature_name"]], on=["Dataset", "feature_name"], how="inner")



    return df

def ood_rv_accuracy_by_thresh_and_stuff(batch_size):
    df = get_all_ood_detector_data(batch_size, filter_organic=True)
    print(df.groupby(["Threshold Method", "OoD==f(x)=y", "Performance Calibrated"])[["ba"]].agg(["min", "mean", "max"]))

def ood_rv_accuracy_by_dataset_and_feature(batch_size):
    df = get_all_ood_detector_data(batch_size, filter_organic=True, filter_thresholding_method=True, filter_correctness_calibration=True)
    print(df.groupby(["OoD==f(x)=y", "Dataset", "feature_name"])["ba"].mean())

def ood_verdict_shiftwise_accuracy_tables(batch_size):
    df = get_all_ood_detector_data(batch_size, filter_thresholding_method=True, filter_ood_correctness=True,
                                   filter_correctness_calibration=True, filter_organic=False, filter_best=True)


    #get only the shifts that affect the performance of the OOD detector
    df_raw = load_all(1, shift="")
    acc_by_dataset_and_shift = df_raw.groupby(["Dataset", "shift"])["correct_prediction"].mean().reset_index()
    organic_shift_accs = acc_by_dataset_and_shift[~acc_by_dataset_and_shift["shift"].isin(SYNTHETIC_SHIFTS+["train", "ind_val", "ind_test"])]

    ind_accs = acc_by_dataset_and_shift[acc_by_dataset_and_shift["shift"].isin(["ind_val", "ind_test"])]

    #filter away shifts that do not have a correct prediction rate below the maximum organic shift accuracy
    max_organic_shift_acc_per_dataset = organic_shift_accs.groupby("Dataset")["correct_prediction"].max().reset_index()
    min_ind_acc_per_dataset = ind_accs.groupby("Dataset")["correct_prediction"].min().reset_index()
    affective_shifts = acc_by_dataset_and_shift.merge(max_organic_shift_acc_per_dataset, on="Dataset", suffixes=("", "_max"))
    affective_shifts = affective_shifts.merge(min_ind_acc_per_dataset, on="Dataset", suffixes=("", "_min"))
    affective_shifts["midpoint"] = (affective_shifts["correct_prediction_max"] + affective_shifts["correct_prediction_min"]) / 2

    affective_shifts["affective"] = affective_shifts["correct_prediction"] <= affective_shifts["midpoint"]
    affective_shifts = affective_shifts[affective_shifts["affective"]==True]

    #filter df to only those shifts for the corresponding datasets
    df.rename(columns={"OoD Test Fold":"shift"}, inplace=True)

    # Merge to keep only matching Dataset+shift rows
    valid_pairs = set(zip(affective_shifts["Dataset"], affective_shifts["shift"]))

    print(valid_pairs)


    print(valid_pairs)
    print(df["shift"].unique())
    df_filtered = df[df.apply(lambda row: (row["Dataset"], row["shift"]) in valid_pairs, axis=1)]

    tprs = df_filtered.groupby(["Dataset", "shift"])[["tpr"]].mean()
    tnrs = df.groupby(["Dataset", "Ind Test Fold"])[["tnr"]].mean().reset_index()

    print(tprs)
    print(tnrs)
    input()
    return tprs, tnrs


def ood_accuracy_vs_pred_accuacy_plot(batch_size):
    df = get_all_ood_detector_data(batch_size, filter_thresholding_method=True, filter_ood_correctness=False,
                                   filter_correctness_calibration=True, filter_organic=False, filter_best=True)
    df = df[df["OoD==f(x)=y"] == True]  # only OOD performance


    # df = df[~((df["OoD==f(x)=y"] == True)&(~df["Performance Calibrated"]))]  # only OOD performance
    #get only the shifts that affect the performance of the OOD detector
    df_raw = load_all(batch_size, shift="")

    acc_by_dataset_and_shift = df_raw.groupby(["Dataset", "fold"])["correct_prediction"].mean().reset_index()

    ood_accs = df.groupby(["Dataset", "OoD Test Fold", "OoD==f(x)=y"])["tpr"].mean().reset_index()
    ind_accs = df.groupby(["Dataset", "InD Test Fold", "OoD==f(x)=y"])["tnr"].mean().reset_index()
    ind_accs["tnr"]=1-ind_accs["tnr"]
    ind_accs.rename(columns={"InD Test Fold":"fold", "tnr":"Detection Rate"}, inplace=True)
    ood_accs.rename(columns={"OoD Test Fold":"fold", "tpr":"Detection Rate"}, inplace=True)

    merged = pd.concat([ood_accs, ind_accs], ignore_index=True)
    merged = merged.merge(acc_by_dataset_and_shift, on=["Dataset", "fold"])
    merged["Shift"] = merged["fold"].apply(lambda x: x.split("_")[0] if "_" in x else "Organic")
    merged["Organic"] = merged["Shift"].apply(lambda x: "Synthetic" if x in SYNTHETIC_SHIFTS else "Organic")

    acc = merged.groupby(["Dataset", "fold"], as_index=False)["correct_prediction"].mean()

    # pull the per-dataset ind_val baseline
    ind = (acc.loc[acc["fold"] == "ind_val", ["Dataset", "correct_prediction"]]
           .rename(columns={"correct_prediction": "ind_val_acc"}))

    # join baseline back to every shift of the same dataset
    acc = acc.merge(ind, on="Dataset", how="left")

    # absolute and relative differences vs ind_val
    acc["acc_diff"] = acc["correct_prediction"] - acc["ind_val_acc"]
    acc["Generalization Gap"] = acc["acc_diff"] / acc["ind_val_acc"]  # e.g., 0.10 == +10%
    acc["Generalization Gap"] = - acc["Generalization Gap"] * 100  # convert to percentage
    merged = merged.merge(acc, on=["Dataset", "fold"], how="left")
    print(merged)

    def plot_ideal_line(data, color=None, **kwargs):
        # Plot a diagonal line from (0, 0) to (1, 1)
        dataset = data["Dataset"].unique()[0]
        plt.axhline(1-DATASETWISE_RANDOM_CORRECTNESS[dataset], color="blue", linestyle="--", label="Maximum Detection Rate")
    g = sns.FacetGrid(merged, col="Dataset", sharex=False, sharey=False, col_wrap=3)
    # g.map_dataframe(sns.regplot, x="Generalization Gap", y="Detection Rate", robust=False, scatter=False)
    g.map_dataframe(sns.scatterplot, x="Generalization Gap", y="Detection Rate", hue="Shift", style="Organic", alpha=0.5, style_order = ["Synthetic","Organic"], edgecolor=None)
    g.map_dataframe(plot_ideal_line)

    g.set_axis_labels("Generalization Gap (\%)", "OoD Detection Rate")
    for ax in g.axes.flat:
        ax.set_ylim(0,1.1)
        # ax.set_xlim(0,1.1)
    plt.savefig("figures/tpr_v_acc.pdf")
    plt.show()


def get_all_ood_detector_verdicts(data):
    data = data[data["fold"] != "train"]
    data = data[data["shift"] != "noise"]
    # data = data[data["shift"]!="noise"]
    # data["ood"] = data["correct_prediction"]
    dfs = []
    with tqdm(total=len(DATASETS) * len(DSDS) * len(THRESHOLD_METHODS)) as pbar:
        for dataset in DATASETS:
            for feature in DSDS:
                data_dataset = data[(data["Dataset"] == dataset) & (data["feature_name"] == feature)]
                for ood_val_fold in data_dataset["shift"].unique():
                    data_copy = data_dataset.copy()
                    if ood_val_fold in ["train", "ind_val", "ind_test"]:
                        continue
                    data_train = data_copy[
                        (data_copy["shift"] == ood_val_fold) | (data_copy["shift"] == "ind_val")]
                    dsd = OODDetector(data_train, ood_val_fold)
                    data_copy["Verdict"] = data_copy.apply(lambda row: dsd.predict(row), axis=1)
                    data_copy["ood_val_fold"] = ood_val_fold
                    data_copy["ood"] = ~data_copy["correct_prediction"]  #
                    data_copy = data_copy[data_copy["ood_val_fold"] != data_copy["shift"]]
                    dfs.append(data_copy)
                pbar.update(1)
    data = pd.concat(dfs)
    return data


def loss_verdict_histogram(batch_size, prefix="final_data"):
    df = load_all(prefix=prefix, compute_ood=False, batch_size=batch_size)

    data = get_all_ood_detector_verdicts(df)

    def baseline(data, color=None, ax=None, **kwargs):
        dataset = data["Dataset"].unique()[0]
        plt.axvline(DATASETWISE_RANDOM_LOSS[dataset], color="red", linestyle="--", label="Random Guessing")


    g = sns.FacetGrid(data, col="Dataset", row="feature_name", sharex=False, sharey=False)
    g.map_dataframe(sns.histplot, x="loss", hue="Verdict", fill=True)
    g.map_dataframe(baseline)

    for ax in g.axes.flat:
        dataset_name_for_ax = ax.get_title().split(" = ")[-1]
        # ax.set_ylim(1, 1e3)
        ax.set_xlim(0, data[data["Dataset"]==dataset_name_for_ax]["loss"].quantile(0.99))
        # ax.set_yscale("log")
    plt.legend()
    plt.show()


def loss_correctness_test():
    data = load_all(1)

    def plot_accs(data, color=None, **kwargs):
        bins = 10
        bin_edges = np.linspace(data["loss"].min(), data["loss"].quantile(0.99), bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        data["bin"] = pd.cut(data["loss"], bins=bin_edges, include_lowest=True)
        count_df = data.groupby(["bin", "correct_prediction"]).size().unstack(fill_value=0)
        print(data["Dataset"].unique())


        proportion_df = count_df.div(count_df.sum(axis=1), axis=0).fillna(0)
        bottom = np.zeros(len(proportion_df))
        for label in proportion_df.columns:
            plt.bar(bin_centers, proportion_df[label], bottom=bottom,
                    width=bin_edges[1] - bin_edges[0], label=label,
                    edgecolor='white', align='center', alpha=0.5)
            bottom += proportion_df[label].values

    g = sns.FacetGrid(data, col="Dataset", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(plot_accs)
    plt.show()


def debiased_ood_detector_correctness_prediction_accuracy(batch_size):
    df = load_all_biased(filter_batch=batch_size)
    df = df[df["fold"]!="train"]
    for dataset in DATASETS:
        data_dict = []
        data_dataset = df[df["Dataset"] == dataset]
        if dataset=="Polyp":
            print(data_dataset.head(10))
        with tqdm(total=df["feature_name"].nunique()*2 * 2, desc=f"Computing for {dataset}") as pbar:
            if os.path.exists(f"ood_detector_data/debiased_ood_detector_correctness_{dataset}_{batch_size}.csv"):
                print("continuing...")
                continue
            for feature in DSDS:
                for k in [-1, 0, 1, 5]:

                    if feature == "knn" and k !=-1:
                        continue
                    if feature == "softmax" and dataset=="Polyp":
                        continue
                    if feature=="rabanser" and k==-1:
                        continue
                    data_filtered = data_dataset[(data_dataset["feature_name"]==feature)&(data_dataset["k"]==k)]
                    if data_filtered.empty:
                        print(f"empty for {dataset}, {feature}, {k})")
                        # input()
                        continue

                        # print("continuing")
                    for ood_perf in [True, False]:
                        for perf_calibrated in [True, False]:
                            if perf_calibrated and not ood_perf:
                                continue  # unimportant
                            for threshold_method in THRESHOLD_METHODS:
                                for ood_val_fold in data_filtered["shift"].unique():
                                    data_copy = data_filtered.copy()
                                    if ood_val_fold in ["train", "ind_val", "ind_test"]:
                                        continue
                                    data_train = data_copy[
                                        ((data_copy["shift"] == ood_val_fold) | (data_copy["shift"] == "ind_val") ) & (data_copy["bias"]=="RandomSampler")]
                                    if data_train.empty:
                                        print(f"No training data for {dataset} {feature}, {k}")
                                        continue
                                    dsd = OODDetector(data_train, ood_val_fold, threshold_method=threshold_method)
                                    # dsd.kde()
                                    for ood_test_fold in data_filtered["shift"].unique():
                                        if ood_test_fold in ["train", "ind_val", "ind_test"]:
                                            continue
                                        for bias in SAMPLERS:
                                            if bias=="ClassOrderSampler" and dataset=="Polyp":
                                                continue

                                            if perf_calibrated:
                                                data_copy["ood"]=~data_copy["correct_prediction"]

                                            data_test = data_copy[((data_copy["shift"]==ood_test_fold)|(data_copy["shift"]=="ind_test"))&(data_copy["bias"]==bias)]

                                            if ood_perf and not perf_calibrated:
                                                data_copy["ood"]=~data_copy["correct_prediction"]
                                            tpr, tnr, ba = dsd.get_metrics(data_test)

                                            if np.isnan(ba):
                                                print("nan val!")
                                                continue
                                            data_dict.append(
                                                {"Dataset": dataset, "feature_name": feature, "Threshold Method": threshold_method,
                                                 "OoD==f(x)=y": ood_perf, "Performance Calibrated": perf_calibrated,
                                                 "OoD Val Fold": ood_val_fold, "OoD Test Fold":ood_test_fold, "bias":SAMPLER_LUT[bias], "k":k, "tpr": tpr, "tnr": tnr, "ba": ba}
                                            )
                                            pbar.set_description(f"Computing for {dataset}, {feature} {ood_perf} {ood_test_fold} {bias}; current ba: {ba}")

                                    # data_copy = data_copy[data_copy["ood_val_fold"]!=data_copy["shift"]]
                            pbar.update(1)

            data = pd.DataFrame(data_dict)
            if not data.empty:
                data.replace(DSD_PRINT_LUT, inplace=True)
                data.to_csv(f"ood_detector_data/debiased_ood_detector_correctness_{dataset}_{batch_size}.csv", index=False)

def eval_debiased_ood_detectors():
    data = load_all_biased(prefix="debiased_data")
    data = data[data["fold"] != "train"]
    data_dict = []

    for batch_size in BATCH_SIZES[1:]:
        for dataset in DATASETS:
            with tqdm(total=len(DSDS) * 2*3) as pbar:
                for feature in DSDS:
                    for assessed_correctness in [True, False]:
                        for k in [-1, 0,1,5]:
                            if feature == "knn" and k != -1:
                                continue
                            if feature=="rabanser" and k==-1:
                                continue

                            data_dataset = data[(data["Dataset"] == dataset) & (data["feature_name"] == feature) & (data["k"]==k) & (data["batch_size"]==batch_size)]
                            if data_dataset.empty:
                                print(f"No data for {dataset}-{feature}-{k}")
                                continue
                            for ood_val_fold in data_dataset["shift"].unique():
                                if ood_val_fold in ["train", "ind_val", "ind_test"]:
                                    continue
                                data_copy = data_dataset.copy()
                                data_train = data_copy[
                                    ((data_copy["shift"] == ood_val_fold) | (data_copy["shift"] == "ind_val"))]
                                data_train = data_train[data_train["bias"] == "RandomSampler"]
                                if data_train.empty:
                                    continue
                                dsd = OODDetector(data_train, ood_val_fold, threshold_method="val_optimal")
                                # dsd.kde()
                                for ood_test_fold in data_dataset["shift"].unique():
                                    if ood_test_fold in ["train", "ind_val", "ind_test"]:
                                        continue
                                    for bias in SAMPLERS:
                                        data_test = data_copy[data_copy["bias"]==bias]
                                        if data_test.empty:
                                            continue
                                        data_test = data_copy[(data_copy["shift"] == ood_test_fold) | (data_copy["shift"]=="ind_test")&(data_copy["bias"]==bias)]

                                        if assessed_correctness:
                                            data_copy["ood"] = ~data_copy["correct_prediction"]  # OOD is the opposite of correct prediction
                                        tpr, tnr, ba = dsd.get_metrics(data_test)
                                        if np.isnan(ba):
                                            continue

                                        data_dict.append(
                                            {"Dataset": dataset, "feature_name": feature,
                                             "OoD Val Fold": ood_val_fold, "OoD Test Fold": ood_test_fold, "tpr": tpr, "tnr": tnr,
                                             "ba": ba, "bias": bias, "k":k, "batch_size":batch_size, "OOD==f(x)==y": assessed_correctness}
                                        )
                            pbar.update(1)

    df = pd.DataFrame(data_dict)
    # print(data.head(10))
    df.replace(DSD_PRINT_LUT, inplace=True)
    df.replace(SAMPLER_LUT, inplace=True)
    df.to_csv(f"ood_detector_data/debiased_ood_detector_correctness.csv", index=False)

def debiased_plots():
    df = []
    for dataset, batch_size in itertools.product(DATASETS, BATCH_SIZES[1:]):
        try:
            df_i = pd.read_csv(f"ood_detector_data/debiased_ood_detector_correctness_{dataset}_{batch_size}.csv")
            df_i["batch_size"] = batch_size
            df_i["Dataset"] = dataset
            df.append(df_i)
        except:
            continue
    df = pd.concat(df)

    df = df[(~df["OoD Test Fold"].isin(SYNTHETIC_SHIFTS))&(~df["OoD Val Fold"].isin(SYNTHETIC_SHIFTS))]
    df = df[~((df["Dataset"]=="CCT") & (df["bias"]=="SequentialSampler"))]  # CCT has no class order sampler

    #vanilla comparisons
    vanilla = df[df["k"].isin([0, -1])]
    vanilla.rename(columns={"k":"Aggregation"}, inplace=True)
    vanilla["Aggregation"].replace({0: "KS Test", -1: "Mean"}, inplace=True)
    vanilla.rename(columns={'OoD==f(x)=y':"OoD Label"}, inplace=True)
    vanilla["OoD Label"] = vanilla["OoD Label"].apply(lambda x: "Correctness" if x else "Partition")

    bias_effect = df[(df["k"].isin([0,-1])) & (df["OoD==f(x)=y"]==False)]

    print(bias_effect.groupby(["Dataset", "feature_name", "bias"])[["ba"]].agg(["mean"]).reset_index())
    input()
    bias_effect["ba"] = bias_effect["ba"] - bias_effect[bias_effect["bias"]=="Unbiased"]["ba"].mean()

    bias_effect = bias_effect[bias_effect["bias"]!="Unbiased"]
    g = sns.FacetGrid(bias_effect, col="feature_name", margin_titles=True, sharex=False, sharey=True, col_wrap=3)
    g.map_dataframe(sns.boxenplot, x="bias", y="ba", palette=sns.color_palette())
    plt.savefig("figures/ood_detector_bias_boxplots.pdf")
    for ax in g.axes.flat:
        ax.axhline(y=0, color="red", linestyle="--")
    plt.show()

    unbiased = vanilla[vanilla["bias"]=="Unbiased"]
    meaned = unbiased.groupby(["Dataset", "feature_name", "Aggregation"])["ba"].mean().reset_index()
    best_features_idx = meaned.groupby(["Dataset", "Aggregation"])["ba"].idxmax().reset_index()
    print(best_features_idx)
    best_features = meaned.loc[best_features_idx["ba"]]
    print(best_features)
    filtered_unbiased = unbiased.merge(best_features[["Dataset", "feature_name"]], on=["Dataset", "feature_name"])
    print(filtered_unbiased.head(10))
    g = sns.FacetGrid(filtered_unbiased, col="Dataset", row="OoD Label", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.lineplot, x="batch_size", y="ba", hue="Aggregation", palette = sns.color_palette())
    for ax in g.axes.flat:
        ax.set_ylim(0.4, 1)
    g.add_legend(bbox_to_anchor=(0.9, 0.3), loc='center left', title="Aggregation", ncol=1)
    plt.savefig("batched_ood_verdict_accuracy.pdf")
    plt.show()
    input()
    # g = sns.FacetGrid(vanilla, col="Dataset", row="OoD Label")
    # g.map_dataframe(sns.boxplot, x="Bias", y="ba", hue="Aggregation", palette=sns.color_palette())
    # plt.show()



    g = sns.FacetGrid(df, col="feature_name")
    g.map_dataframe(sns.boxenplot, x="k", y="ba", palette=sns.color_palette())
    plt.show()
    average_for_each_dsd = df.groupby(["Dataset", "k", "feature_name"])["ba"].mean().reset_index()
    best_idx = average_for_each_dsd.groupby(["Dataset", "k"])["ba"].idxmax()
    best = average_for_each_dsd.loc[best_idx]
    print(best.groupby(["k", "Dataset", "feature_name"]).mean())
    sns.boxplot(best, x="k", y="ba", palette=sns.color_palette())
    plt.show()

    # g = sns.FacetGrid(df, col="Dataset", margin_titles=True, sharex=False, sharey=False, col_wrap=3)
    # g.map_dataframe(sns.boxplot, x="bias", y="balanced_accuracy", hue="Batch Size", palette=sns.color_palette(), order=["Unbiased", "Synthetic", "Temporal", "Class"])
    # g.add_legend(bbox_to_anchor=(0.7, 0.25), loc='center left', title="Batch Size", ncol=1)
    # plt.tight_layout()
    # plt.savefig("ood_detector_bias_boxplots.pdf")
    # # for ax in g.axes.flat:
    # #     ax.legend()
    # plt.show()

def ood_verdict_plots_batched():
    dfs = []
    for dataset, batch_size in itertools.product(DATASETS, BATCH_SIZES):
        df = pd.read_csv(f"ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv")
        df["batch_size"] = batch_size
        dfs.append(df)
    data = pd.concat(dfs)
    data = data[(~data["OoD Test Fold"].isin(SYNTHETIC_SHIFTS))&(~data["OoD Val Fold"].isin(SYNTHETIC_SHIFTS))]

    data = data[(data["Threshold Method"]=="val_optimal")&(data["Performance Calibrated"]==False)]
    g = sns.FacetGrid(data, col="Dataset", row="OoD==f(x)=y", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.lineplot, x="batch_size", y="ba", hue="feature_name", markers=True, dashes=False)
    for ax in g.axes.flat:
        ax.set_ylim(0.4, 1)

    g.add_legend(bbox_to_anchor=(0.7, 0.3), loc='center left', title="Feature", ncol=1)
    plt.savefig("batched_ood_verdict_accuracy.pdf")
    plt.show()

def plot_batching_effect(dataset, feature):
    df = load_pra_df(dataset, feature, batch_size=1)
    df_batched = load_pra_df(dataset, feature, batch_size=30)
    oods = df[(df["ood"])&(~df["correct_prediction"])]
    inds = df[(~df["ood"])&(df["correct_prediction"])]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis
    plot_df = pd.concat([oods, inds])
    sns.kdeplot(plot_df, x="feature", hue="ood", fill=True, common_norm=False,ax=ax1)
    sns.kdeplot(df_batched, x="feature", hue="ood", fill=False, common_norm=False, ax=ax2, linestyle="--")
    plt.tight_layout()
    # plt.yscale("log")
    plt.xlim(0,2500)
    plt.savefig(f"{dataset}_{feature}_kdes.pdf")
    plt.show()


def get_error_rate_given_rv():
    results_list = []
    for dataset in DATASETS:
        for dsd in DSDS:
            if dsd=="rabanser":
                continue
            df = load_pra_df(dataset, dsd, batch_size=1)
            ood_folds = df[df["ood"]]["fold"].unique()
            for ood_val, ood_test in itertools.product(ood_folds, ood_folds):
                data_train = df[(df["fold"]=="ind_val")|(df["fold"]==ood_val)]
                data_test = df[(df["fold"]=="ind_test")|(df["fold"]==ood_test)]
                ood_detector = OODDetector(data_train, ood_val)
                data_test["detected_ood"]=ood_detector.predict(data_test)
                missed_predictions = data_test[~data_test["correct_prediction"]]
                missed_ood = missed_predictions[~missed_predictions["detected_ood"]]
                for fold in data_test["fold"].unique():
                    rv_prop = len(missed_ood[missed_ood["fold"]==fold])/len(data_test[data_test["fold"]==fold])
                    vanilla_prop = 1-data_test[data_test["fold"]==fold]["correct_prediction"].mean()
                    results_list.append({"Dataset":dataset, "Feature":dsd, "Fold":fold, "RV Proportion":rv_prop, "Vanilla Prop":vanilla_prop})

    results = pd.DataFrame(results_list)
    print(results.groupby(["Dataset", "Feature", "Fold"])[["RV Proportion", "Vanilla Prop"]].mean())
    results.to_csv("datasetwise_incorrect_detections.csv")
