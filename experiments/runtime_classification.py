
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import seaborn as sns
import matplotlib.pyplot as plt
from components import OODDetector
from itertools import product
from rateestimators import ErrorAdjustmentEstimator
from simulations import UniformBatchSimulator
from utils import *


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
            cba_data.append({"Component":"OOD Detector", "Accuracy":acc, "Risk Estimate":d_risk})
            pbar.update(1)

    df = pd.DataFrame(cba_data).groupby(["Component", "Accuracy"]).mean().reset_index()

    sns.lineplot(df, x="Accuracy", y="Risk Estimate", hue="Component")
    plt.savefig("cba.pdf")
    plt.show()


def parallel_compute_ood_detector_prediction_accuracy(data_filtered, threshold_method, dataset, feature, ood_perf, perf_calibrated):
    data_dict = []
    ood_val_folds = data_filtered[(data_filtered["Organic"] == True) & (data_filtered["ood"] == True)]["fold"].unique()

    for ind_val_fold, ind_test_fold in itertools.product(["ind_val", "ind_test"],repeat=2) :
        for ood_val_fold in ood_val_folds:
            data_copy = data_filtered.copy()
            if ood_val_fold in ["train", "ind_val", "ind_test"] or ood_val_fold.split("_")[0] in SYNTHETIC_SHIFTS:
                # dont calibrate on ind data or synthetic ood data
                continue
            data_train = data_copy[
                (data_copy["fold"] == ood_val_fold) | (data_copy["fold"] == ind_val_fold)].copy()
            if perf_calibrated:
                data_train["ood"] = ~data_train["correct_prediction"]

            dsd = OODDetector(data_train, threshold_method=threshold_method)
            # dsd.kde()
            for ood_test_fold in data_copy[data_copy["ood"]==True]["fold"].unique():
                if ood_test_fold in ["train", "ind_val", "ind_test"]:
                    continue # not ood

                data_test = data_copy[(data_copy["fold"] == ood_test_fold) | (data_copy["fold"] == ind_test_fold)].copy()
                shift = ood_test_fold.split("_")[0]
                shift_intensity = ood_test_fold.split("_")[-1] if "_" in ood_test_fold else "Organic"

                if ood_perf:
                    data_test["ood"] = ~data_test["correct_prediction"]

                # dsd.val_kde(data_test, fname=f"figures/kdes/nico_knn_indval_kde_{ood_val_fold}_{ind_val_fold}_{ood_test_fold}_{ind_test_fold}.png")
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
def ood_detector_correctness_prediction_accuracy(batch_size, prefix="fine_data", shift="normal"):
    df = load_all(prefix=prefix, batch_size=batch_size, shift=shift, samples=100)
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
    # n_procs=1 #debug
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
    if not data.empty:
        data["feature_name"].replace(DSD_PRINT_LUT, inplace=True)

    if flat_errors:
        pd.DataFrame(flat_errors).to_csv(f"ood_detector_data/ood_detector_errors_{batch_size}.csv", index=False)

    for dataset in DATASETS:
        data_ds = data[data["Dataset"] == dataset]
        if data_ds.empty:
            continue
        data_ds.to_csv(f"ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv", index=False)

def get_all_ood_detector_data(batch_size, filter_thresholding_method=False, filter_ood_correctness=False, filter_correctness_calibration=False, filter_organic=False, filter_best=False, prefix="ood_detector_data"):
    dfs = []
    for dataset, feature in itertools.product(DATASETS, DSDS):
        dfs.append(pd.read_csv(f"{prefix}/ood_detector_correctness_{dataset}_{batch_size}.csv"))
    df = pd.concat(dfs)
    if filter_thresholding_method:
        df = df[df["Threshold Method"] == "val_optimal"]

    if filter_ood_correctness:
        df = df[df["OoD==f(x)=y"] == False]
    if filter_correctness_calibration:
        df = df[df["Performance Calibrated"] == False]
    if filter_organic:
        df = df[df["Shift Intensity"] == "Organic"]
    if filter_best:
        meaned_ba = df.groupby(["Dataset", "feature_name"])["ba"].mean().reset_index()
        best_ba = meaned_ba.loc[meaned_ba.groupby("Dataset")["ba"].idxmax()]
        df = df.merge(best_ba[["Dataset", "feature_name"]], on=["Dataset", "feature_name"], how="inner")

    return df
def ood_rv_accuracy_by_dataset_and_feature(batch_size):
    df = get_all_ood_detector_data(batch_size, filter_organic=True, filter_thresholding_method=True, filter_correctness_calibration=True, filter_ood_correctness=False, prefix="ood_detector_data")
    df = df[df["OoD Val Fold"]!=df["OoD Test Fold"]]
    df = df[df["InD Val Fold"]!=df["InD Test Fold"]]
    print(df.head(100))
    print(df.groupby(["OoD==f(x)=y", "Dataset", "feature_name"])[["tpr", "tnr", "ba"]].mean())


def ood_rv_accuracy_by_thresh_and_stuff(batch_size):
    df = get_all_ood_detector_data(batch_size, filter_organic=True)
    print(df.groupby(["Threshold Method", "OoD==f(x)=y", "Performance Calibrated"])[["ba"]].agg(["min", "mean", "max"]))



def ood_verdict_shiftwise_accuracy_tables(batch_size, filter_organic=False):
    df = get_all_ood_detector_data(batch_size, filter_thresholding_method=True, filter_ood_correctness=True,
                                   filter_correctness_calibration=True, filter_organic=filter_organic, filter_best=True)


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
    df_filtered = df[df.apply(lambda row: (row["Dataset"], row["shift"]) in valid_pairs, axis=1)]
    print(df.columns)
    tprs = df_filtered.groupby(["Dataset", "shift"])[["tpr"]].mean().reset_index()
    tnrs = df.groupby(["Dataset", "InD Test Fold"])[["tnr"]].mean().reset_index()
    tnrs.rename(columns={"InD Test Fold":"shift"}, inplace=True)

    return tprs, tnrs


def ood_accuracy_vs_pred_accuacy_plot(batch_size):
    df = get_all_ood_detector_data(batch_size, filter_thresholding_method=True, filter_ood_correctness=False,
                                   filter_correctness_calibration=True, filter_organic=False, filter_best=True)
    df = df[df["OoD==f(x)=y"] == True]  # only OOD performance
    print(df.columns)
    print()
    df_synth = df[df["Shift Intensity"]!="Organic"]
    df_synth.replace(SHIFT_PRINT_LUT, inplace=True)
    unique_shifts  = df_synth["Shift"].unique().tolist()

    g = sns.FacetGrid(df_synth, col="Dataset", sharex=False, sharey=False, col_wrap=3)
    g.map_dataframe(sns.lineplot, x="Shift Intensity", y="ba", hue="Shift", hue_order=sorted(unique_shifts), marker="o", alpha=0.7)
    g.set_axis_labels("Shift Intensity", "Balanced Accuracy")
    g.add_legend(bbox_to_anchor=(0.7, 0.25), loc='center left', frameon=True, title="Shift Type")
    for ax in g.axes.flat:
        ax.set_xticks([])
    plt.tight_layout()
    plt.savefig("figures/shift_intensity_vs_ba.pdf")
    plt.show()

    # df = df[~((df["OoD==f(x)=y"] == True)&(~df["Performance Calibrated"]))]  # only OOD performance
    #get only the shifts that affect the performance of the OOD detector
    df_raw = load_all(batch_size, shift="", prefix="fine_data")

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

    acc["Generalization Gap"] = acc["correct_prediction"] - acc["ind_val_acc"]
    acc["Accuracy"] = acc["correct_prediction"]
    # acc["Generalization Gap"] = acc["acc_diff"] / acc["ind_val_acc"]  # e.g., 0.10 == +10%
    # acc["Generalization Gap"] = - acc["Generalization Gap"] * 100  # convert to percentage
    merged = merged.merge(acc, on=["Dataset", "fold"], how="left")
    print(merged)
    merged["shift"] = merged.replace(SHIFT_PRINT_LUT, inplace=True)

    hue_order = sorted(merged["Shift"].unique().tolist())
    print(hue_order)
    def plot_ideal_line(data, color=None, **kwargs):
        # Plot a diagonal line from (0, 0) to (1, 1)
        dataset = data["Dataset"].unique()[0]
        plt.axhline(1-DATASETWISE_RANDOM_CORRECTNESS[dataset], color="blue", linestyle="--", label="Maximum Detection Rate")
    sns.scatterplot(data=merged, x="Generalization Gap", y="Detection Rate", hue="Dataset", alpha=0.5, edgecolor=None)
    plt.show()
    g = sns.FacetGrid(
        merged, col="Dataset", sharex=False, sharey=False, col_wrap=2, aspect=1, height=2.5
    )
    g.map_dataframe(sns.scatterplot, x="Generalization Gap", y="Detection Rate",
                    hue="Shift", alpha=0.5, edgecolor=None, hue_order=hue_order)
    g.map_dataframe(plot_ideal_line)

    # Place legend inside (or wherever you want) without reserving right margin
    g.add_legend(bbox_to_anchor=(0.6, 0.17), loc='center left', frameon=True, title="Shift Type")

    # If any residual padding remains:
    g.figure.subplots_adjust(right=0.98)  # or 1.0
    g.set_axis_labels("Generalization Gap", "OoD Detection Rate")
    for ax in g.axes.flat:
        ax.set_ylim(0,1.1)
        # ax.set_xlim(0,1.1)
    plt.savefig("figures/tpr_v_acc.pdf")
    # plt.tight_layout()
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




def debiased_ood_detector_correctness_prediction_accuracy(batch_size):
    df = load_all_biased(filter_batch=batch_size)
    df = df[df["fold"]!="train"]
    for dataset in DATASETS:
        data_dict = []
        data_dataset = df[df["Dataset"] == dataset]
        if dataset=="Polyp":
            print(data_dataset.head(10))
        with tqdm(total=df["feature_name"].nunique()*2 * 2, desc=f"Computing for {dataset}") as pbar:
            # if os.path.exists(f"ood_detector_data/debiased_ood_detector_correctness_{dataset}_{batch_size}.csv"):
            #     print("continuing...")
            #     continue
            for feature in DSDS:
                for k in [-1, 0, 1, 5, 10]:

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
                                    dsd = OODDetector(data_train, threshold_method=threshold_method)
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
            print("No data for ", dataset, batch_size)
            continue
    df = pd.concat(df)

    df = df[(~df["OoD Test Fold"].isin(SYNTHETIC_SHIFTS))&(~df["OoD Val Fold"].isin(SYNTHETIC_SHIFTS))]
    # df = df[~((df["Dataset"]=="CCT") & (df["bias"]=="SequentialSampler"))]  # CCT has no class order sampler

    #vanilla comparisons
    vanilla = df[df["k"].isin([0, -1])]
    vanilla.rename(columns={"k":"Aggregation"}, inplace=True)
    vanilla["Aggregation"].replace({0: "KS Test", -1: "Mean"}, inplace=True)
    vanilla.rename(columns={'OoD==f(x)=y':"OoD Label"}, inplace=True)
    vanilla["OoD Label"] = vanilla["OoD Label"].apply(lambda x: "Correctness" if x else "Partition")

    bias_effect = df[(df["k"].isin([0,-1])) & (df["OoD==f(x)=y"]==False)]

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
    best_features = meaned.loc[best_features_idx["ba"]]
    filtered_unbiased = unbiased.merge(best_features[["Dataset", "feature_name"]], on=["Dataset", "feature_name"])
    g = sns.FacetGrid(filtered_unbiased, col="Dataset", row="OoD Label", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.lineplot, x="batch_size", y="ba", hue="Aggregation", palette = sns.color_palette())
    for ax in g.axes.flat:
        ax.set_ylim(0.4, 1)
    g.add_legend(bbox_to_anchor=(0.9, 0.3), loc='center left', title="Aggregation", ncol=1)
    plt.savefig("figures/batched_ood_verdict_accuracy.pdf")
    plt.show()
    # g = sns.FacetGrid(vanilla, col="Dataset", row="OoD Label")
    # g.map_dataframe(sns.boxplot, x="Bias", y="ba", hue="Aggregation", palette=sns.color_palette())
    # plt.show()


    df.replace({"rabanser":"Rabanser"}, inplace=True)
    g = sns.FacetGrid(df[df["feature_name"]!="kNN"], col="feature_name", height=2.5, col_wrap=3)
    g.map_dataframe(sns.boxenplot, x="k", y="ba", palette=sns.color_palette())
    g.set_titles(template="{col_name}")

    plt.savefig("figures/debiased_boxenplots.pdf")
    plt.show()

    average_for_each_dsd = df.groupby(["Dataset", "k", "feature_name"])["ba"].mean().reset_index()
    best_idx = average_for_each_dsd.groupby(["Dataset", "k"])["ba"].idxmax()
    best = average_for_each_dsd.loc[best_idx]
    print(best.groupby(["k", "Dataset", "feature_name"]).mean())
    sns.boxplot(best, x="k", y="ba", palette=sns.color_palette())
    plt.show()

    g = sns.FacetGrid(df[df["feature_name"]!="kNN"], col="Dataset", margin_titles=True, sharex=False, sharey=False, height=2.5, col_wrap=3)
    g.map_dataframe(sns.boxenplot, x="k", y="ba", hue="k", palette=sns.color_palette())
    g.set_titles(template="{col_name}")
    plt.tight_layout()
    # Get figure width
    fig_width = g.fig.get_size_inches()[0]
    num_plots = len(g.axes.flat)
    num_cols = 3  # Top row columns
    last_row_plots = num_plots % num_cols
    # Compute total space occupied by the last row's plots
    last_row_width = (fig_width / num_cols) * last_row_plots

    # Compute left padding to center the row
    left_padding = (fig_width - last_row_width) / 2

    # Adjust position of the last row's plots
    for ax in g.axes[-last_row_plots:]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + left_padding / fig_width, pos.y0, pos.width, pos.height])
    plt.savefig("figures/debiased_dataset_boxenplots.pdf")

    plt.show()

def ood_verdict_plots_batched():
    dfs = []
    for dataset, batch_size in itertools.product(DATASETS, BATCH_SIZES):
        df = pd.read_csv(f"ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv")
        df["batch_size"] = batch_size
        dfs.append(df)
    data = pd.concat(dfs)
    print(data["Shift Intensity"].unique())
    data = data[(data["Shift Intensity"]=="Organic")]
    print(data["OoD Test Fold"].unique())
    print(data["OoD Val Fold"].unique())
    # data = data[data["OoD==f(x)=y"]==True]
    data = data[(data["Threshold Method"]=="val_optimal")&(data["Performance Calibrated"]==False)]
    data["Labeling"]= data["OoD==f(x)=y"].apply(lambda x: "Correctness" if x else "Partition")

    g = sns.FacetGrid(data, col="feature_name", margin_titles=True, sharex=True, sharey=True, col_wrap=2)
    print(data.groupby(["batch_size", "Dataset", "feature_name", "Labeling"])["ba"].mean().reset_index().groupby(["Dataset", "Labeling"])["ba"].max())
    g.map_dataframe(sns.lineplot, x="batch_size", y="ba", hue="Dataset", style="Labeling", style_order=["Correctness", "Partition"], markers=True, dashes=False)
    g.set_axis_labels("Batch Size", "Balanced Accuracy")

    # for ax in g.axes.flat:
    #     ax.set_ylim(0.4, 1)

    g.add_legend(bbox_to_anchor=(0.80, 0.5), loc='center left', title="Feature", ncol=1)
    # plt.tight_layout(h_pad=1)

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
            df = load_pra_df(dataset, dsd, batch_size=1)
            if df.empty:
                print("No data for ", dataset, dsd)
                continue

            ood_folds = df[df["ood"]]["fold"].unique()
            for ood_val, ood_test in itertools.product(ood_folds, ood_folds):
                data_train = df[(df["fold"]=="ind_val")|(df["fold"]==ood_val)]
                # copy() to avoid SettingWithCopyWarning
                data_test = df[(df["fold"]=="ind_test")|(df["fold"]==ood_test)].copy()

                ood_detector = OODDetector(data_train, threshold_method="val_optimal")
                data_test["detected_ood"] = ood_detector.predict(data_test)

                counts = (
                    data_test
                    .groupby(["fold", "correct_prediction", "detected_ood"])
                    .size()
                    .reset_index(name="count")
                )

                # ensure all 4 cells exist for each fold
                full_cols = pd.MultiIndex.from_product(
                    [[False, True], [False, True]],
                    names=["correct_prediction", "detected_ood"]
                )

                for fold in counts["fold"].unique():
                    g = (
                        counts.loc[counts["fold"] == fold, ["correct_prediction", "detected_ood", "count"]]
                        .set_index(["correct_prediction", "detected_ood"])["count"]
                        .reindex(full_cols, fill_value=0)
                    )

                    # vanilla error rate (incorrect among all samples)
                    vanilla_error_rate = g.loc[False].sum() / g.sum()

                    # error rate after abstention = incorrect & not-abstained / all not-abstained
                    error_rate_after_rv = g.loc[(False, False)] / g.sum()

                    # proportion of lost correct predictions = correct & abstained / all correct
                    rate_dropped_predictions = g.loc[(True, True)] / g.sum()

                    # use/store these three numbers per fold...

                    results_list.append({"Dataset":dataset, "Feature":dsd, "Fold":fold, "Error Rate w/RV":error_rate_after_rv, "Vanilla Error Rate":vanilla_error_rate, "Incorrect Abstention Rate":rate_dropped_predictions})

    results = pd.DataFrame(results_list)
    print(results.groupby(["Dataset", "Feature", "Fold"])[["Error Rate w/RV", "Incorrect Abstention Rate", "Vanilla Error Rate"]].mean())
    results.to_csv("datasetwise_incorrect_detections.csv")

