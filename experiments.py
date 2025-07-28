from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import itertools
# from albumentations.random_utils import normal

from seaborn import FacetGrid
import warnings

from plots import load_dfs

warnings.filterwarnings("ignore")
from simulations import *
import matplotlib.pyplot as plt
from utils import *
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from components import OODDetector


def simulate_dsd_accuracy_estimation(data, rate, val_set, test_set, ba, tpr, tnr, dsd):
    sim = UniformBatchSimulator(data, ood_test_shift=test_set, ood_val_shift=val_set, estimator=BernoulliEstimator,
                                use_synth=False)
    results = sim.sim(rate, 600)
    results = results.groupby(["Tree"]).mean().reset_index()
    # results = results.mean()
    results["dsd"] = dsd
    results["ba"] = ba
    results["tpr"] = tpr
    results["tnr"] = tnr
    results["rate"] = rate
    results["test_set"] = test_set
    results["val_set"] = val_set
    return results

def xval_errors(values):
    return np.mean([np.sum(np.abs(np.subtract.outer(valwise_accuracies, valwise_accuracies))) / np.sum(
        np.ones_like(valwise_accuracies) - np.eye(valwise_accuracies.shape[0])) for valwise_accuracies in values])

# pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(precision=3, suppress=True)


def get_dsd_verdicts_given_true_trace(trace, tpr, tnr):
    def transform(v):
        if v==1:
            if np.random.rand() < tpr:
                return 1
            else:
                return 0
        else:
            if np.random.rand() < tnr:
                return 0
            else:
                return 1
    return [transform(i) for i in trace]


def plot_dsd_accuracies(samples=1000):
    data = []
    for batch_size in BATCH_SIZES:
        batch_size_df = fetch_dsd_accuracies(batch_size, samples=100)
        # batch_size_df = batch_size_df.groupby(["Dataset", "DSD", "val_fold", "test_fold"])[["ba"]].mean().reset_index()
        batch_size_df["batch_size"]=batch_size
        for dataset in DATASETS:
            for dsd in DSDS:
                filt = batch_size_df[(batch_size_df["Dataset"]==dataset)&(batch_size_df["DSD"]==dsd)]
                print(filt)
                pivoted = filt.pivot(index=["DSD", "batch_size", "Dataset", "val_fold"],
                                     columns="test_fold", values="ba")
                error = xval_errors(pivoted.values)
                data.append({
                    "DSD":dsd, "Dataset":dataset, "batch_size":batch_size,
                    "error":error, "ba":filt["ba"].mean(),
                })

                # print(pivoted)
    df = pd.DataFrame(data)
    g = FacetGrid(df, col="Dataset")

    def plot_with_error(data, **kwargs):
        """Helper function to plot line and error bands with proper colors"""
        palette = sns.color_palette(n_colors=data["DSD"].nunique())  # Get a color palette
        dsd_unique = data["DSD"].unique()
        color_dict = {dsd: palette[i] for i, dsd in enumerate(dsd_unique)}  # Assign colors per DSD

        for dsd, group in data.groupby("DSD"):
            color = color_dict[dsd]
            plt.plot(group["batch_size"], group["ba"], label=dsd, color=color)
            plt.fill_between(group["batch_size"], group["ba"] - group["error"], group["ba"] + group["error"],
                             alpha=0.2, color=color)

    g.map_dataframe(plot_with_error)
    g.add_legend()
    plt.savefig("dsd_accuracy.pdf")
    plt.show()


def accuracy_by_fold():
    errors = []

    for batch_size in BATCH_SIZES:
        print(batch_size)
        all_data = pd.concat([load_pra_df(dataset_name=dataset_name, feature_name="knn", batch_size=batch_size, samples=100) for dataset_name in
                              DATASETS])
        table = all_data.groupby(["Dataset", "shift"])["correct_prediction"].mean()

        fold_wise_error = table.reset_index()

        fold_wise_error_ood = fold_wise_error[~fold_wise_error["shift"].isin(["ind_val", "ind_test", "train"])]
        fold_wise_error_ind = fold_wise_error[fold_wise_error["shift"].isin(["ind_val", "ind_test", "train"])]

        # Calculate the difference matrix

        for dataset_name in DATASETS:
            filt = fold_wise_error_ind[fold_wise_error_ind["Dataset"]==dataset_name]
            error = np.abs(filt[filt["shift"]=="ind_val"]["correct_prediction"].mean()-filt[filt["shift"]=="ind_test"]["correct_prediction"].mean())
            errors.append({"ood":False, "Dataset": dataset_name, "batch_size":batch_size, "Error":error})

        diff_matrices = {}
        for dataset, group in fold_wise_error_ood.groupby('Dataset'):
            group.set_index('shift', inplace=True)
            accuracy_values = group['correct_prediction'].to_numpy()
            diff_matrix = pd.DataFrame(
                data=np.subtract.outer(accuracy_values, accuracy_values),
                index=group.index,
                columns=group.index
            )
            # Store the matrix for each dataset

            diff_matrices[dataset] = diff_matrix
            try:
                [errors.append({"ood": True, "Dataset":dataset, "Error":np.abs(error), "batch_size":batch_size}) for error in np.unique(diff_matrix[diff_matrix!=0])]
            except:
                print("no non-zero values in OoD matrix; setting zero error...")
                errors.append({"ood": True, "Dataset":dataset, "Error":0, "batch_size":batch_size})

            # Display the difference matrices for each dataset
    errors = pd.DataFrame(errors)
    g = sns.FacetGrid(errors, col="Dataset")
    g.map_dataframe(sns.lineplot, x="batch_size", y="Error", hue="ood",  errorbar=("pi", 100) )
    for ax in g.axes.flat:
        ax.set_yscale("log")
    plt.savefig("tree1_errors.pdf")
    plt.show()
    return errors



def accuracy_by_fold_and_dsd_verdict():
    data = []
    for batch_size in BATCH_SIZES:
        print("loading")
        df = load_all(batch_size, samples=100)
        #filter unneded data
        df = df[df["shift"]!="train"]
        df = df[df["shift"]!="noise"]

        for i, dataset in enumerate(DATASETS):
            for feature in DSDS:
                filtered  = df[(df["Dataset"]==dataset)&(df["feature_name"]==feature)]
                shifts = filtered["shift"].unique()

                for ood_val_shift in shifts:
                    if ood_val_shift in ["train", "ind_val", "ind_test"]:
                        continue
                    filtered_copy = filtered.copy()
                    dsd = OODDetector(filtered, ood_val_shift) #train a dsd for ood_val_shift
                    filtered_copy["D(ood)"] = filtered_copy.apply(lambda row: dsd.predict(row), axis=1)
                    filtered_copy["ood_val_shift"]=ood_val_shift
                    filtered_copy["feature_name"] = feature
                    accuracy = filtered_copy.groupby(["Dataset", "ood_val_shift", "shift", "feature_name", "D(ood)", "ood"])["correct_prediction"].mean().reset_index()
                    accuracy["batch_size"]=batch_size

                    data.append(accuracy)

    data = pd.concat(data)

    # print(data)
    errors = []

    for dataset in DATASETS:
        for ood in data["ood"].unique():
            for dood in data["D(ood)"].unique():
                for batch_size in BATCH_SIZES:
                    for feature_name in DSDS:
                        filt = data[(data["Dataset"]==dataset)&(data["ood"]==ood)&(data["batch_size"]==batch_size)&(data["feature_name"]==feature_name)&(data["D(ood)"]==dood)]
                        if filt.empty:
                            print(f"No data for combination {dataset} OOD={ood}, batch_size={batch_size}, feature={feature_name}")
                            continue
                        pivoted = filt.pivot(index=["feature_name", "batch_size", "Dataset", "D(ood)","ood_val_shift"], columns="shift", values="correct_prediction")
                        pivoted.fillna(0,inplace=True)
                        values = pivoted.values
                        cross_validated_error = xval_errors(values)
                        if np.isnan(cross_validated_error):
                            cross_validated_error=0
                        errors.append({
                            "Dataset": dataset, "feature_name":feature_name, "ood":ood, "D(ood)":dood,
                            "batch_size": batch_size, "Error": cross_validated_error
                        })

    results = pd.DataFrame(errors)
    results.to_csv("conditional_accuracy_errors")
    # results = results.groupby(["Dataset", "feature_name", "ood", "D(ood)"]).mean().reset_index()
    g = sns.FacetGrid(results, col="Dataset", row="ood")
    g.map_dataframe(sns.boxplot, x="batch_size", y="Error", hue="D(ood)")
    g.add_legend()
    for ax in g.axes.flat:
        ax.set_ylim(0,1)
    plt.savefig("conditional_accuracy_errors.pdf")
    plt.show()



    # g = sns.FacetGrid(data.reset_index(), col="Dataset", row="D(ood)")
    # g.map_dataframe(sns.lineplot, x="batch_size", y="correct_prediction", hue="feature_name")


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
        sim = UniformBatchSimulator(data, ood_test_shift=ood_test_set, ood_val_shift=ood_val_set, maximum_loss=0.5, estimator=BernoulliEstimator, dsd_tnr=0.9, dsd_tpr=0.9)
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
                                    estimator=BernoulliEstimator, dsd_tnr=0.9, dsd_tpr=0.9)
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


def get_datasetwise_risk():
    results_list = []
    for dsd in DSDS:
        for model_name in ["deeplabv3plus", "unet", "segformer"]:
            df = load_pra_df("Polyp", dsd, model=model_name, batch_size=1, samples=1000)
            for dataset in ["noise", "ind_test", "EndoCV2020", "EtisLaribDB", "CVC-ClinicDB" ]:
                if dataset not in df["shift"].unique():
                    print(f"Dataset {dataset} not in df")
                    continue
                print()
                for ood_val_shift in ["EndoCV2020", "EtisLaribDB", "CVC-ClinicDB"]:
                    sim = UniformBatchSimulator(df, ood_test_shift=dataset, ood_val_shift=ood_val_shift, maximum_loss=0.5, use_synth=False)
                    results = sim.sim(1, 600)
                    results["Dataset"]=dataset
                    results["Model"]=model_name
                    results["ood_val_shift"]=ood_val_shift
                    results = results.groupby(["Model", "Tree",  "Dataset"])[["True Risk", "Accuracy"]].mean().reset_index()
                    results["DSD"]=dsd
                    results_list.append(results)
    results = pd.concat(results_list)
    print(results.groupby(["Model", "DSD", "Dataset", "Tree"])[["True Risk", "Accuracy"]].mean())
    results.to_csv("datasetwise_risk.csv")

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

def ood_detector_correctness_prediction_accuracy(batch_size, shift="normal"):
    df = load_all(prefix="final_data", batch_size=batch_size, shift=shift, samples=100)
    df = df[df["shift_intensity"].isin(["InD", "OoD", "0.30000000000000004"])] #extract only maximum shifts
    df = df[df["fold"]!="train"]
    # data = data[data["shift"]!="noise"]
    # data["ood"] = data["correct_prediction"]
    for dataset in DATASETS:
        data_dict = []
        data_dataset = df[df["Dataset"] == dataset]
        with tqdm(total=df["feature_name"].nunique()*2 * 2, desc=f"Computing for {dataset}") as pbar:
            for feature in DSDS:
                data_filtered = data_dataset[data_dataset["feature_name"]==feature]
                if data_filtered.empty:
                    print(f"No data for {dataset} {feature}")
                    continue
                for ood_perf in [True, False]:
                    for perf_calibrated in [True, False]:
                        if perf_calibrated and not ood_perf:
                            continue  # unimportant
                        for threshold_method in THRESHOLD_METHODS:
                            for ood_val_fold in data_filtered["shift"].unique():
                                data_copy = data_filtered.copy()
                                if ood_val_fold in ["train", "ind_val", "ind_test"]:
                                    # dont calibrate on ind data or synthetic ood data
                                    continue
                                data_train = data_copy[
                                    (data_copy["shift"] == ood_val_fold) | (data_copy["shift"] == "ind_val")]
                                dsd = OODDetector(data_train, ood_val_fold, threshold_method=threshold_method)
                                # dsd.kde()
                                for ood_test_fold in data_filtered["shift"].unique():
                                    if ood_test_fold in ["train", "ind_val", "ind_test"]:
                                        continue
                                    if perf_calibrated:
                                        data_copy["ood"]=~data_copy["correct_prediction"]
                                    data_test = data_copy[(data_copy["shift"]==ood_test_fold)|(data_copy["shift"]=="ind_test")]
                                    if ood_perf and not perf_calibrated:
                                        data_copy["ood"]=~data_copy["correct_prediction"]
                                    tpr, tnr, ba = dsd.get_metrics(data_test)
                                    if np.isnan(ba):
                                        continue
                                    data_dict.append(
                                        {"Dataset": dataset, "feature_name": feature, "Threshold Method": threshold_method,
                                         "OoD==f(x)=y": ood_perf, "Performance Calibrated": perf_calibrated,
                                         "OoD Val Fold": ood_val_fold, "OoD Test Fold":ood_test_fold, "tpr": tpr, "tnr": tnr, "ba": ba}
                                    )
                                    pbar.set_description(f"Computing for {dataset}, {feature} {ood_perf} {ood_test_fold}; current ba: {ba}")

                                # data_copy = data_copy[data_copy["ood_val_fold"]!=data_copy["shift"]]
                        pbar.update(1)

            data = pd.DataFrame(data_dict)
            data.replace(DSD_PRINT_LUT, inplace=True)
            data.to_csv(f"ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv", index=False)



def ood_verdict_accuracy_table(batch_size):
    dfs = []
    for dataset, feature in itertools.product(DATASETS, DSDS):
        dfs.append(pd.read_csv(f"ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv"))
    df = pd.concat(dfs)
    df["Organic"]=(~df["OoD Test Fold"].isin(SYNTHETIC_SHIFTS))&(~df["OoD Val Fold"].isin(SYNTHETIC_SHIFTS))
    df = df[~df["OoD Val Fold"].isin(SYNTHETIC_SHIFTS)]
    print(df.groupby(["Threshold Method", "OoD==f(x)=y", "Performance Calibrated"])[["ba"]].agg(["min", "mean", "max"]))
    # print(results)
    df = df[(~df["Performance Calibrated"])&(df["Threshold Method"]=="val_optimal")]
    #calibration comparison
    organic_results = df[df["Organic"]
                            ]  # only synthetic shifts
    # organic_results = organic_results[organic_results["OoD Test Fold"]==organic_results["OoD Val Fold"]]

    # print(organic_results.groupby(["Dataset", "feature_name", "OoD==f(x)=y"])["ba"].agg(["mean"]).reset_index())
    batched_consistency = df.groupby(["Dataset", "feature_name", "OoD==f(x)=y", "Organic" ])[["ba"]].agg(["mean"]).reset_index()
    print(batched_consistency)
    # print(df.columns)
    # print(df[~df["OoD Shift"].isin(SYNTHETIC_SHIFTS)].groupby(["OoD==f(x)=y", "Performance Calibrated", "Threshold Method"])[["balanced_accuracy"]].agg(["min", "mean", "max"]).reset_index())
    #
    # results = df[df["Threshold Method"] == "val_optimal"]
    # regular_ood = results[(results["OoD==f(x)=y"]==False) & (results["Performance Calibrated"]==False)]
    # correctness = results[(results["OoD==f(x)=y"]==True) & (results["Performance Calibrated"]==True)]
    # regular_ood = regular_ood.groupby(["Dataset", "feature_name"])[["balanced_accuracy"]].mean().reset_index()
    # regular_ood["correctness_ba"] = correctness.groupby(["Dataset", "feature_name"])[["balanced_accuracy"]].mean().reset_index()["balanced_accuracy"]
    # regular_ood["diff"] = regular_ood["balanced_accuracy"]-regular_ood["correctness_ba"]
    # print(regular_ood.groupby(["Dataset", "feature_name"])[["balanced_accuracy", "correctness_ba"]].mean().reset_index())
    # best_config = df[(df["Threshold Method"] == "ind_span")&(df["Performance Calibrated"])&(df["OoD==f(x)=y"])]
    # print(best_config.groupby(["Dataset", "feature_name", "OoD Shift"])["balanced_accuracy"].mean())


def ood_verdict_plots_batched():
    dfs = []
    for dataset, batch_size in itertools.product(DATASETS, BATCH_SIZES):
        df = pd.read_csv(f"ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv")
        df["batch_size"] = batch_size
        dfs.append(df)
    data = pd.concat(dfs)
    data = data[(data["Threshold Method"]=="val_optimal")&(data["Performance Calibrated"]==False)&data["OoD==f(x)=y"]==True]
    g = sns.FacetGrid(data, col="Dataset", margin_titles=True, sharex=False, sharey=False, col_wrap=3)
    g.map_dataframe(sns.lineplot, x="batch_size", y="ba", hue="feature_name", markers=True, dashes=False)
    for ax in g.axes.flat:
        ax.set_ylim(0.4, 1)

    g.add_legend(bbox_to_anchor=(0.7, 0.3), loc='center left', title="Feature", ncol=1)
    plt.savefig("batched_ood_verdict_accuracy.pdf")
    plt.show()


def get_ood_detector_data(data):
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

    data = get_ood_detector_data(df)

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


def eval_debiased_ood_detectors():
    data = load_all_biased(prefix="debiased_data")
    # print(data["batch_size"].unique())
    data = data[data["fold"] != "train"]
    data_dict = []

    for batch_size in BATCH_SIZES[1:]:
        for dataset in DATASETS:
            with tqdm(len(DSDS) * 3) as pbar:
                for feature in DSDS:
                    if feature!="rabanser":
                        continue
                    for k in [0,5]:
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
                                    # data_copy["ood"] = ~data_copy["correct_prediction"]  # OOD is the opposite of correct prediction
                                    tpr, tnr, ba = dsd.get_metrics(data_test)
                                    if np.isnan(ba):
                                        continue
                                    data_dict.append(
                                        {"Dataset": dataset, "feature_name": feature,
                                         "OoD Val Fold": ood_val_fold, "OoD Test Fold": ood_test_fold, "tpr": tpr, "tnr": tnr,
                                         "ba": ba, "bias": bias, "k":k, "batch_size":batch_size}
                                    )

                            # data_copy["ood"] = ~data_copy["correct_prediction"]  #

                        pbar.update(1)

    df = pd.DataFrame(data_dict)
    # print(data.head(10))
    df.replace(DSD_PRINT_LUT, inplace=True)
    df.replace(SAMPLER_LUT, inplace=True)
    df.to_csv(f"ood_detector_data/debiased_ood_detector_correctness_{dataset}.csv", index=False)
    df = df.groupby(["batch_size", "Dataset", "feature_name", "bias", "k"])["ba"].mean().reset_index()
        # print(data)
    g = sns.FacetGrid(df, col="Dataset")
    g.map_dataframe(sns.boxplot, x="bias", y="ba", hue="k", palette=sns.color_palette())
    g.add_legend()
    plt.show()
    return data


def debiased_plots():

    across_batch_sizes = []
    for batch_size in BATCH_SIZES[1:]:
        data = eval_debiased_ood_detectors(batch_size)
        data["Batch Size"] = batch_size
        across_batch_sizes.append(data)
        table = data.copy()

        # g = sns.FacetGrid(table, col="Dataset")
        # g.map_dataframe(sns.boxplot, x="feature_name", y="balanced_accuracy", hue="k", palette=sns.color_palette())
        # plt.legend()
        # plt.show()
        #
        # g = sns.FacetGrid(table, col="Dataset")
        # g.map_dataframe(sns.boxplot, x="k", y="balanced_accuracy", hue="k", palette=sns.color_palette())
        # plt.legend()
        # plt.show()
        # # table.replace(SAMPLER_LUT, inplace=True)
        # #
        # g = sns.FacetGrid(table, col="Dataset", col_wrap=3, margin_titles=True, sharex=False, sharey=False)
        # g.map_dataframe(sns.boxplot, x="bias", y="balanced_accuracy", hue="k", palette=sns.color_palette())
        # g.add_legend(bbox_to_anchor=(0.7, 0.3), loc='center left', title="k", ncol=1)
        # plt.savefig("debiased_ood_detector_bias_boxplots.pdf")
        # plt.show()
        # print(balanced.groupby(["Dataset", "feature_name",  "k", "bias",]).mean())
    data = pd.concat(across_batch_sizes)
    print(data.groupby(["Batch Size", "Dataset", "feature_name", "bias", "k",])[["balanced_accuracy"]].mean().reset_index())
    data = data[data["k"]==-1]
    data = data[~((data["Dataset"]=="CCT")&(data["bias"]=="SequentialSampler"))]
    data.replace(SAMPLER_LUT, inplace=True)
    g = sns.FacetGrid(data, col="Dataset", margin_titles=True, sharex=False, sharey=False, col_wrap=3)
    g.map_dataframe(sns.boxplot, x="bias", y="balanced_accuracy", hue="Batch Size", palette=sns.color_palette(), order=["Unbiased", "Synthetic", "Temporal", "Class"])
    g.add_legend(bbox_to_anchor=(0.7, 0.25), loc='center left', title="Batch Size", ncol=1)
    plt.tight_layout()
    plt.savefig("ood_detector_bias_boxplots.pdf")
    # for ax in g.axes.flat:
    #     ax.legend()
    plt.show()


def show_thresholding_problems():
    df = load_dfs(100, simulate=True)
    df = df[df["KS"] == False]
    df.loc[df["fold"] == "ind", "Shift"] = "ind"
    df = df[df["fold"].isin(["ind", "ood", "noise_0.25", "dropout_0.25"])]
    df["ind"] = df["fold"] == "ind"

    # Explicit hue order
    # hue_order = ['ind', 'noise', 'normal', 'dropout']
    # hue_order=["ind", ""]
    g = sns.FacetGrid(df, col="feature_name", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.kdeplot, x="feature", common_norm=False)

    # Manually add legend if automatic is not working
    plt.legend(title="Shift Category")

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

if __name__ == '__main__':
    #accuracies on each dataset
    from experiments.dataset_analysis import *
    accuracy_table()
    #class distribution
    # dataset_summaries()
    """
        Runtime Verification
    """
    # loss_correctness_test()
    # for batch_size in BATCH_SIZES:
    #     print(f"Running batch size {batch_size}")
    #     ood_detector_correctness_prediction_accuracy(batch_size, shift="")
        # ood_verdict_accuracy_table(batch_size)
    #     input()
            # input()#&data_copy["ood"] #

    #batching
    # ood_verdict_plots_batched()

# ood_detector_correctness_prediction_accuracy(64)
        # ood_verdict_accuracy_table(32)
        # ood_verdict_accuracy_table(batch)



    # for batch_size in [1, 8, 16]:
    #     ood_detector_correctness_prediction_accuracy(batch_size)


    #runtime verification
    # plot_batching_effect("NICO", "entropy")
    eval_debiased_ood_detectors()
    # eval_debiased_ood_detectors(8)
    # debiased_plots()

    #loss regression
    # get_gam_data(load=False)
    # regplots(64)


    # get_datasetwise_risk()
    # iou_distribution()
    # compare_kdes()
    # for i in range(3):
    # verdictwise_proportions(cal_idx=1, batch_size=1)
    # loss_verdict_histogram(1)


    # ood_verdict_plots_batched()


    # collect_rate_estimator_data()
    # eval_rate_estimator()
    # t_check()
    # fetch_dsd_accuracies(1, plot=False)
    # ood_verdict_accuracy_table(1)
    # plot_dsd_accuracies(1000)
    # plot_rate_estimation_errors_for_dsds()

    # accuracy_by_fold_and_dsd_verdict()
    #print(data)

    # collect_tpr_tnr_sensitivity_data()
    # collect_dsd_accuracy_estimation_data()
    # uniform_bernoulli(data, load = False)
    # show_rate_risk()
    # cost_benefit_analysis()