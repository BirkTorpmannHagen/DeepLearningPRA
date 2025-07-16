import itertools
from itertools import combinations, count
from os import listdir

import pygam
from matplotlib import pyplot as plt, patches as patches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import numpy as np
# from albumentations.random_utils import normal
from matplotlib.pyplot import yscale
from numpy.ma.core import product
from scipy.cluster.hierarchy import single
from scipy.stats import spearmanr

from seaborn import FacetGrid
import warnings

from plots import load_dfs
from utils import BATCH_SIZES

warnings.filterwarnings("ignore")
from datasets.polyps import CVC_ClinicDB, EndoCV2020
from riskmodel import UNNECESSARY_INTERVENTION
from simulations import *
import matplotlib.pyplot as plt
from utils import *
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from components import OODDetector, DebiasedOODDetector
from multiprocessing import Pool
import matplotlib.patches as mpatches

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


def collect_tpr_tnr_sensitivity_data():
    bins = 10
    total_num_tpr_tnr = np.sum(
    [i + j / 2 >= 0.5 for i in np.linspace(0, 1, bins) for j in np.linspace(0, 1, bins)])
    for dataset in DATASETS:
        dfs = []

        data = load_pra_df(dataset_name=dataset, feature_name="knn", batch_size=1,
                           samples=1000)  # we are just interested in the loss and oodness values, knn is arbitray
        ood_sets = data[~data["shift"].isin(["ind_val", "ind_test", "train"])]["shift"].unique()

        with tqdm(total=bins**2*(len(ood_sets)-1)*len(ood_sets)) as pbar:
            for val_set in ood_sets:
                for test_set in ood_sets:
                    if test_set=="noise":
                        continue #used only to estimate accuracies
                    for rate in np.linspace(0, 1, bins):
                        for ba in np.linspace(0.5, 1, bins):
                            sim = UniformBatchSimulator(data, ood_test_shift=test_set, ood_val_shift=val_set, estimator=BernoulliEstimator, dsd_tpr=ba, dsd_tnr=ba)
                            results = sim.sim(rate, 600)
                            results = results.groupby(["Tree"]).mean().reset_index()

                            # results = results.mean()
                            results["tpr"] = ba
                            results["tnr"] = ba
                            results["ba"] = ba
                            results["rate"] = rate
                            results["test_set"] = test_set
                            results["val_set"] = val_set
                            results["Dataset"] = dataset
                            # results = results.groupby(["tpr", "tnr", "rate", "test_set", "val_set", "Tree"]).mean().reset_index()

                            dfs.append(results)
                            pbar.update(1)
        df_final = pd.concat(dfs)
        print(df_final.head(10))
        df_final.to_csv(f"pra_data/{dataset}_sensitivity_results.csv")

def collect_dsd_accuracy_estimation_data():

    bins=11
    for batch_sizes in BATCH_SIZES:
        dsd_accuracies = fetch_dsd_accuracies(batch_size=batch_sizes, plot=False, samples=1000)
        dsd_accuracies = dsd_accuracies.groupby(["Dataset", "DSD"])[["tpr", "tnr", "ba"]].mean().reset_index()
        best_dsds = dsd_accuracies.loc[dsd_accuracies.groupby("Dataset")["ba"].idxmax()].reset_index(drop=True)
        print(best_dsds)
        for dataset in DATASETS:
            dfs = []
            dsd = best_dsds[(best_dsds["Dataset"]==dataset)]["DSD"].values[0]
            filt = dsd_accuracies[(dsd_accuracies["Dataset"]==dataset)&(dsd_accuracies["DSD"]==dsd)]
            tpr, tnr, ba = filt["tpr"].mean(), filt["tnr"].mean(), filt["ba"].mean()
            print(f"Dataset: {dataset}, dsd:{dsd}, ba: {ba}")

            data = load_pra_df(dataset, dsd, batch_size=batch_sizes, samples=1000)
            ood_sets = data[~data["shift"].isin(["ind_val", "ind_test", "train"])]["shift"].unique()
            with tqdm(total=bins*len(ood_sets)*(len(ood_sets)-1)) as pbar:
                for val_set in ood_sets:
                    for test_set in ood_sets:
                        if test_set=="noise":
                            continue
                        pool = Pool(bins)
                        print("multiprocessing...")
                        results = pool.starmap(simulate_dsd_accuracy_estimation, [(data, rate, val_set, test_set, tpr, tnr, ba, dsd) for rate in np.linspace(0, 1, bins)])
                        pool.close()
                            # results = results.groupby(["tpr", "tnr", "rate", "test_set", "val_set", "Tree"]).mean().reset_index()
                        for result in results:
                            dfs.append(result)
                            pbar.update(1)
            df_final = pd.concat(dfs)
            print(df_final.head(10))
            df_final.to_csv(f"pra_data/dsd_results_{dataset}_{batch_sizes}.csv")


def plot_ba_rate_sensitivity():
    df = pd.read_csv("tpr_tnr_sensitivity.csv").groupby(["tpr", "tnr", "rate"]).mean().reset_index()
    df["tpr"] = df["tpr"].round(2)
    df["tnr"] = df["tnr"].round(2)
    df.rename(columns={"rate": "Bernoulli Expectation"}, inplace=True)

    df["DSD Accuracy"] = (df["tpr"] + df["tnr"]) / 2
    df = df.groupby(["DSD Accuracy", "Bernoulli Expectation"]).mean().reset_index() # Average over tpr and tnr
    df["Error"] = np.abs(df["Risk Estimate"] - df["True Risk"])/df["True Risk"]
    sns.lineplot(df, x="DSD Accuracy", y="Error")
    plt.show()
    df = df.sort_values(by=["DSD Accuracy", "Bernoulli Expectation"])

    pivot_table = df.pivot(index="Bernoulli Expectation", columns="DSD Accuracy", values="Error")

    # Reverse the `tpr` axis (y-axis) order

    pivot_table = pivot_table.loc[::-1]
    sns.heatmap(pivot_table)
    plt.xticks([0, df["DSD Accuracy"].nunique()], [0,1], rotation=0)  # x-axis: only 0 and 1
    plt.yticks([0, df["Bernoulli Expectation"].nunique()], [1, 0])  # y-axis: only 0 and 1
    plt.savefig("ba_sensitivity.eps")
    plt.show()

def fetch_dsd_accuracies(batch_size=32, plot=False, samples=1000):
    data = []
    with tqdm(total=len(DATASETS) * len(DSDS)) as pbar:
        for dataset in DATASETS:
            for feature in DSDS:
                df = load_pra_df(dataset, feature, batch_size=batch_size, samples=samples)
                # df["ood"]=df["correct_prediction"]==False
                df = df[df["shift"]!="noise"]
                if df.empty:
                    continue
                ind_val = df[df["shift"]=="ind_val"]
                ind_test = df[df["shift"]=="ind_test"]
                ood_folds = df[~df["fold"].isin(["ind_val", "ind_test", "train"])]["shift"].unique()
                for ood_val_fold in ood_folds:
                    for ood_test_fold in ood_folds:
                        ood_val = df[df["shift"]==ood_val_fold]
                        ood_test = df[df["shift"]==ood_test_fold]

                        dsd = OODDetector(df, ood_val_fold)
                        test = pd.concat([ind_test, ood_test])
                        tpr, tnr, ba = dsd.get_metrics(test)
                        threshold = dsd.threshold

                        if ood_test_fold == ood_folds[0] and plot:
                            plt.hist(ind_val["feature"], bins=100, alpha=0.5, label="ind_val", density=True)
                            plt.hist(ind_test["feature"], bins=100, alpha=0.5, label="ind_test", density=True )
                            plt.hist(ood_val["feature"], bins=100, alpha=0.5, label=ood_val_fold, density=True)
                            plt.hist(ood_test["feature"], bins=100, alpha=0.5, label=ood_test_fold, density=True)

                            plt.axvline(threshold, color="red", label="Threshold")
                            plt.title(f"{dataset} {feature} {ood_val_fold} {ood_test_fold}")
                            plt.legend()
                            plt.show()
                        data.append({"Dataset": dataset, "DSD":feature, "val_fold": ood_val_fold, "test_fold":ood_test_fold, "tpr": tpr, "tnr": tnr, "ba": ba, "t":threshold})
                        pbar.update(1)
    df = pd.DataFrame(data)

    df.to_csv("dsd_accuracies.csv")
    df = df[df["val_fold"]!=df["test_fold"]]  # remove cases where val and test folds are the same
    df.replace(DSD_PRINT_LUT, inplace=True)
    print(df.groupby(["Dataset", "DSD"])[["tpr", "tnr", "ba"]].mean())
    # print(df.groupby(["Dataset", "DSD", "val_fold", "test_fold"])[["tpr", "tnr", "ba"]].mean())
    return df

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


def collect_rate_estimator_data():
    data = []
    with tqdm(total=26*26*26*9) as pbar:
        for rate in tqdm(np.linspace(0, 1, 26)):
            for tpr in np.linspace(0, 1, 26):
                for tnr in np.linspace(0, 1, 26):
                    for tl in [10, 20, 30, 50, 60, 100, 200, 500, 1000]:
                        ba = round((tpr + tnr) / 2,2)
                        if ba <= 0.5:
                            continue
                        pbar.update(1)
                        re = BernoulliEstimator(prior_rate=rate, tpr=tpr, tnr=tnr)
                        sample = re.sample(10_000, rate)
                        dsd = get_dsd_verdicts_given_true_trace(sample, tpr, tnr)
                        for i in np.array_split(dsd, int(10_000//tl)):
                            re.update(i)
                            rate_estimate = re.get_rate()
                            error = np.abs(rate-rate_estimate)
                            data.append({"rate": rate, "ba": ba,  "tl": tl, "rate_estimate": rate_estimate, "error":error})
    df = pd.DataFrame(data)
    df = df.groupby(["ba","tl", "rate"]).mean()
    df.to_csv("rate_estimator_eval.csv")

def eval_rate_estimator():
    df = pd.read_csv("rate_estimator_eval.csv")
    df_barate = df.groupby(["ba", "rate"]).mean().reset_index()
    pivot_table = df_barate.pivot(index="ba", columns="rate", values="error")
    pivot_table = pivot_table.loc[::-1]
    sns.heatmap(pivot_table, vmin=0, vmax=1)
    plt.savefig("rate_sensitivity.eps")
    plt.tight_layout()
    plt.show()

    #rate x ba

def plot_rate_estimation_errors_for_dsds(batch_size=16, cross_validate=False):
    data = []
    dsd_data = fetch_dsd_accuracies(batch_size)
    if not cross_validate:
        dsd_data = dsd_data[dsd_data["val_fold"]==dsd_data["test_fold"]]
    # print(dsd_data.groupby(["Dataset", "DSD", "val_fold", "test_fold"])[["tpr", "tnr", "ba"]].mean())
    # input()
    with tqdm(total=len(DATASETS) * len(DSDS) * 26) as pbar:
        for dataset in DATASETS:
            for feature in DSDS:
                for rate in tqdm(np.linspace(0, 1, 26)):
                    subdata = dsd_data[(dsd_data["Dataset"]==dataset)&(dsd_data["DSD"]==feature)]
                    tpr, tnr, ba = subdata["tpr"].mean(), subdata["tnr"].mean(), subdata["ba"].mean()
                    for tl in [10, 20, 30, 50, 60, 100, 200, 500, 1000]:
                        re = BernoulliEstimator(prior_rate=rate, tpr=tpr, tnr=tnr)
                        sample = re.sample(10_000, rate)
                        dsd = get_dsd_verdicts_given_true_trace(sample, tpr, tnr)
                        for i in np.array_split(dsd, int(10_000 // tl)):
                            re.update(i)
                            rate_estimate = re.get_rate()
                            error = np.abs(rate - rate_estimate)
                            data.append(
                                {"Dataset": dataset, "DSD":feature, "rate": rate, "ba": ba, "tl": tl, "rate_estimate": rate_estimate, "error": error})
                        pbar.update(1)

    df = pd.DataFrame(data)
    df.replace(DSD_PRINT_LUT, inplace=True)

    df = df.groupby(["Dataset", "DSD", "tl", "rate"]).mean().reset_index()
    # df = df[df["tl"]==100]
    print(df)
    df = df
    g = sns.FacetGrid(df, col="Dataset")
    g.map_dataframe(sns.lineplot, x="rate", y="error", hue="DSD")
    # g.add_legend()
    g.tight_layout()
    plt.legend(frameon=True, ncol=len(np.unique(df["DSD"])), loc="upper center", bbox_to_anchor = (-2, -0.15))
    # plt.tight_layout(w_pad=0.5)Note that these error estimates are computed based
    plt.subplots_adjust(bottom=0.3)
    plt.savefig("dsd_rate_error.pdf")
    plt.show()
    df.to_csv("rate_estimator_eval.csv")


def plot_rate_estimates():
    df = pd.read_csv("rate_estimator_eval.csv")
    df["ba"] = np.round((df["tpr"] + df["tnr"]) / 2, 2)
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df["error"] = np.abs(df["rate"] - df["rate_estimate"])

    df = df[df["tl"] == 10]
    df = df[df["ba"]>0.5]
    df = df.groupby(["ba", "rate"]).mean().reset_index()
    print(df)
    pivot_table = df.pivot(index="ba", columns="rate", values="error")
    pivot_table = pivot_table.loc[::-1]
    sns.heatmap(pivot_table)

    # plt.xticks([0, df["ba"].nunique()], [0, 1], rotation=0)  # x-axis: only 0 and 1
    # plt.yticks([0, df["rate"].nunique()], [1, 0])  # y-axis: only 0 and 1
    plt.savefig("rate_sensitivity.eps")
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

def accuracy_table():
    df = load_all(1, samples=1000, prefix="final_data")
    df = df[df["shift"]!="noise"]
    accs = df.groupby(["Dataset", "shift"])["correct_prediction"].mean().reset_index()
    print(accs)
    return accs



def t_check():
    df = load_pra_df("Polyp", "knn",batch_size=32)
    df = df[df["shift"]!="noise"]
    for shift in df["shift"].unique():
        print(f"plotting : {shift}")

        plt.hist(df[df["shift"]==shift]["feature"], bins=100, alpha=0.5, density=True, label=shift)
        plt.legend()
    plt.show()

def show_rate_risk():
    data = load_pra_df("Polyp", "knn", batch_size=1, samples=1000)
    oods = data[~data["shift"].isin(["ind_val", "ind_test", "train", "noise"])]["shift"].unique()
    rates = np.linspace(0, 1, 11)
    dfs = []
    with tqdm(total=len(oods)*(len(oods)-1)*len(rates)) as pbar:
        for ood_val_set, ood_test_set, rate in itertools.product(oods, oods, rates):
            if ood_val_set == ood_test_set:
                continue
            sim = UniformBatchSimulator(data, ood_test_shift=ood_test_set, ood_val_shift=ood_val_set, maximum_loss=0.5, estimator=BernoulliEstimator, use_synth=False, dsd_tpr=0.9, dsd_tnr=0.9)
            results = sim.sim(rate, 600)
            results["Rate"] = rate
            results["Risk Error"]=results["Risk Estimate"]-results["True Risk"]
            dfs.append(results)
            pbar.update()

    df = pd.concat(dfs)
    df = df.groupby(["Tree", "Rate"]).mean().reset_index()
    print(df)
    ax = sns.lineplot(df, x="Rate", y="Risk Estimate", hue="Tree")
    for tree in df["Tree"].unique():
        df_tree = df[df["Tree"]==tree]
        ax.fill_between(df_tree["Rate"],
                        df_tree["Risk Estimate"],
                        df_tree["True Risk"],
                        alpha=0.2)
        # plt.plot(df_tree["Rate"], df_tree["True Risk"], label=f"{tree} True Risk", linestyle="dashed")
    ax.axhline(UNNECESSARY_INTERVENTION, color="red", label="Manual Intervention")
    plt.legend()
    plt.savefig("rate_risk.pdf")
    plt.show()

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

def load_polyp_data():
    dfs = []
    for dsd_name in ["energy", "knn", "grad_magnitude", "cross_entropy", "mahalanobis"]:
        for model in ["deeplabv3plus", "unet", "segformer"]:
            for ood_val in ["CVC-ClinicDB", "EndoCV2020", "EtisLaribDB"]:
                df = load_pra_df(dataset_name="Polyp", feature_name=dsd_name, model=model, batch_size=1, samples=1000)
                print(df.head(10))
                dsd = OODDetector(df, ood_val)
                df["verdict"] = df.apply(lambda row: dsd.predict(row), axis=1)
                df["ood_val"] = ood_val
                df["IoU"] = 1 - df["loss"]
                df["model"] = model
                df["feature_name"] = dsd_name
                dfs.append(df)
    df = pd.concat(dfs)
    # df = df[df["shift"] != "train"]
    print(df.head(10))
    df.replace(DSD_PRINT_LUT, inplace=True)
    return df

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

def ood_detector_correctness_prediction_accuracy(batch_size):
    data = load_all(prefix="final_data", batch_size=batch_size)
    print(data["shift_intensity"].unique())
    data = data[data["shift_intensity"].isin(["InD", "OoD", "0.30000000000000004"])] #extract only maximum shifts
    data = data[data["fold"]!="train"]
    data = data[data["shift"]!="noise"]
    # data = data[data["shift"]!="noise"]
    # data["ood"] = data["correct_prediction"]
    dfs = []
    with tqdm(total=len(DATASETS)*len(DSDS)*len(THRESHOLD_METHODS)*2*2) as pbar:
        for ood_perf in [True, False]:
            for perf_calibrated in [True, False]:
                for threshold_method in THRESHOLD_METHODS:
                    if threshold_method=="density":
                        continue
                    for dataset in DATASETS:
                        for feature in DSDS:
                            data_dataset = data[(data["Dataset"]==dataset) & (data["feature_name"]==feature)]


                            for ood_val_fold in data_dataset["shift"].unique():
                                data_copy = data_dataset.copy()
                                print(data_copy["shift"].unique())
                                if ood_val_fold in ["train", "ind_val", "ind_test"] or ood_val_fold in SYNTHETIC_SHIFTS :
                                    #dont calibrate on ind data or synthetic ood data
                                    continue
                                if perf_calibrated and not ood_perf:
                                    continue #unimportant

                                if perf_calibrated:
                                    data_copy["ood"]=~data_copy["correct_prediction"] #
                                data_train = data_copy[(data_copy["shift"]==ood_val_fold)|(data_copy["shift"]=="ind_val")]

                                dsd = OODDetector(data_train, ood_val_fold, threshold_method=threshold_method)

                                data_copy["detected_ood"] = data_copy.apply(lambda row: dsd.predict(row), axis=1)
                                data_copy["ood_val_fold"] = ood_val_fold
                                data_copy["Threshold Method"] = threshold_method
                                data_copy["OoD==f(x)=y"] = ood_perf
                                data_copy["Performance Calibrated"] = perf_calibrated
                                # if dataset=="CCT" and feature=="knn" and threshold_method=="val_optimal":
                                #     dsd.kde()
                                # sns.kdeplot(data_copy, x="feature", hue="ood")
                                # plt.title(f"{dataset} {feature} {ood_perf} {perf_calibrated} {threshold_method}")
                                # plt.axvline(dsd.threshold)
                                # plt.show()
                                if ood_perf and not perf_calibrated:
                                    data_copy["ood"]=~data_copy["correct_prediction"] #
                                data_copy["correct_ood_detection"] = data_copy["ood"] == data_copy["detected_ood"]

                                data_copy = data_copy[data_copy["ood_val_fold"]!=data_copy["shift"]]
                                dfs.append(data_copy)
                            pbar.update(1)
    data = pd.concat(dfs)
    # print(data)
    # data = data[data["ood_val_fold"]==data["shift"]]
    data.replace(DSD_PRINT_LUT, inplace=True)
    tprs = data[~data["shift"].isin(["ind_val", "ind_test"])].groupby(["Dataset", "feature_name","Threshold Method", "OoD==f(x)=y","Performance Calibrated", "shift"])["correct_ood_detection"].mean().reset_index().groupby(["Dataset", "feature_name", "Threshold Method", "OoD==f(x)=y","Performance Calibrated"])["correct_ood_detection"].mean().reset_index()

    tnrs = data[data["shift"].isin(["ind_val", "ind_test"])].groupby(["Dataset", "feature_name","Threshold Method", "OoD==f(x)=y","Performance Calibrated", "shift"])["correct_ood_detection"].mean().reset_index().groupby(["Dataset", "feature_name", "Threshold Method", "OoD==f(x)=y", "Performance Calibrated"])["correct_ood_detection"].mean().reset_index()

    tprs = tprs.rename(columns={"correct_ood_detection": "TPR"})
    tnrs = tnrs.rename(columns={"correct_ood_detection": "TNR"})
    tprs.to_csv(f"ood_detector_data/tprs_{batch_size}.csv", index=False)
    tnrs.to_csv(f"ood_detector_data/tnrs_{batch_size}.csv", index=False)
    # Merge on Dataset and feature_name
    balanced = pd.merge(tprs.groupby(["Dataset", "feature_name","Threshold Method", "OoD==f(x)=y","Performance Calibrated"])["TPR"].mean(), tnrs.groupby(["Dataset", "feature_name","Threshold Method", "OoD==f(x)=y","Performance Calibrated"])["TNR"].mean(), on=["Dataset", "feature_name", "Threshold Method", "OoD==f(x)=y","Performance Calibrated"])
    # Compute balanced accuracy
    balanced["balanced_accuracy"] = (balanced["TPR"] + balanced["TNR"]) / 2
    print(balanced)

    balanced.reset_index().to_csv(f"ood_detector_data/ood_detector_correctness_{batch_size}.csv", index=False)
    return balanced
    # print(balanced)
    # plot = balanced[(balanced["OoD==f(x)=y"]==True) & (balanced["Performance Calibrated"]==True) & balanced["Threshold Method"]=="val_optimal"]

    # print(data.groupby(["Dataset", "feature_name", "shift"])["correct_ood_detection"].mean().reset_index().groupby(["Dataset", "feature_name"])["correct_ood_detection"].mean().reset_index())




def iou_distribution():

    df = load_polyp_data()
    df = df[df["shift"] != "train"]
    df = df[df["shift"] != "ind_val"]
    df.replace({"ind_test":"Kvasir"}, inplace=True)
    palette = sns.color_palette("tab10", n_colors=df["Model"].nunique())
    g = sns.FacetGrid(df, col="shift", palette=palette)
    # g = sns.FacetGrid(df, col="shift")
    g.map_dataframe(sns.kdeplot, x="IoU", hue="Model", clip=(0, 1), fill=True, multiple="stack", common_norm=False)
    # manually extract legend from one of the axes
    # g._legend.remove()  # in case it partially shows
    df.replace({"deeplabv3plus":"DeepLabV3+", "unet":"UNet++", "segformer":"SegFormer"}, inplace=True)
    model_names = sorted(df["Model"].unique())
    colors = sns.color_palette("tab10", n_colors=len(model_names))
    color_dict = dict(zip(model_names, colors))
    patches = [mpatches.Patch(color=color_dict[m], label=m) for m in model_names]
    plt.legend(handles=patches, title="Model", frameon=True,
               loc="best", ncol=1)
    plt.savefig("IoU_distributions.pdf")
    plt.show()

# def ood_verdict_accuracy_table(batch_size):
    results = pd.read_csv(f"ood_detector_data/ood_detector_correctness_{batch_size}.csv")
    # print(results)
    print(results.groupby(["OoD==f(x)=y", "Performance Calibrated", "Threshold Method"])[["balanced_accuracy"]].agg(["min", "mean", "max"]).reset_index())

    results = results[results["Threshold Method"] == "val_optimal"]
    regular_ood = results[(results["OoD==f(x)=y"]==False) & (results["Performance Calibrated"]==False)]
    correctness = results[(results["OoD==f(x)=y"]==True) & (results["Performance Calibrated"]==True)]
    regular_ood = regular_ood.groupby(["Dataset", "feature_name"])[["balanced_accuracy"]].mean().reset_index()

    regular_ood["correctness_ba"] = correctness.groupby(["Dataset", "feature_name"])[["balanced_accuracy"]].mean().reset_index()["balanced_accuracy"]
    regular_ood["diff"] = regular_ood["balanced_accuracy"]-regular_ood["correctness_ba"]
    print(regular_ood.groupby(["Dataset"])[["balanced_accuracy", "correctness_ba"]].max().reset_index())
    # df_ood_label = df[df["OoD==f(x)=y"]==True]
    #
    # plotter = pd.concat([df[df[]]])
    # sns.scatterplot(data=results, y="balanced_accuracy", hue="Dataset", style="feature_name", size="balanced_accuracy", sizes=(20, 200), alpha=0.7)

def ood_verdict_plots_batched():
    dfs = []
    for i in BATCH_SIZES:
        df = pd.read_csv(f"ood_detector_correctness_{i}.csv")
        df["batch_size"] = i
        print(df)
        dfs.append(df)
    data = pd.concat(dfs)
    data = data[(data["Threshold Method"]=="val_optimal")&(data["Performance Calibrated"]==False)&data["OoD==f(x)=y"]==True]
    data = data[data["feature_name"]!="Mahalanobis"]
    print(data.groupby(["Dataset", "batch_size"])[["balanced_accuracy"]].max().reset_index())
    g = sns.FacetGrid(data, col="Dataset", margin_titles=True, sharex=False, sharey=False, col_wrap=3)
    g.map_dataframe(sns.lineplot, x="batch_size", y="balanced_accuracy", hue="feature_name", markers=True, dashes=False)
    for ax in g.axes.flat:
        ax.set_ylim(0.4, 1)

    g.add_legend(bbox_to_anchor=(0.7, 0.3), loc='center left', title="Feature", ncol=1)
    plt.savefig("batched_ood_verdict_accuracy.pdf")
    plt.show()

def compare_kdes():
    df = load_all(prefix="final_data")
    df = df[df["Dataset"]=="CCT"]

    g = sns.FacetGrid(df, col="feature_name", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.histplot, x="feature", hue="correct_prediction", fill=True, element="step")
    plt.show()
    g = sns.FacetGrid(df, col="feature_name", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.histplot, x="feature", hue="fold", fill=True, element="step")
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


def bias_correctness_test():
    data = load_all_biased(prefix="final_data")
    data = data[data["fold"] != "train"]
    data = data[data["shift"] != "noise"]

    dfs = []
    with tqdm(total=len(DATASETS) * len(DSDS)) as pbar:
        for dataset in DATASETS:
            for feature in DSDS:
                data_dataset = data[(data["Dataset"] == dataset) & (data["feature_name"] == feature)]
                for ood_val_fold in data_dataset["shift"].unique():
                    data_copy = data_dataset.copy()
                    if ood_val_fold in ["train", "ind_val", "ind_test"]:
                        continue
                    data_train = data_copy[
                        (data_copy["shift"] == ood_val_fold) | (data_copy["shift"] == "ind_val")&(data_copy["bias"]=="Unbiased")]
                    dsd = OODDetector(data_train, ood_val_fold)

                    data_copy["Verdict"] = data_copy.apply(lambda row: dsd.predict(row), axis=1)
                    data_copy["ood_val_fold"] = ood_val_fold
                    data_copy["ood"] = ~data_copy["correct_prediction"]  #
                    data_copy["correct_ood_detection"] = data_copy["Verdict"] == data_copy["ood"]
                    data_copy = data_copy[data_copy["ood_val_fold"] != data_copy["shift"]]
                    dfs.append(data_copy)
                pbar.update(1)
    data = pd.concat(dfs)

    tprs = data[~data["shift"].isin(["ind_val", "ind_test"])].groupby(["Dataset", "feature_name", "shift", "bias" ])["correct_ood_detection"].mean().reset_index().groupby(["Dataset", "feature_name", "bias"])["correct_ood_detection"].mean().reset_index()

    tnrs = data[data["shift"].isin(["ind_val", "ind_test"])].groupby(["Dataset", "feature_name", "shift", "bias"])["correct_ood_detection"].mean().reset_index().groupby(["Dataset", "feature_name", "bias"])["correct_ood_detection"].mean().reset_index()

    tprs = tprs.rename(columns={"correct_ood_detection": "TPR"})
    tnrs = tnrs.rename(columns={"correct_ood_detection": "TNR"})
    balanced = pd.merge(tprs, tnrs,
                        on=["Dataset", "feature_name", "bias"])

    # Compute balanced accuracy
    balanced["balanced_accuracy"] = (balanced["TPR"] + balanced["TNR"]) / 2
    balanced.replace(DSD_PRINT_LUT, inplace=True)

    print(balanced.groupby(["Dataset", "feature_name", "bias"])["balanced_accuracy"].mean().reset_index())


def dataset_summaries():
    data = load_all(1)
    data = data[data["shift"].isin(["ind_val", "ind_test", "train"])]
    data = data[data["feature_name"]=="energy"] #random
    data = data[data["Dataset"]!="Polyp"] #polyp is special

    print(data)
    g = sns.FacetGrid(data, col="Dataset", height=3, aspect=1.5, sharex=False, sharey=False, col_wrap=2)
    g.map_dataframe(sns.countplot, x="class")
    for ax in g.axes.flat:
        dataset_name_for_ax = ax.get_title().split(" = ")[-1]
        ax.set_title(dataset_name_for_ax)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        # Set x-ticks to be the class names
        ax.set_xticklabels([])
    plt.savefig("class_distribution.pdf")
    plt.show()

def eval_debiased_ood_detectors(batch_size):
    data = load_all_biased(prefix="debiased_data")
    # print(data["batch_size"].unique())
    data = data[data["batch_size"]==batch_size]
    data = data[data["fold"] != "train"]
    data = data[data["shift"] != "noise"]
    dfs = []
    with tqdm(total=len(DATASETS) * len(DSDS)*3) as pbar:
        for dataset in DATASETS:
            for feature in DSDS:
                for k in data["k"].unique():
                    data_dataset = data[(data["Dataset"] == dataset) & (data["feature_name"] == feature) & (data["k"]==k)]

                    if data_dataset.empty:

                        continue
                    for ood_val_fold in data_dataset["shift"].unique():
                        data_copy = data_dataset.copy()
                        if ood_val_fold in ["train", "ind_val", "ind_test"]:
                            continue
                        data_train = data_copy[
                            ((data_copy["shift"] == ood_val_fold) | (data_copy["shift"] == "ind_val"))]
                        data_train = data_train[data_train["bias"]=="RandomSampler"]
                        if data_train.empty:
                            continue
                        # dsd = DebiasedOODDetector(data_train, ood_val_fold, k=5, batch_size=32)
                        dsd = OODDetector(data_train, ood_val_fold, threshold_method="ind_span")
                        # dsd.plot_hist() # for debugging
                        data_copy["Verdict"] =  data_copy.apply(lambda row: dsd.predict(row), axis=1)
                        data_copy["ood_val_fold"] = ood_val_fold
                        # data_copy["ood"] = ~data_copy["correct_prediction"]  #
                        # data_copy["ood"] = ~data_copy["correct_prediction"]  # OOD is the opposite of correct prediction

                        data_copy["correct_ood_detection"] = data_copy["Verdict"] == data_copy["ood"]
                        data_copy = data_copy[data_copy["ood_val_fold"] != data_copy["shift"]]
                        dfs.append(data_copy)
                    pbar.update(1)
    data = pd.concat(dfs)
    tprs = data[~data["shift"].isin(["ind_val", "ind_test"])].groupby(["Dataset", "feature_name", "shift", "bias", "k"])[
        "correct_ood_detection"].mean().reset_index().groupby(["Dataset", "feature_name", "bias", "k"])[
        "correct_ood_detection"].mean().reset_index()

    tnrs = data[data["shift"].isin(["ind_val", "ind_test"])].groupby(["Dataset", "feature_name", "shift", "bias", "k"])[
        "correct_ood_detection"].mean().reset_index().groupby(["Dataset", "feature_name", "bias", "k"])[
        "correct_ood_detection"].mean().reset_index()

    tprs = tprs.rename(columns={"correct_ood_detection": "TPR"})
    tnrs = tnrs.rename(columns={"correct_ood_detection": "TNR"})
    balanced = pd.merge(tprs, tnrs,
                        on=["Dataset", "feature_name", "bias", "k"])

    # Compute balanced accuracy
    balanced["balanced_accuracy"] = (balanced["TPR"] + balanced["TNR"]) / 2
    balanced.replace(DSD_PRINT_LUT, inplace=True)
    table = balanced.groupby(["Dataset", "feature_name", "k", "bias"]).mean().reset_index()
    print(table)
    # Extract the reference values for k == -1
    ref = (
        table[table["k"] == -1]
        .set_index(["Dataset", "feature_name", "bias"])["balanced_accuracy"]
        .rename("ref_balanced_accuracy")
    )

    # Join reference values back (broadcasting over other k's)
    table = table.set_index(["Dataset", "feature_name", "bias"]).join(ref, on=["Dataset", "feature_name", "bias"])

    # Reset index
    table = table.reset_index()

    # Now compute difference (this ensures difference is zero when k == -1)
    table["balanced_accuracy_diff"] = table["balanced_accuracy"] - table["ref_balanced_accuracy"]
    # table = table[table["k"]!=-1]
    print(table)

    g = sns.FacetGrid(table, col="bias")
    g.map_dataframe(sns.boxplot, x="Dataset", y="balanced_accuracy",  hue="k", palette=sns.color_palette())
    # for ax in g.axes.flat:
    #     ax.axhline(0)

    plt.legend()
    plt.show()
    print(balanced.groupby(["Dataset", "feature_name",  "k", "bias",]).mean())




def get_gam_data(load=True):
    for sample_size in BATCH_SIZES:
        print("loading")
        total_df = load_all(batch_size=sample_size, shift="", samples=100)
        total_df = total_df[total_df["fold"]!="train"] #exclude training data, to not skew results
        for feature_name in total_df["feature_name"].unique():

            for dataset in total_df["Dataset"].unique():
                data = []
                pred_data = []
                df = total_df[(total_df["Dataset"]==dataset) & (total_df["feature_name"]==feature_name)]
                assert len(df["Dataset"].unique())==1, "Dataset should be unique in the filtered dataframe"
                assert len(df["feature_name"].unique())==1, "Feature name should be unique in the filtered dataframe"
                for shift in df["shift"].unique():
                    if shift in ["ind_val", "ind_test", "train"]:
                        continue
                    train = df[(df["feature_name"]==feature_name) & (df["Dataset"]==dataset) & ((df["shift"]==shift)|(df["shift"]=="ind_val")) ]
                    test = df[(df["feature_name"]==feature_name) & (df["Dataset"]==dataset) & ((df["shift"]==shift)|(df["shift"]!="ind_val"))]
                    if train.empty or test.empty:
                        print("Skipping due to empty train or test set: ")
                        print("Dataset:", dataset, "Feature:", feature_name, "Shift:", shift)

                        continue
                    X_train = train['feature']  # Ensure this is adjusted to your dataset
                    y_train = train['loss']
                    X_test = test['feature']
                    y_test = test['loss']#-train["loss"].mean()
                    # Fit the GAM
                    # combined_X = np.concatenate((X_train, X_test))

                    print("Feature:", feature_name, "Shift:", shift, "Dataset:", dataset)
                    print("\tX:", len(X_train.unique()))
                    print("\tY:", len(y_train.unique()))
                    assert len(X_train.unique()>1), "Unique values in training set must be greater than 1"
                    spr = spearmanr(X_train, y_train)[0]
                    print("\t", spr)

                    if spr < 0:
                        gam_monotonic = pygam.LinearGAM(fit_intercept=False, constraints="monotonic_dec")
                        gam_monotonic.fit(X_train, y_train)
                    else:
                        gam_monotonic = pygam.LinearGAM(fit_intercept=False, constraints="monotonic_inc")
                        gam_monotonic.fit(X_train, y_train)

                    XX = gam_monotonic.generate_X_grid(term=0)
                    monotonic_grid_preds = gam_monotonic.predict(XX)
                    monotonic_grid_preds_conf = gam_monotonic.prediction_intervals(XX, width=0.95)

                    for i, (x, ym, ym_c) in enumerate(zip(XX, monotonic_grid_preds, monotonic_grid_preds_conf)):
                        pred_data.append({"feature":x[0],
                                            "monotonic_pred_loss":ym, "monotonic_pred_loss_lower":ym_c[0], "monotonic_pred_loss_upper":ym_c[1],

                                          "Dataset":dataset, "train_shift":shift, "feature_name":feature_name, "sample_size":sample_size})

                    if spr>0.1:
                        plt.plot(XX[:, 0], gam_monotonic.predict(X=XX))
                        plt.scatter(X_train, y_train, c='b', alpha=0.1)
                        plt.scatter(X_test, y_test, c='r', alpha=0.1)
                        plt.show()
                    for test_shift in test["shift"].unique():
                        test_data = test[test["shift"] == test_shift]
                        for sev in test_data["shift_intensity"].unique():
                            test_data_fold = test_data[test_data["shift_intensity"] == sev]

                            preds_monotonic = gam_monotonic.predict(test_data_fold["feature"])
                            mape_monotonic = np.mean(np.abs(preds_monotonic - test_data_fold["loss"]) / np.abs(
                                test_data_fold["loss"]))
                            mae_monotonic = np.mean(np.abs(preds_monotonic - test_data_fold["loss"]))

                            data.append({"Dataset":dataset, "feature_name":feature_name, "train_shift":shift, "test_shift":test_shift, "shift_intensity":sev, "sample_size":sample_size, "monotonic mape": mape_monotonic, "monotonic mae": mae_monotonic})

                    # preds = gam.predict(X_test)
                    # smape = np.mean(np.abs(preds - y_test) / (np.abs(preds) + np.abs(y_test)))
                    # print(f"Dataset: {dataset} Feature: {feature_name} Shift: {shift}: {smape*100}")
                pred_errors = pd.DataFrame(data)
                pred_df = pd.DataFrame(pred_data)
                pred_df.to_csv(f"gam_data/gam_fits_{dataset}_{feature_name}_{sample_size}.csv")
                pred_errors.to_csv(f"gam_data/gam_prediction_errors_{dataset}_{feature_name}_{sample_size}.csv")


def regplots(sample_size):
    def bin_Y(group, bins):
        group['feature_bin'] = pd.qcut(group['feature'], bins, labels=False, duplicates='drop')
        return group

    df = load_all(batch_size=sample_size, prefix="final_data", shift="", samples=40)
    df = df[df["fold"]!="train"] #exclude training data, to not skew results
    df["ind"]=df["fold"]=="ind"
    df = df[df["shift"]!="smear"]
    for shift in df["shift"].unique():
        if shift not in ["hue", "saltpepper", "noise", "multnoise", "smear", "contrast", "brightness", "ind_val", "ind_test", "train"]:
            print(shift)
            df.replace({shift: "Organic Shift"}, inplace=True)


    df.replace({"normal":"Organic Shift"}, inplace=True)
    hues = df["shift"].unique()
    def plot_threshold(data,color=None, **kwargs):
        threshold = OODDetector(data, ood_val_shift="Organic Shift", threshold_method="val_optimal").threshold
        plt.axvline(threshold, color=color, linestyle="--", label="Threshold")

    g = sns.FacetGrid(df, row="Dataset", col="feature_name", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x="feature", y="loss", hue="shift", hue_order=hues,  alpha=0.5)
    g.map_dataframe(plot_threshold)
    g.add_legend()
    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.set_xscale("log")
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


def regplot_by_shift(sample_size, simulate=False):

    df = load_dfs(sample_size=sample_size, simulate=simulate)
    df["ind"]=df["fold"]=="ind"
    df["Shift Severity"]=df["fold"].apply(lambda x: round(float(x.split("_")[1]),2) if "_" in x else x)
    df.rename(columns={"feature_name":"Feature"}, inplace=True)
    df.replace({"typicality":"Typicality", "cross_entropy":"Cross Entropy", "knn":"KNN", "odin":"ODIN", "grad_magnitude":"GradNorm", "energy":"Energy", "softmax":"Softmax"}, inplace=True)
    hues = df["Shift Severity"].unique()
    if simulate:
        df = df[df["KS"]==False]
    df.replace({"normal":"Organic Shift"}, inplace=True)
    g = sns.FacetGrid(df, row="Shift", col="Feature", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x="feature", y="loss", hue="Shift Severity", hue_order=hues)
    g.add_legend()
    plt.show()


def plot_variances(df):
    sampled = load_dfs(10, simulate=True, samples=30)
    sampled = sampled[(sampled["KS"] == False) & (sampled["Dataset"] == "NICO")]
    df["Shift Severity"] = df["fold"].apply(
        lambda x: round(float(x.split("_")[1]), 2) if "_" in x else 0 if "ind" in x else 0.35)
    sampled["Shift Severity"] = sampled["fold"].apply(
        lambda x: round(float(x.split("_")[1]), 2) if "_" in x else 0 if "ind" in x else 0.35)

    data_feat = df.groupby(["Dataset", "feature_name", "Shift", "Shift Severity"])["feature"].std().reset_index()
    data_loss = df.groupby(["Dataset", "feature_name", "Shift", "Shift Severity"])["loss"].std().reset_index()
    data_mean_feat = df.groupby(["Dataset", "feature_name", "Shift", "Shift Severity"])["feature"].mean().reset_index()
    data_mean_loss = df.groupby(["Dataset", "feature_name", "Shift", "Shift Severity"])["loss"].mean().reset_index()
    data_feat.rename(columns={"feature": "Feature Variance"}, inplace=True)
    data_loss.rename(columns={"loss": "Loss Variance"}, inplace=True)
    data = pd.merge(data_feat, data_loss, on=["Dataset", "feature_name", "Shift", "Shift Severity"])
    data = pd.merge(data, data_mean_feat, on=["Dataset", "feature_name", "Shift", "Shift Severity"])
    data = pd.merge(data, data_mean_loss, on=["Dataset", "feature_name", "Shift", "Shift Severity"])
    name_map = {"grad_magnitude": "GradNorm", "cross_entropy": "Cross Entropy", "knn": "KNN", "softmax":"Softmax", "typicality":"Typicality", "energy":"Energy"}
    shift_map = {"normal": "Organic Shift", "noise": "Additive Noise", "multnoise":"Multiplicative Noise", "hue":"Hue Shift", "saltpepper": "Salt & Pepper Noise"}
    g = sns.FacetGrid(data, row="Dataset", col="feature_name", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.lineplot, x="Shift Severity", y="Feature Variance", hue="Shift", alpha=0.5)
    g.add_legend()
    plt.show()

    g = sns.FacetGrid(data, row="Dataset", col="feature_name", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.lineplot, x="Shift Severity", y="Loss Variance", hue="Shift", alpha=0.5)
    g.add_legend()
    plt.show()
    data = data[data["Dataset"] == "NICO"]

    fig, ax = plt.subplots(len(data["Shift"].unique()), len(data["feature_name"].unique()), figsize=(20, 20))
    color_map = dict(
        zip(sorted(data["Shift Severity"].unique()), sns.color_palette("magma", len(data["Shift Severity"].unique()))))

    # Add a color column to the sampled DataFrame
    sampled["color"] = sampled["Shift Severity"].map(color_map)

    for i, sev in enumerate(data["Shift"].unique()):
        for j, feature in enumerate(data["feature_name"].unique()):
            subdf = data[(data["Shift"] == sev) & (data["feature_name"] == feature)]
            sampled_subdf = sampled[(sampled["Shift"] == sev) & (sampled["feature_name"] == feature)]

            # Plot scatter points for the sampled data with a black outline
            ax[i, j].scatter(sampled_subdf['feature'], sampled_subdf['loss'],
                             color=sampled_subdf['color'], alpha=0.5,
                             edgecolors='black', linewidth=1.5)

            for row_n, (_, row) in enumerate(subdf.iterrows()):
                color = color_map[row['Shift Severity']]
                ellipse = patches.Ellipse((row['feature'], row['loss']), row['Feature Variance'], row['Loss Variance'],
                                          color=color, alpha=0.3)
                ax[i, j].add_patch(ellipse)

            ax[i, j].set_title(f"{shift_map[sev]}|{name_map[feature]}")
    plt.tight_layout()
    plt.savefig("variance_plot.pdf")
    plt.show()


def compare_gam_errors():
    df = get_gam_data()
    df = df[df["KS"]==False]
    print(df.groupby(["Dataset","train_shift","sample_size", "feature_name"])[[ "regular mape", "monotonic mape"]].mean())
    plot_df = df[df["train_shift"]=="noise"]
    plot_df = plot_df[plot_df["KS"]==False]
    g = sns.FacetGrid(plot_df, col="Dataset", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.lineplot, x="sample_size", y="monotonic mape", hue="feature_name", palette="pastel")
    for ax in g.axes.flat:
        ax.set_ylim(0, 1)
        ax.set_xscale("log")
    g.add_legend( title="Feature")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #accuracies on each dataset
    # accuracy_table()
    #class distribution
    # dataset_summaries()
    # loss_correctness_test()
    # for batch in BATCH_SIZES:
    #     ood_detector_correctness_prediction_accuracy(batch)
    #     ood_verdict_accuracy_table(batch)



    # for batch_size in BATCH_SIZES:
    #     ood_detector_correctness_prediction_accuracy(batch_size)


    #runtime verification
    bias_correctness_test()
    # eval_debiased_ood_detectors(16)

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