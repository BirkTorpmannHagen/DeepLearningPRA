import itertools
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from components import OODDetector
from rateestimators import BernoulliEstimator
from riskmodel import UNNECESSARY_INTERVENTION
from simulations import UniformBatchSimulator
from utils import DATASETS, DSDS, DSD_PRINT_LUT, load_pra_df, BATCH_SIZES, DSD_LUT
pd.set_option("display.precision", 3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
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
                            data.append({"rate": rate, "tpr":tpr, "tnr":tnr, "ba": ba,  "tl": tl, "rate_estimate": rate_estimate, "error":error})

    df = pd.DataFrame(data)
    df = df.groupby(["tpr", "tnr", "ba","tl", "rate"]).mean()
    df.to_csv("rate_estimator_sensitivity_analysis.csv")


def eval_rate_estimator():
    df = pd.read_csv("rate_estimator_sensitivity_analysis.csv")
    df_barate = df.groupby(["ba", "rate"]).mean().reset_index()
    pivot_table = df_barate.pivot(index="ba", columns="rate", values="error")
    pivot_table = pivot_table.loc[::-1]
    sns.heatmap(pivot_table, vmin=0, vmax=0.5)
    plt.savefig("rate_sensitivity.eps")
    plt.tight_layout()
    plt.show()



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


def show_rate_risk():
    data = load_pra_df("Polyp", "knn", batch_size=16, samples=100)
    oods = data[~data["shift"].isin(["ind_val", "ind_test", "train"])]["shift"].unique()
    rates = np.linspace(0, 1, 11)
    dfs = []
    with tqdm(total=len(oods)*(len(oods)-1)*len(rates)) as pbar:
        for ood_val_set, ood_test_set, rate in itertools.product(oods, oods, rates):
            if ood_val_set == ood_test_set:
                continue
            sim = UniformBatchSimulator(data, ood_test_shift=ood_test_set, ood_val_shift=ood_val_set, maximum_loss=0.5, estimator=BernoulliEstimator, use_synth=True, dsd_tpr=0.9, dsd_tnr=0.9)
            results = sim.sim(rate, 600)
            results["Rate"] = rate
            results["Risk Error"]=results["Risk Estimate"]-results["True Risk"]
            dfs.append(results)
            pbar.update()

    df = pd.concat(dfs)
    df = df.groupby(["Tree", "Rate"]).mean().reset_index()
    df.replace({"Base Tree": "Estimated Risk w/o RV", "Detector Tree": "Estimated Risk w/RV"}, inplace=True)
    sns.lineplot(df, x="Rate", y="Risk Estimate", hue="Tree")
    df.replace({"Estimated Risk w/o RV":"True Risk w/o RV", "Estimated Risk w/RV":"True Risk w/ RV"}, inplace=True)

    sns.lineplot(df, x="Rate", y="True Risk", hue="Tree", linestyle="--")
        # plt.plot(df_tree["Rate"], df_tree["True Risk"], label=f"{tree} True Risk", linestyle="dashed")
    plt.axhline(UNNECESSARY_INTERVENTION, color="red", label="Manual Intervention")
    plt.xlabel("p(E)")
    plt.legend()
    plt.savefig("rate_risk.pdf")
    plt.show()


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
    from experiments.runtime_classification import ood_detector_correctness_prediction_accuracy
    bins=11
    for batch_size in BATCH_SIZES:
        for dataset in DATASETS:
            try:
                dsd_accuracies = pd.read_csv(f"ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv")
            except FileNotFoundError:
                dsd_accuracies = ood_detector_correctness_prediction_accuracy(batch_size)
            dsd_accuracies = dsd_accuracies[
                (
                    (~dsd_accuracies["OoD==f(x)=y"])&
                    (dsd_accuracies["Threshold Method"]=="val_optimal")&
                    (~dsd_accuracies["Performance Calibrated"])
                )
            ]
            dsd_accuracies = dsd_accuracies.groupby(["Dataset", "feature_name"])[["tpr", "tnr", "ba"]].mean().reset_index()
            best_dsds = dsd_accuracies.loc[dsd_accuracies.groupby("Dataset")["ba"].idxmax()].reset_index(drop=True)
            print(best_dsds)
            dfs = []
            dsd = best_dsds[(best_dsds["Dataset"]==dataset)]["feature_name"].values[0]
            filt = dsd_accuracies[(dsd_accuracies["Dataset"]==dataset)&(dsd_accuracies["feature_name"]==dsd)]
            tpr, tnr, ba = filt["tpr"].mean(), filt["tnr"].mean(), filt["ba"].mean()
            print(tpr, tnr, ba)
            data = load_pra_df(dataset, DSD_LUT[dsd], batch_size=batch_size, samples=1000)
            if data.empty:
                continue
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
            df_final.to_csv(f"pra_data_final/dsd_results_{dataset}_{batch_size}.csv")


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
                if feature=="rabanser":
                    continue
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


def plot_tpr_tnr_sensitivity():
    # Load and preprocess data
    dfs = []
    for filename in os.listdir("pra_data"):
        df = pd.read_csv(f"pra_data/{filename}")
        dfs.append(df)
    df = pd.concat(dfs)

    print(df.head(10))
    df = df[df["ba"] >= 0.5]
    print(df.groupby(["ba", "Tree", "val_set", "test_set"])[["Rate Error", "Accuracy Error"]].mean())
    # input()
    df["ba"] = round(df["ba"], 2)
    df["rate"] = round(df["rate"], 2)
    # Prepare data for heatmaps
    facet = df.groupby(["ba", "rate", "val_set", "test_set"])[["Accuracy Error"]].mean().reset_index()
    # facet = facet.pivot(index=["val_set", "test_set"], columns="rate", values="Error")

    # Define heatmap function
    def draw_heatmap(data, **kws):
        # Extract numeric data for the heatmap
        heatmap_data = data.pivot(index="ba", columns="rate", values="Accuracy Error")
        heatmap_data = heatmap_data.loc[::-1]

        sns.heatmap(heatmap_data, **kws, cmap="mako", vmin=0, vmax=(df["ind_acc"]-df["ood_val_acc"]).mean())
    # Create FacetGrid and plot heatmaps
    g = sns.FacetGrid(facet.reset_index(), col="test_set", row="val_set", col_order=[CVCCLINIC, ETISLARIB, ENDOCV], row_order=[CVCCLINIC, ETISLARIB, ENDOCV], margin_titles=True)
    g.map_dataframe(draw_heatmap)
    plt.savefig("cross_validated_accuracy_estimation_error.eps")
    plt.show()
    # Additional analysis and plotting
    print(df[df["ba"] == 1].groupby(["ba", "rate"])[["E[f(x)=y]", "Accuracy Error"]].mean().reset_index())
    df = df.groupby(["ba", "rate"])["Accuracy Error"].mean().reset_index()
    pivot_table = df.pivot(index="ba", columns="rate", values="Accuracy Error")
    pivot_table = pivot_table.loc[::-1]
    sns.heatmap(pivot_table, cmap="mako")
    plt.legend()
    plt.savefig("tpr_tnr_sensitivity.eps")
    plt.show()


def plot_dsd_acc_errors():
    dfs = []
    for dataset in DATASETS:
        for batch_size in BATCH_SIZES:
            try:
                df = pd.read_csv(f"pra_data_final/dsd_results_{dataset}_{batch_size}.csv")
                # best_guess = (df["ind_acc"].mean() + df["ood_val_acc"].mean()) / 2
                best_guess = df["ind_acc"].mean()
                df["Dataset"]=dataset
                df["batch_size"]=batch_size
                df["lineplot_idx"]=BATCH_SIZES.index(batch_size)
                df["lineplot_rate_idx"] = pd.factorize(df['rate'])[0]
                print(dataset, " : ", best_guess)
                df["best_guess_error"] = np.abs(df["Accuracy"] - best_guess)
                dfs.append(df)
            except:
                print(f"No data found for {dataset} with batch size {batch_size}")
    df = pd.concat(dfs)
    df.replace(DSD_PRINT_LUT, inplace=True)
    print(df.head(10))
    df = df[df["Tree"]=="Detector Tree"]
    # df = df[df["batch_size"]==1]
    g = sns.FacetGrid(df[df["batch_size"]==1], col="Dataset", sharey=False, col_wrap=3)
    g.map_dataframe(sns.boxplot, x="rate", y="Accuracy Error", hue="test_set", showfliers=False, palette=sns.color_palette())
    g.map_dataframe(sns.lineplot, x="lineplot_rate_idx", y="best_guess_error", hue="test_set", linestyle="--", marker="o", palette=sns.color_palette(), legend=False)
    sorted_datasets = sorted(df["Dataset"].unique())

    for ax, dataset in zip(g.axes.flat, sorted_datasets):
        ax.set_title(dataset)
        ax.set_xlabel("P(E)")
        ax.set_ylabel("Accuracy Error")
        ax.set_xticklabels(df["rate"].unique())
        ax.set_xticks(range(len(df["rate"].unique())))
        ax.legend(title="Test Set", ncols=3, fontsize=8)
        ax.set_yscale("log")

    num_plots = len(g.axes.flat)
    num_cols = 3  # Top row columns
    last_row_plots = num_plots % num_cols

    if last_row_plots > 0:
        # Get figure width
        fig_width = g.fig.get_size_inches()[0]

        # Compute total space occupied by the last row's plots
        last_row_width = (fig_width / num_cols) * last_row_plots

        # Compute left padding to center the row
        left_padding = (fig_width - last_row_width) / 2

        # Adjust position of the last row's plots
        for ax in g.axes[-last_row_plots:]:
            pos = ax.get_position()
            ax.set_position([pos.x0 + left_padding / fig_width, pos.y0, pos.width, pos.height])
    plt.savefig("dsd_acc_erorrs_by_rate.pdf")
    plt.show()


    g = sns.FacetGrid(df, col="Dataset", height=3, aspect=1.5, col_wrap=3, sharey=False)

    df = df[df["val_set"]!=df["test_set"]]
    g.map_dataframe(sns.boxplot, x="batch_size", y="Accuracy Error", hue="test_set", showfliers=False, palette=sns.color_palette())
    g.map_dataframe(sns.lineplot, x="lineplot_idx", y="best_guess_error", hue="test_set", linestyle="--", marker="o", palette=sns.color_palette(), legend=False)
    # g.map_dataframe(sns.lineplot, x="batch_size", y="Accuracy Error", hue="test_set")
    for ax, dataset in zip(g.axes.flat, sorted_datasets):
        ax.set_title(dataset)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Accuracy Error")
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xticks(range(len(BATCH_SIZES)))
        ax.legend(title="Test Set", ncols=3, fontsize=8)
        ax.set_yscale("log")

    num_plots = len(g.axes.flat)
    num_cols = 3  # Top row columns
    last_row_plots = num_plots % num_cols

    if last_row_plots > 0:
        # Get figure width
        fig_width = g.fig.get_size_inches()[0]

        # Compute total space occupied by the last row's plots
        last_row_width = (fig_width / num_cols) * last_row_plots

        # Compute left padding to center the row
        left_padding = (fig_width - last_row_width) / 2

        # Adjust position of the last row's plots
        for ax in g.axes[-last_row_plots:]:
            pos = ax.get_position()
            ax.set_position([pos.x0 + left_padding / fig_width, pos.y0, pos.width, pos.height])
    plt.savefig("dsd_acc_errors.pdf")
    plt.show()


def plot_sensitivity_errors():
    dfs = []
    for dataset in DATASETS:
        try:
            df = pd.read_csv(f"pra_data/{dataset}_sensitivity_results.csv")
            best_guess = (df["ind_acc"].mean() + df["ood_val_acc"].mean()) / 2
            print(dataset, " : ", best_guess)
            df["Dataset"]=dataset
            df["best_guess_error"] = np.abs(df["Accuracy"] - best_guess)
            dfs.append(df)
        except:
            print(f"No data found for {dataset}")
    df = pd.concat(dfs)
    df = df[df["Tree"]=="Base Tree"]
    df = df[df["val_set"]!=df["test_set"]]
    df["rate"]=round(df["rate"], 2)
    df["ba"]=round(df["ba"], 2)
    df.replace(DSD_PRINT_LUT, inplace=True)
    print(df.columns)
    df = df.groupby(["Dataset", "rate", "ba"])[["Accuracy Error", "ind_acc", "ood_val_acc"]].mean().reset_index()
    df.rename(columns={"rate":"$P(E)$", "ba":"$p(D_{e}(x)=E)$"}, inplace=True)
    g = sns.FacetGrid(df, col="Dataset", col_wrap=3)
    #sort by ba increasing order
    def plot_heatmap(data, **kws):
        heatmap_data = data.pivot(index="$p(D_{e}(x)=E)$", columns="$P(E)$", values="Accuracy Error")
        heatmap_data = heatmap_data.loc[::-1] #higher ba is up
        sns.heatmap(heatmap_data, **kws, cmap="mako", vmin=0, vmax=(df["ind_acc"]-df["ood_val_acc"]).mean())
    g.map_dataframe(plot_heatmap)


    num_plots = len(g.axes.flat)
    num_cols = 3  # Top row columns
    last_row_plots = num_plots % num_cols

    if last_row_plots > 0:
        fig_width = g.fig.get_size_inches()[0]
        last_row_width = (fig_width / num_cols) * last_row_plots
        left_padding = (fig_width - last_row_width) / 2

        for ax in g.axes[-last_row_plots:]:
            pos = ax.get_position()
            ax.set_position([pos.x0 + left_padding / fig_width, pos.y0, pos.width, pos.height])
            cbar = ax.collections[0].colorbar
            cbar.ax.set_position([cbar.ax.get_position().x0 + left_padding / fig_width, cbar.ax.get_position().y0, cbar.ax.get_position().width, cbar.ax.get_position().height])
    plt.savefig("sensitivity_errors.pdf")
    plt.show()


def get_risk_tables():
    df = pd.read_csv("datasetwise_risk.csv")
    df.replace(DSD_PRINT_LUT, inplace=True)
    df.replace({"ind_test": "Kvasir"}, inplace=True)
    df_base = df[df["Tree"]=="Base Tree"]
    df_dsd = df[df["Tree"]!="Base Tree"]

    print(df_base.groupby(["Model", "Dataset"])["True Risk"].mean())
    print(df_dsd.groupby(["Model", "DSD", "Dataset"])["True Risk"].mean())
