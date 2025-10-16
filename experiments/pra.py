import itertools
import os.path
from multiprocessing import Pool

import seaborn as sns
from matplotlib import pyplot as plt

from experiments.runtime_classification import get_all_ood_detector_data, ood_verdict_shiftwise_accuracy_tables
from rateestimators import ErrorAdjustmentEstimator, SimpleEstimator
from riskmodel import UNNECESSARY_INTERVENTION
from simulations import UniformBatchSimulator
from utils import *

pd.set_option("display.precision", 3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering


def simulate_dsd_accuracy_estimation(data, rate, val_set, test_set, feature_name, calibrated_by_fold):
    sim = UniformBatchSimulator(data, ood_test_shift=test_set, ood_val_shift=val_set,
                                estimator=ErrorAdjustmentEstimator,
                                calibrated_by_fold=calibrated_by_fold, use_synth=False)
    results = sim.sim(rate, 600)
    results = results.groupby(["Tree"]).mean().reset_index()
    # results = results.mean()
    results["dsd"] = feature_name
    results["rate"] = rate
    results["Calibrated By Fold"] = calibrated_by_fold
    results["test_set"] = test_set
    results["val_set"] = val_set
    return results


def xval_errors(values):
    return np.mean([np.sum(np.abs(np.subtract.outer(valwise_accuracies, valwise_accuracies))) / np.sum(
        np.ones_like(valwise_accuracies) - np.eye(valwise_accuracies.shape[0])) for valwise_accuracies in values])


def get_dsd_verdicts_given_true_trace(trace, tpr, tnr):
    def transform(v):
        if v == 1:
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
    with tqdm(total=26 * 26 * 26 * 9) as pbar:
        for rate in tqdm(np.linspace(0, 1, 26)):
            for tpr in np.linspace(0, 1, 26):
                for tnr in np.linspace(0, 1, 26):
                    for tl in [10, 20, 30, 50, 60, 100, 200, 500, 1000]:
                        ba = round((tpr + tnr) / 2, 2)
                        if ba <= 0.5:
                            continue
                        pbar.update(1)
                        re = ErrorAdjustmentEstimator( tpr=tpr, tnr=tnr)
                        sample = re.sample(10_000, rate)
                        dsd = get_dsd_verdicts_given_true_trace(sample, tpr, tnr)
                        for i in np.array_split(dsd, int(10_000 // tl)):
                            re.update(i)
                            rate_estimate = re.get_rate()
                            error = np.abs(rate - rate_estimate)
                            data.append({"rate": rate, "tpr": tpr, "tnr": tnr, "ba": ba, "tl": tl,
                                         "rate_estimate": rate_estimate, "error": error})

    df = pd.DataFrame(data)
    df = df.groupby(["tpr", "tnr", "ba", "tl", "rate"]).mean()
    df.to_csv("rate_estimator_sensitivity_analysis.csv")


def eval_rate_estimator():
    df = pd.read_csv("rate_estimator_sensitivity_analysis.csv")
    df_barate = df.groupby(["ba", "rate"]).mean().reset_index()
    pivot_table = df_barate.pivot(index="ba", columns="rate", values="error")
    pivot_table = pivot_table.loc[::-1]
    sns.heatmap(pivot_table, vmin=0, vmax=0.5)
    plt.savefig("rate_sensitivity.pdf")
    plt.tight_layout()
    plt.show()


def plot_rate_estimation_errors_for_dsds():
    data = []

    with tqdm(total=(len(DATASETS) * len(DSDS) * 11 * len(BATCH_SIZES))) as pbar:
        for batch_size in BATCH_SIZES:
            dsd_data = get_all_ood_detector_data(batch_size=batch_size, filter_thresholding_method=True,
                                                 filter_ood_correctness=True, filter_correctness_calibration=True,
                                                 filter_best=True)
            dsd_data = dsd_data[~dsd_data["OoD Val Fold"].isin(SYNTHETIC_SHIFTS)]
            for dataset in DATASETS:
                for feature in DSDS:
                    if feature == "rabanser":
                        continue
                    subdata_dataset = dsd_data[
                        (dsd_data["Dataset"] == dataset) & (dsd_data["feature_name"] == DSD_PRINT_LUT[feature])]
                    for rate in np.linspace(0, 1, 11):
                        for shift in subdata_dataset["OoD Test Fold"].unique():
                            if shift in SYNTHETIC_SHIFTS:
                                continue
                            subdata = subdata_dataset[subdata_dataset["OoD Test Fold"] == shift]
                            tpr, tnr, ba = subdata["tpr"].mean(), subdata["tnr"].mean(), subdata["ba"].mean()
                            for tl in [10, 20, 30, 50, 60, 100, 200, 500, 1000]:
                                if ba<=0.5:
                                    re = SimpleEstimator(prior_rate=0.5)
                                else:
                                    re = ErrorAdjustmentEstimator(tpr=tpr, tnr=tnr)
                                sample = re.sample(10_000, rate)
                                dsd = get_dsd_verdicts_given_true_trace(sample, tpr, tnr)
                                simple_re = SimpleEstimator(prior_rate=rate)

                                for i in np.array_split(dsd, int(10_000 // tl)):
                                    # i is a trace of length tl
                                    re.update(i)
                                    simple_re.update(i)
                                    rate_estimate = re.get_rate()
                                    simple_rate_estimate = simple_re.get_rate()

                                    error = np.abs(rate - rate_estimate)
                                    simple_error = np.abs(rate - simple_rate_estimate)

                                    data.append(
                                        {"Dataset": dataset, "feature_name": feature, "batch_size": batch_size,
                                         "Estimator": "Error Adjustment", "Shift": shift, "rate": rate, "ba": ba,
                                         "tl": tl, "rate_estimate": rate_estimate, "error": error})

                                    data.append(
                                        {"Dataset": dataset, "feature_name": feature, "batch_size": batch_size,
                                         "Estimator": "Simple", "Shift": shift, "rate": rate, "ba": ba, "tl": tl,
                                         "rate_estimate": simple_rate_estimate, "error": simple_error})
                        pbar.update(1)
    df = pd.DataFrame(data)
    df.replace(DSD_PRINT_LUT, inplace=True)

    df = df.groupby(["Dataset", "feature_name", "tl", "rate", "batch_size", "Estimator"])[
        ["rate_estimate", "error"]].mean().reset_index()
    # df = df[df["tl"]==100]
    g = sns.FacetGrid(df, col="Dataset", sharey=False, col_wrap=3)
    g.map_dataframe(sns.lineplot, x="batch_size", y="error", hue="Estimator", palette=sns.color_palette())
    g.set_axis_labels("Batch Size", "Error")
    g.add_legend(bbox_to_anchor=(0.7, 0.5), loc="upper center")
    plt.savefig("rate_estimation_errors_per_estimator.pdf")
    plt.show()

    df["baseline"] = np.abs(df["rate"] - 0.5)
    df["Error Compared to Baseline"] = df["error"] - df["baseline"]
    df = df[df["Estimator"] == "Error Adjustment"]
    g = sns.FacetGrid(df, col="Dataset", col_wrap=3)
    g.map_dataframe(sns.lineplot, x="rate", y="Error Compared to Baseline", hue="batch_size",
                    palette=sns.color_palette())

    g.tight_layout()
    plt.legend(frameon=True, ncol=1, loc="upper center", bbox_to_anchor=(1.5, 0.5))
    # plt.tight_layout(w_pad=0.5)Note that these error estimates are computed based
    for ax in g.axes.flat:
        ax.axhline(0, color="red", linestyle="--", label="Baseline")
    plt.savefig("dsd_rate_error.pdf")
    plt.show()
    df.to_csv("rate_estimator_eval.csv")


def plot_rate_estimates():
    df = pd.read_csv("rate_estimator_eval.csv")
    df["ba"] = np.round((df["tpr"] + df["tnr"]) / 2, 2)
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df["error"] = np.abs(df["rate"] - df["rate_estimate"])

    df = df[df["tl"] == 10]
    df = df[df["ba"] > 0.5]
    df = df.groupby(["ba", "rate"]).mean().reset_index()
    print(df)
    pivot_table = df.pivot(index="ba", columns="rate", values="error")
    pivot_table = pivot_table.loc[::-1]
    sns.heatmap(pivot_table)

    # plt.xticks([0, df["ba"].nunique()], [0, 1], rotation=0)  # x-axis: only 0 and 1
    # plt.yticks([0, df["rate"].nunique()], [1, 0])  # y-axis: only 0 and 1
    plt.savefig("rate_sensitivity.pdf")
    plt.show()


def get_ratewise_risk_data(load=True):
    if load and os.path.exists("pra_data/ratewise_risk_data.csv"):
        df = pd.read_csv("pra_data/ratewise_risk_data.csv")
    else:
        data = load_pra_df("Polyp", "knn", batch_size=1, samples=100)
        oods = data[~data["shift"].isin(["ind_val", "ind_test", "train"])]["shift"].unique()
        rates = np.linspace(0, 1, 11)
        dfs = []
        with tqdm(total=len(oods) * (len(oods) - 1) * len(rates)) as pbar:
            for ood_val_set, ood_test_set, rate in itertools.product(oods, oods, rates):
                if ood_val_set == ood_test_set:
                    continue
                sim = UniformBatchSimulator(data, ood_test_shift=ood_test_set, ood_val_shift=ood_val_set,
                                            maximum_loss=0.5, estimator=ErrorAdjustmentEstimator, use_synth=False)
                results = sim.sim(rate, 600)
                results["Rate"] = rate
                results["Risk Error"] = results["Risk Estimate"] - results["True Risk"]
                dfs.append(results)
                pbar.update()
        df = pd.concat(dfs)
        df.to_csv("pra_data/ratewise_risk_data.csv", index=False)

    df = df.groupby(["Tree", "Rate"]).mean().reset_index()
    df.replace({"Base Tree": "Estimated Risk w/o RV", "Detector Tree": "Estimated Risk w/RV"}, inplace=True)
    sns.lineplot(df, x="Rate", y="Risk Estimate", hue="Tree")
    df.replace({"Estimated Risk w/o RV": "True Risk w/o RV", "Estimated Risk w/RV": "True Risk w/ RV"}, inplace=True)

    sns.lineplot(df, x="Rate", y="True Risk", hue="Tree", linestyle="--")
    # plt.plot(df_tree["Rate"], df_tree["True Risk"], label=f"{tree} True Risk", linestyle="dashed")
    plt.axhline(UNNECESSARY_INTERVENTION, color="red", label="Manual Intervention")
    plt.xlabel("p(E)")
    plt.legend()
    plt.savefig("figures/rate_risk.pdf")
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

        with tqdm(total=bins ** 2 * (len(ood_sets) - 1) * len(ood_sets)) as pbar:
            for val_set in ood_sets:
                for test_set in ood_sets:
                    if test_set == "noise":
                        continue  #used only to estimate accuracies
                    for rate in np.linspace(0, 1, bins):
                        for ba in np.linspace(0.5, 1, bins):
                            sim = UniformBatchSimulator(data, ood_test_shift=test_set, ood_val_shift=val_set,
                                                        estimator=ErrorAdjustmentEstimator, dsd_tpr=ba, dsd_tnr=ba)
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


def collect_re_accuracy_estimation_data():
    bins = 11

    for batch_size in BATCH_SIZES:
        best = get_all_ood_detector_data(batch_size=batch_size, filter_thresholding_method=True,
                                         filter_ood_correctness=True, filter_correctness_calibration=True,
                                         filter_best=True, filter_organic=False)
        for dataset in DATASETS:
            if dataset!="OfficeHome" and batch_size!=1:
                continue
            dsd_accuracies = best[best["Dataset"] == dataset]
            dfs = []
            config = dsd_accuracies.groupby(["feature_name"])[["tpr", "tnr", "ba"]].mean().reset_index()
            feature_name, _, _, _ = config.iloc[0]
            data = load_pra_df(dataset, DSD_LUT[feature_name], batch_size=batch_size, samples=1000, shift="",
                               prefix="fine_data/")
            data = data[
                (data["shift"].isin(SYNTHETIC_SHIFTS) &
                 (data["shift_intensity"] == data.groupby("shift")["shift_intensity"].transform("max")))
                | (~data["shift"].isin(SYNTHETIC_SHIFTS))
                ]
            if data.empty:
                continue
            ood_sets = data[~data["shift"].isin(["ind_val", "ind_test", "train"])]["shift"].unique()

            with tqdm(total=bins * len(ood_sets) * (len(ood_sets) - 1)) as pbar:
                for val_set in ood_sets:
                    for test_set in ood_sets:
                        for calibrated_by_fold in [False, True]:
                            pool = Pool(bins)
                            print("multiprocessing...")
                            results = pool.starmap(simulate_dsd_accuracy_estimation, [
                                (data, rate, val_set, test_set, feature_name, calibrated_by_fold) for rate
                                in np.linspace(0, 1, bins)])
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
    df = df.groupby(["DSD Accuracy", "Bernoulli Expectation"]).mean().reset_index()  # Average over tpr and tnr
    df["Error"] = np.abs(df["Risk Estimate"] - df["True Risk"]) / df["True Risk"]
    sns.lineplot(df, x="DSD Accuracy", y="Error")
    plt.show()
    df = df.sort_values(by=["DSD Accuracy", "Bernoulli Expectation"])

    pivot_table = df.pivot(index="Bernoulli Expectation", columns="DSD Accuracy", values="Error")

    # Reverse the `tpr` axis (y-axis) order

    pivot_table = pivot_table.loc[::-1]
    sns.heatmap(pivot_table)
    plt.xticks([0, df["DSD Accuracy"].nunique()], [0, 1], rotation=0)  # x-axis: only 0 and 1
    plt.yticks([0, df["Bernoulli Expectation"].nunique()], [1, 0])  # y-axis: only 0 and 1
    plt.savefig("ba_sensitivity.pdf")
    plt.show()





def plot_dsd_acc_errors():
    dfs = []
    print("Loading data...")
    for dataset in DATASETS:
        for batch_size in BATCH_SIZES:
            try:
                df = pd.read_csv(f"pra_data_final/dsd_results_{dataset}_{batch_size}.csv")
                # best_guess = (df["ind_acc"].mean() + df["ood_val_acc"].mean()) / 2
                best_guess = df["ind_acc"].mean()
                print(best_guess, )
                df["Dataset"] = dataset
                df["batch_size"] = batch_size
                df["lineplot_idx"] = BATCH_SIZES.index(batch_size)
                df["lineplot_rate_idx"] = pd.factorize(df['rate'])[0]
                df["best_guess_error"] = np.abs(df["Accuracy"] - best_guess)
                dfs.append(df)
            except:
                print(f"No data found for {dataset} with batch size {batch_size}")
    print("Loaded")
    df = pd.concat(dfs)
    print(df.columns)
    df = df[df["Calibrated By Fold"] == True]
    df.replace(DSD_PRINT_LUT, inplace=True)
    df = df[df["Tree"] == "Detector Tree"]
    df["Baseline Error Improvement"] = df["Accuracy Error"] - df["best_guess_error"]
    df["Distributional Shift"] = df["test_set"].apply(lambda x: "Synthetic" if x in SYNTHETIC_SHIFTS else "Organic")
    # df = df[df["batch_size"]==1]
    df["rate"] = df["rate"].round(2)
    g = sns.FacetGrid(df[(df["batch_size"] == 1) & (~df["test_set"].isin(SYNTHETIC_SHIFTS)) & (
        ~df["val_set"].isin(SYNTHETIC_SHIFTS))], col="Dataset", sharex=False, sharey=False, col_wrap=3, height=3, aspect=1,
                      )
    g.map_dataframe(
        sns.pointplot,
        x="rate",
        y="Accuracy Error",
        hue="test_set",
        palette=sns.color_palette(),
        errorbar=("pi", 100),
        join=False,
        dodge=1,
        alpha=0.5,
    )

    g.map_dataframe(sns.lineplot, x="lineplot_rate_idx", y="best_guess_error", hue="test_set", linestyle="--",
                    marker="o", palette=sns.color_palette(), legend=False)
    sorted_datasets = sorted(df["Dataset"].unique())
    for ax in g.axes.flat:
        ax.set_xlabel("P(E)")
        ax.set_ylabel("$$\delta$$ Error")
        # ax.set_xticklabels(df["rate"].unique())
        # ax.set_xticks(range(len(df["rate"].unique())))
        ax.set_yscale("log")
        ax.legend(title="Test Set", ncols=2, fontsize=8, frameon=True)
    g.fig.subplots_adjust(hspace=0.3)  # tweak this value as needed

    plt.savefig("figures/dsd_acc_errors_by_rate_absolute.pdf")
    plt.show()
    input()

    g = sns.FacetGrid(df[(~df["test_set"].isin(SYNTHETIC_SHIFTS)) & (
        ~df["val_set"].isin(SYNTHETIC_SHIFTS))], col="Dataset", row="batch_size", sharex=False, sharey=False, height=3,
                      aspect=1)
    g.map_dataframe(
        sns.pointplot,
        x="rate",
        y="Accuracy Error",
        hue="test_set",
        palette=sns.color_palette(),
        errorbar=("pi", 100),
        join=False,
        dodge=1,
        alpha=0.5,
    )

    g.map_dataframe(sns.lineplot, x="lineplot_rate_idx", y="best_guess_error", hue="test_set", linestyle="--",
                    marker="o", palette=sns.color_palette(), legend=False)
    sorted_datasets = sorted(df["Dataset"].unique())
    for ax in g.axes.flat:
        ax.set_xlabel("P(E)")
        ax.set_ylabel("$$\delta$$ Error")
        # ax.set_xticklabels(df["rate"].unique())
        # ax.set_xticks(range(len(df["rate"].unique())))
        ax.set_yscale("log")
        ax.legend(title="Test Set", ncols=2, fontsize=8, frameon=True)
    plt.savefig("figures/dsd_acc_errors_by_rate_absolute_and_batch_size.pdf")
    plt.show()

    g = sns.FacetGrid(df[df["batch_size"] == 1], col="Dataset", sharex=False, sharey=False, col_wrap=3)
    g.map_dataframe(sns.lineplot, x="rate", y="Baseline Error Improvement", hue="test_set", palette=sns.color_palette())
    # g.map_dataframe(sns.boxplot, x="rate", y="Accuracy Error", hue="Distributional Shift", palette=sns.color_palette())
    # g.map_dataframe(sns.lineplot, x="lineplot_rate_idx", y="best_guess_error", hue="test_set", linestyle="--", marker="o", palette=sns.color_palette(), legend=False)
    sorted_datasets = sorted(df["Dataset"].unique())
    for ax in g.axes.flat:
        ax.set_xlabel("P(E)")
        ax.set_ylabel("$$\delta$$ Error")
        ax.legend(title="Test Set", ncols=2, fontsize=8, frameon=False)
        ax.axhline(0, color="red", linestyle="--", label="Baseline")
        # ax.set_yscale("log")

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
    plt.savefig("figures/dsd_acc_errors_by_rate.pdf")
    plt.show()

    print(df.columns)
    print(df["Distributional Shift"].unique())
    g = sns.FacetGrid(df[df["Distributional Shift"]=="Organic"], col="Dataset", height=3, aspect=1, col_wrap=3, sharey=False)
    g.map_dataframe(sns.boxplot, x="batch_size", y="Accuracy Error", hue="test_set", showfliers=False,
                    palette=sns.color_palette())
    g.map_dataframe(sns.lineplot, x="lineplot_idx", y="best_guess_error", hue="test_set", linestyle="--",
                    marker="o", palette=sns.color_palette(), legend=False)
    for ax, dataset in zip(g.axes.flat, sorted_datasets):
        # ax.set_title(dataset)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Accuracy Error")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1)
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xticks(range(len(BATCH_SIZES)))
        ax.legend(title="Test Set", ncols=3, fontsize=8)
        # ax.set_yscale("log")
    plt.savefig("figures/dsd_acc_errors_by_batch_size.pdf")
    plt.show()

    g = sns.FacetGrid(df[df["batch_size"] == 1], col="Dataset", height=3, aspect=1, col_wrap=3, sharey=False)
    df = df[df["val_set"] != df["test_set"]]
    g.map_dataframe(sns.boxplot, x="rate", y="Accuracy Error", hue="Distributional Shift", hue_order=["Synthetic", "Organic"], showfliers=False,
                    palette=sns.color_palette())
    g.map_dataframe(sns.lineplot, x="lineplot_rate_idx", y="best_guess_error", hue="Distributional Shift", linestyle="--",
                    marker="o", palette=sns.color_palette(), legend=False)
    # g.map_dataframe(sns.lineplot, x="batch_size", y="Accuracy Error", hue="test_set")
    for ax, dataset in zip(g.axes.flat, sorted_datasets):
        # ax.set_title(dataset)
        ax.set_xlabel("p(E)")
        ax.set_ylabel("Accuracy Error")
        ax.set_yscale("log")
        ax.set_ylim(1e-3,1)
        # ax.set_xticklabels(BATCH_SIZES)
        # ax.set_xticks(range(len(BATCH_SIZES)))
        ax.legend(title="Test Set", ncols=3, fontsize=8)
        # ax.set_yscale("log")

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
    plt.savefig("figures/dsd_acc_errors.pdf")
    plt.show()


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

    def get_datasetwise_risk():
        results_list = []
        for dsd in DSDS:
            for model_name in ["deeplabv3plus", "unet", "segformer"]:
                df = load_pra_df("Polyp", dsd, model=model_name, batch_size=1, samples=1000)
                for dataset in ["noise", "ind_test", "EndoCV2020", "EtisLaribDB", "CVC-ClinicDB"]:
                    if dataset not in df["shift"].unique():
                        print(f"Dataset {dataset} not in df")
                        continue
                    print()
                    for ood_val_shift in ["EndoCV2020", "EtisLaribDB", "CVC-ClinicDB"]:
                        sim = UniformBatchSimulator(df, ood_test_shift=dataset, ood_val_shift=ood_val_shift,
                                                    maximum_loss=0.5, use_synth=False)
                        results = sim.sim(1, 600)
                        results["Dataset"] = dataset
                        results["Model"] = model_name
                        results["ood_val_shift"] = ood_val_shift
                        results = results.groupby(["Model", "Tree", "Dataset"])[
                            ["True Risk", "Accuracy"]].mean().reset_index()
                        results["DSD"] = dsd
                        results_list.append(results)
        results = pd.concat(results_list)
        print(results.groupby(["Model", "DSD", "Dataset", "Tree"])[["True Risk", "Accuracy"]].mean())
        results.to_csv("datasetwise_risk.csv")

    def conditional_accs():
        df = load_polyp_data()
        # Set bin edges globally
        bins = 20
        bin_edges = np.linspace(df["IoU"].min(), df["IoU"].max(), bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Prepare FacetGrid
        g = sns.FacetGrid(df, col="feature_name", margin_titles=True, sharex=True, sharey=True, col_wrap=3)

        # Plot manually in each facet
        def stacked_bar(data, color=None, **kwargs):
            data["bin"] = pd.cut(data["IoU"], bins=bin_edges, include_lowest=True)
            count_df = data.groupby(["bin", "verdict"]).size().unstack(fill_value=0)
            proportion_df = count_df.div(count_df.sum(axis=1), axis=0).fillna(0)

            bottom = np.zeros(len(proportion_df))
            for label in proportion_df.columns:
                plt.bar(bin_centers, proportion_df[label], bottom=bottom,
                        width=bin_edges[1] - bin_edges[0], label=label,
                        edgecolor='white', align='center')
                bottom += proportion_df[label].values

        g.map_dataframe(stacked_bar)

        # Optional: Add legend only once
        handles, labels = plt.gca().get_legend_handles_labels()
        g.fig.legend(handles, labels, title="verdict", loc='lower center', fontsize="large", bbox_to_anchor=(0.8, 0.20))
        # increase size of legend

        g.set_axis_labels("IoU", "Proportion")
        g.set_titles(col_template="{col_name}")
        # Check if the total number of facets is more than 3 and adjust accordingly
        plt.tight_layout()

        plt.savefig("proportions.pdf")
        plt.show()

def plot_sensitivity_errors():
    dfs = []
    for dataset in DATASETS:
        try:
            df = pd.read_csv(f"pra_data/{dataset}_sensitivity_results.csv")
            best_guess = (df["ind_acc"].mean() + df["ood_val_acc"].mean()) / 2
            print(dataset, " : ", best_guess)
            df["Dataset"] = dataset
            df["best_guess_error"] = np.abs(df["Accuracy"] - best_guess)
            dfs.append(df)
        except:
            print(f"No data found for {dataset}")
    df = pd.concat(dfs)
    df = df[df["Tree"] == "Base Tree"]
    df = df[df["val_set"] != df["test_set"]]
    df["rate"] = round(df["rate"], 2)
    df["ba"] = round(df["ba"], 2)
    df.replace(DSD_PRINT_LUT, inplace=True)
    print(df.columns)
    df = df.groupby(["Dataset", "rate", "ba"])[["Accuracy Error", "ind_acc", "ood_val_acc"]].mean().reset_index()
    df.rename(columns={"rate": "$P(E)$", "ba": "$p(D_{e}(x)=E)$"}, inplace=True)
    g = sns.FacetGrid(df, col="Dataset", col_wrap=3)

    #sort by ba increasing order
    def plot_heatmap(data, **kws):
        heatmap_data = data.pivot(index="$p(D_{e}(x)=E)$", columns="$P(E)$", values="Accuracy Error")
        heatmap_data = heatmap_data.loc[::-1]  #higher ba is up
        sns.heatmap(heatmap_data, **kws, cmap="mako", vmin=0, vmax=(df["ind_acc"] - df["ood_val_acc"]).mean())

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
            cbar.ax.set_position([cbar.ax.get_position().x0 + left_padding / fig_width, cbar.ax.get_position().y0,
                                  cbar.ax.get_position().width, cbar.ax.get_position().height])
    plt.savefig("sensitivity_errors.pdf")
    plt.show()


def get_datasetwise_risk():
    results_list = []
    with tqdm(total=len(DSDS)*3*4*3) as pbar:
        for dsd in  ["energy", "knn", "grad_magnitude", "cross_entropy", "mahalanobis"]:
            for model_name in ["deeplabv3plus", "unet", "segformer"]:
                df = load_pra_df("Polyp", dsd, model=model_name, batch_size=1, samples=1000, prefix="polyp_data")
                if df.empty:
                    print(f"No data for {dsd} and {model_name}")
                    continue
                for dataset in ["ind_test", "EndoCV2020", "EtisLaribDB", "CVC-ClinicDB" ]:
                    if dataset not in df["shift"].unique():
                        print(f"Dataset {dataset} not in df")
                        continue
                    for ood_val_shift in ["EndoCV2020", "EtisLaribDB", "CVC-ClinicDB"]:
                        sim = UniformBatchSimulator(df, ood_test_shift=dataset, ood_val_shift=ood_val_shift, maximum_loss=0.5, use_synth=False)
                        results = sim.sim(1, 600)
                        results["Dataset"]=dataset
                        results["Model"]=model_name
                        results["ood_val_shift"]=ood_val_shift
                        results = results.groupby(["Model", "Tree",  "Dataset"])[["True Risk", "Accuracy"]].mean().reset_index()
                        results["DSD"]=dsd
                        results_list.append(results)
                        pbar.update(1)
    results = pd.concat(results_list)
    print(results.groupby(["Model", "DSD", "Dataset", "Tree"])[["True Risk", "Accuracy"]].mean())
    results.to_csv("datasetwise_risk.csv")


def get_risk_tables():
    df = pd.read_csv("datasetwise_risk.csv")
    df.replace(DSD_PRINT_LUT, inplace=True)
    df.replace({"ind_test": "Kvasir"}, inplace=True)
    df_base = df[df["Tree"] == "Base Tree"]
    df_dsd = df[df["Tree"] != "Base Tree"]
    print(df_base.groupby(["Dataset"])["True Risk"].mean())

    print(df_dsd.groupby(["Model", "DSD", "Dataset"])["True Risk"].mean())


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
                    dsd = OODDetector(filtered, "val_optimal") #train a dsd for ood_val_shift
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
                            "Dataset": dataset, "feature_name":feature_name, "OoD":ood, "D(OoD)":dood,
                            "batch_size": batch_size, "Error": cross_validated_error
                        })

    results = pd.DataFrame(errors)
    results.to_csv("conditional_accuracy_errors")
    # results = results.groupby(["Dataset", "feature_name", "ood", "D(ood)"]).mean().reset_index()
    g = sns.FacetGrid(results, row="OoD", col="D(OoD)", sharey=True)
    g.map_dataframe(sns.boxplot, x="batch_size", y="Error", hue="Dataset", palette=sns.color_palette())
    g.add_legend()
    g.set_axis_labels("Batch Size", "Prediction Accuracy MAE")
    # for ax in g.axes.flat:
    #     ax.set_yscale("log")
    #     ax.set_ylim(1-3,1)

    plt.savefig("figures/conditional_accuracy_errors.pdf")
    plt.show()


def assess_re_tree_predaccuracy_estimation_errors():
    errors = []

    for batch_size in BATCH_SIZES:
        all_data = load_all(batch_size)
        table = all_data.groupby(["Dataset", "shift"])["correct_prediction"].mean()
        fold_wise_error = table.reset_index()

        fold_wise_error_ood = fold_wise_error[~fold_wise_error["shift"].isin(["ind_val", "ind_test", "train"])]
        fold_wise_error_ind = fold_wise_error[fold_wise_error["shift"].isin(["ind_val", "ind_test"])]

        # compute per-dataset
        for dataset in fold_wise_error["Dataset"].unique():
            ind_vals = fold_wise_error_ind.loc[fold_wise_error_ind["Dataset"] == dataset, "correct_prediction"]
            ood_vals = fold_wise_error_ood.loc[fold_wise_error_ood["Dataset"] == dataset, "correct_prediction"]

            ind_diff = mean_pairwise_abs_diff(ind_vals)
            ood_diff = mean_pairwise_abs_diff(ood_vals)

            errors.append({"Dataset": dataset, "Batch Size": batch_size, "Type": "OoD", "Error": ood_diff})
            errors.append({"Dataset": dataset, "Batch Size": batch_size, "Type": "InD", "Error": ind_diff})
    errors_df = pd.DataFrame(errors)
    print(errors_df[errors_df["Dataset"] == "OfficeHome"])
    g = sns.FacetGrid(errors_df, col="Dataset", sharey=False, col_wrap=3)
    g.map_dataframe(sns.lineplot, x="Batch Size", y="Error", hue="Type", palette=sns.color_palette(), hue_order = ["InD", "OoD"])
    g.add_legend(bbox_to_anchor=(0.7, 0.5), loc="upper center")
    plt.savefig("figures/tree1_errors.pdf")
    plt.show()

    g = sns.FacetGrid(errors_df, col="Type", sharey=True)
    g.map_dataframe(sns.lineplot, x="Batch Size", y="Error", hue="Dataset", palette=sns.color_palette())
    g.add_legend()
    for ax in g.axes.flat:
        ax.set_xticks(BATCH_SIZES)
    plt.savefig("figures/tree1_errors.pdf")
    plt.show()


def ood_detector_accuracy_estimation_errors():
    errors = []

    for batch_size in BATCH_SIZES:
        tprs, tnrs = ood_verdict_shiftwise_accuracy_tables(batch_size=batch_size, filter_organic=True)

        # compute per-dataset
        for dataset in DATASETS:
            ind_vals = tnrs.loc[tnrs["Dataset"] == dataset, "tnr"]
            ood_vals = tprs.loc[tprs["Dataset"] == dataset, "tpr"]

            ind_diff = mean_pairwise_abs_diff(ind_vals)
            ood_diff = mean_pairwise_abs_diff(ood_vals)

            errors.append({"Dataset": dataset, "Batch Size": batch_size, "Type": "OoD", "Error": ood_diff})
            errors.append({"Dataset": dataset, "Batch Size": batch_size, "Type": "InD", "Error": ind_diff})

    errors_df = pd.DataFrame(errors)
    g = sns.FacetGrid(errors_df, col="Type", sharey=True, sharex=True)
    g.map_dataframe(sns.lineplot, x="Batch Size", y="Error", hue="Dataset")
    for ax in g.axes.flat:
        ax.set_xticks(BATCH_SIZES)
    g.add_legend()
    plt.savefig("figures/dsd_accuracy.pdf")
    plt.show()
