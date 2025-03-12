import numpy as np
from click import style
from seaborn import FacetGrid
from watchdog.observers.inotify_c import inotify_init

from simulations import *
import matplotlib.pyplot as plt
from utils import *
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from components import OODDetector

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

def uniform_bernoulli(data, estimator=BernoulliEstimator, load=True):
    if load:
        results = pd.read_csv("risk_uniform_bernoulli.csv")
    else:
        x = np.linspace(0,1,11)
        result_list = []
        for i in x:
            sim = SystemSimulator(data, ood_test_shift=ENDOCV, ood_val_shift=ETISLARIB, maximum_loss=0.5, estimator=estimator, dsd_tpr=1, dsd_tnr=1)
            results = sim.uniform_rate_sim(i, 10000)
            print(results.mean())
            results["p"] = i
            result_list.append(results)
        results = pd.concat(result_list)
        results.to_csv("risk_uniform_bernoulli.csv")

    # sns.lineplot(results, x="p", y="error")
    # plt.savefig("risk_uniform_bernoulli.eps")
    # plt.show()
    return results

def single_run(data, estimator=BernoulliEstimator):
    sim = SystemSimulator(data, ood_test_shift="dslr", ood_val_shift="webcam", maximum_loss=0.5, estimator=estimator, dsd_tnr=0.91, dsd_tpr=0.9)
    results = sim.uniform_rate_sim(1, 10000)
    sim.detector_tree.print_tree()
    print(results.groupby(["Tree"]).mean())
    sim = SystemSimulator(data, ood_test_shift="dslr", ood_val_shift="webcam", maximum_loss=0.5, estimator=estimator, dsd_tnr=0.9, dsd_tpr=0.9)
    results = sim.uniform_rate_sim(0.5, 10000)
    sim.detector_tree.print_tree()
    print(results.groupby(["Tree"]).mean())
    sim = SystemSimulator(data, ood_test_shift="dslr", ood_val_shift="webcam", maximum_loss=0.5, estimator=estimator, dsd_tnr=0.9, dsd_tpr=0.9)
    results = sim.uniform_rate_sim(0, 10000)
    sim.detector_tree.print_tree()
    print(results.groupby(["Tree"]).mean())

    # sim = SystemSimulator(data, ood_test_shift=CVCCLINIC, maximum_loss=0.5, estimator=estimator)
    # results = sim.uniform_rate_sim(0.5, 10000)
    # print(results.mean())

def collect_tpr_tnr_sensitivity_data():
    bins = 5
    dfs = []
    total_num_tpr_tnr = np.sum(
    [i + j / 2 >= 0.5 for i in np.linspace(0, 1, bins) for j in np.linspace(0, 1, bins)])
    for dataset in DATASETS:
        data = load_pra_df(dataset_name=dataset, feature_name="knn", batch_size=1,
                           samples=1000)  # we are just interested in the loss and oodness values, knn is arbitray
        ood_sets = data[~data["shift"].isin(["ind_val", "ind_test", "train"])]["shift"].unique()

        with tqdm(total=bins * total_num_tpr_tnr*(len(ood_sets)-1)*len(ood_sets)) as pbar:
            for val_set in ood_sets:
                for test_set in ood_sets:
                    if test_set=="noise":
                        continue #used only to estimate accuracies
                    for rate in np.linspace(0, 1, bins):
                        for tpr in np.linspace(0, 1, bins):
                            for tnr in np.linspace(0, 1, bins):
                                if (tnr+tpr)/2 < 0.5:
                                    continue
                                sim = SystemSimulator(data, ood_test_shift=test_set, ood_val_shift=val_set, estimator=BernoulliEstimator, dsd_tpr=tpr, dsd_tnr=tnr)
                                results = sim.uniform_rate_sim(rate, 600)
                                results = results.groupby(["Tree"]).mean().reset_index()

                                # results = results.mean()
                                results["tpr"] = tpr
                                results["tnr"] = tnr
                                results["ba"] = (tpr + tnr) / 2
                                results["rate"] = rate
                                results["test_set"] = test_set
                                results["val_set"] = val_set
                                results["Dataset"] = dataset
                                # results = results.groupby(["tpr", "tnr", "rate", "test_set", "val_set", "Tree"]).mean().reset_index()

                                dfs.append(results)
                                pbar.update(1)
    df_final = pd.concat(dfs)
    print(df_final.head(10))
    df_final.to_csv(f"pra_data/sensitivity_results.csv")

def collect_dsd_accuracy_estimation_data():

    dfs = []
    bins=5
    with tqdm(total=len(DSDS)*len(BATCH_SIZES)) as pbar:
        for batch_sizes in BATCH_SIZES:
            dsd_accuracies = fetch_dsd_accuracies(batch_size=batch_sizes, plot=False, samples=1000)
            best_dsds = dsd_accuracies[dsd_accuracies.groupby(["Dataset", "DSD"])[["ba"]].idxmax()].reset_index()

            for dataset in DATASETS:
                dsd = best_dsds[(best_dsds["Dataset"]==dataset)]["DSD"].values[0]
                tpr, tnr, ba = dsd_accuracies[(dsd_accuracies["Dataset"]==dataset)&(dsd_accuracies["DSD"]==dsd)][["tpr", "tnr", "ba"]].mean()

                data = load_pra_df(dataset, dsd, batch_size=batch_sizes, samples=1000)
                ood_sets = data[~data["shift"].isin(["ind_val", "ind_test", "train"])]["shift"].unique()
                for val_set in ood_sets:
                    for test_set in ood_sets:
                        if test_set=="noise":
                            continue #used only to estimate accuracies
                        for rate in np.linspace(0, 1, bins):
                            for dsd in DSDS:
                                sim = SystemSimulator(data, ood_test_shift=test_set, ood_val_shift=val_set, estimator=BernoulliEstimator, dsd_tpr=tpr, dsd_tnr=tnr)
                                results = sim.uniform_rate_sim(rate, 600)
                                results = results.groupby(["Tree"]).mean().reset_index()
                                # results = results.mean()
                                results["dsd"] = dsd
                                results["rate"] = rate
                                results["test_set"] = test_set
                                results["val_set"] = val_set
                                # results = results.groupby(["tpr", "tnr", "rate", "test_set", "val_set", "Tree"]).mean().reset_index()

                                dfs.append(results)
                                pbar.update(1)
    df_final = pd.concat(dfs)
    print(df_final.head(10))
    df_final.to_csv(f"pra_data/dsd_results.csv")


def compare_risk_tree_accuracy_estimators():
    df = pd.read_csv("tpr_tnr_sensitivity.csv").groupby(["tpr", "tnr", "rate", "test_set", "val_set"]).mean().reset_index()
    df["tpr"] = df["tpr"].round(2)
    df["tnr"] = df["tnr"].round(2)
    df["rate"] = df["rate"].round(2)
    df["ba"] = round((df["tpr"] + df["tnr"]) / 2, 2)
    df = df[df["ba"] >= 0.5]
    df["RV Tree Error"] = np.abs(df['E[f(x)=y]'] - df['correct_prediction'])
    df["Base Event Tree Accuracy"] = df["Estimated Rate"] * df["ood_val_acc"] + (1 - df["Estimated Rate"]) * df["ind_acc"]
    df["Base Event Tree Error"] = np.abs(df["baseline_acc"] - df["correct_prediction"])



    sns.barplot(df, x="ba", y="RV Tree Error")
    # baseline error is just the small risk tree

    sns.barplot(df, x="ba", )


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
    for dataset in DATASETS:
        for feature in DSDS:
            df = load_pra_df(dataset, feature, batch_size=batch_size, samples=samples)
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
    df = pd.DataFrame(data)
    df.to_csv("dsd_accuracies.csv")
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

def risk_tree_cba():
    risk_tree = DetectorEventTree(0.95, 0.95, 0.95, 0.90, 0.91, 0.15)

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



def t_check():
    df = load_pra_df("Polyp", "knn",batch_size=32)
    df = df[df["shift"]!="noise"]
    for shift in df["shift"].unique():
        print(f"plotting : {shift}")

        plt.hist(df[df["shift"]==shift]["feature"], bins=100, alpha=0.5, density=True, label=shift)
        plt.legend()
    plt.show()

if __name__ == '__main__':
    #data = load_pra_df(dataset_name="Office31", feature_name="knn", batch_size=1, samples=1000)
    # collect_rate_estimator_data()
    # eval_rate_estimator()
    # t_check()
    # fetch_dsd_accuracies(32, plot=True)
    # plot_dsd_accuracies(1000)
    # plot_rate_estimation_errors_for_dsds()
    # accuracy_by_fold()
    # accuracy_by_fold_and_dsd_verdict()
    #print(data)

    # collect_tpr_tnr_sensitivity_data()
    collect_dsd_accuracy_estimation_data()
    # uniform_bernoulli(data, load = False)
