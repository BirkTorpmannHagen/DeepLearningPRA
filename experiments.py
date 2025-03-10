import itertools

from simulations import *
import matplotlib.pyplot as plt
from utils import *
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import itertools
from components import OODDetector

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

def collect_tpr_tnr_sensitivity_data(data, ood_sets):
    bins = 11
    for val_set in ["noise"]+ood_sets:
        for test_set in ood_sets:
            dfs = []
            total_num_tpr_tnr = np.sum(
                [i + j / 2 >= 0.5 for i in np.linspace(0, 1, bins) for j in np.linspace(0, 1, bins)])
            with tqdm(total=bins * total_num_tpr_tnr) as pbar:

                # if test_set == val_set:
                #     continue
                for rate in np.linspace(0, 1, bins):
                    for tpr in np.linspace(0, 1, bins):
                        for tnr in np.linspace(0, 1, bins):
                            if tnr+tpr/2 < 0.5:
                                continue
                            sim = SystemSimulator(data, ood_test_shift=test_set, ood_val_shift=val_set, estimator=BernoulliEstimator, dsd_tpr=tpr, dsd_tnr=tnr)
                            results = sim.uniform_rate_sim(rate, 600)
                            # results = results.mean()
                            results["tpr"] = tpr
                            results["tnr"] = tnr
                            results["ba"] = (tpr + tnr) / 2
                            results["rate"] = rate
                            results["test_set"] = test_set
                            results["val_set"] = val_set
                            results["dataset"] = data["dataset"].unique()[0]
                            # results = results.groupby(["tpr", "tnr", "rate", "test_set", "val_set", "Tree"]).mean().reset_index()
                            dfs.append(results)
                            pbar.update(1)
                df_final = pd.concat(dfs)
                print(df_final.head(10))
                df_final.to_csv(f"pra_data/{data['dataset'].unique()[0]}_{val_set}_{test_set}_results.csv")

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

def fetch_dsd_accuracies(batch_size=32, plot=False):
    data = []
    for dataset in DATASETS:
        for feature in DSDS:
            df = load_pra_df(dataset, feature, batch_size=batch_size, samples=1000)
            df = df[df["shift"]!="noise"]
            if df.empty:
                continue
            ind_val = df[df["shift"]=="ind_val"]
            ind_test = df[df["shift"]=="ind_test"]
            ood_folds = df[~df["fold"].isin(["ind_val", "ind_test", "train"])]["shift"].unique()
            for ood_val_fold in ood_folds:
                for ood_test_fold in ood_folds:
                    if ood_val_fold != ood_test_fold:
                        continue
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
    print(df.groupby(["Dataset", "DSD", "val_fold", "test_fold"])[["tpr", "tnr", "ba"]].mean())
    return df

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
    all_data = pd.concat([load_pra_df(dataset_name=dataset_name, feature_name="knn", batch_size=1, samples=1000) for dataset_name in
                          DATASETS])
    table = all_data.groupby(["dataset", "shift"])["correct_prediction"].mean()
    print(table)
    fold_wise_error = table.reset_index()
    fold_wise_error_ood = fold_wise_error[~fold_wise_error["shift"].isin(["ind_val", "ind_test", "train"])]
    fold_wise_error_ind = fold_wise_error[fold_wise_error["shift"].isin(["ind_val", "ind_test", "train"])]

    # Calculate the difference matrix
    for label, data_group in zip(["InD", "OoD"], [fold_wise_error_ind, fold_wise_error_ood]):
        diff_matrices = {}
        for dataset, group in data_group.groupby('dataset'):
            group.set_index('shift', inplace=True)
            accuracy_values = group['correct_prediction'].to_numpy()
            diff_matrix = pd.DataFrame(
                data=np.subtract.outer(accuracy_values, accuracy_values),
                index=group.index,
                columns=group.index
            )
            # Store the matrix for each dataset
            diff_matrices[dataset] = diff_matrix

            # Display the difference matrices for each dataset

        print(label)
        for dataset, matrix in diff_matrices.items():
            mat = matrix.to_numpy()
            avg  = round(np.sum(np.abs(mat)) / (matrix.shape[0] * matrix.shape[1] - matrix.shape[0]),3)
            min = round(np.min(np.abs(mat[mat>0])),3)
            max = round(np.max(np.abs(mat)),3)
            print(f"{dataset} & {min}  & {avg} & {max} \\\\")

def accuracy_by_fold_and_dsd_verdict(batch_size=16):
    df = load_all(batch_size)
    df = df[df["shift"]!="noise"]
    fig, ax = plt.subplots(nrows=2, ncols = len(DATASETS))
    data = []
    for i, dataset in enumerate(DATASETS):
        for feature in DSDS:
            for j, ood in enumerate([False, True]):
                filtered  = df[(df["Dataset"]==dataset)&(df["feature_name"]==feature)]
                shifts = filtered["shift"].unique()

                for ood_val_shift in shifts:
                    filtered_copy = filtered
                    dsd = OODDetector(filtered, ood_val_shift)
                    filtered_copy["D(ood)"] = filtered_copy.apply(lambda row: dsd.predict(row), axis=1)
                    filtered_copy["ood_val_shift"]=ood_val_shift
                    
                    accuracy = filtered_copy.groupby(["Dataset", "ood_val_shift", "shift", "D(ood)"])["correct_prediction"].mean()
                    data.append(accuracy.reset_index())
    df = pd.concat(data)
    print(df.groupby(["Dataset", "ood_val_shift", "shift", "D(ood)"]).mean())


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

    # plot_rate_estimation_errors_for_dsds()
    # accuracy_by_fold()
    accuracy_by_fold_and_dsd_verdict()
    #print(data)
    #collect_tpr_tnr_sensitivity_data(data, ood_sets = ["dslr", "webcam"])
    # uniform_bernoulli(data, load = False)
