from anaconda_project.requirements_registry.provider import ProvideResult
from rich.pretty import pretty_repr
from seaborn import FacetGrid

from simulations import *
import matplotlib.pyplot as plt
from utils import *
import seaborn as sns
from tqdm import tqdm

from utils import load_pra_df


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
    sim = SystemSimulator(data, ood_test_shift=ENDOCV, ood_val_shift=ETISLARIB, maximum_loss=0.5, estimator=estimator, dsd_tnr=1, dsd_tpr=1)
    results = sim.uniform_rate_sim(1, 10000)
    sim.detector_tree.print_tree()
    sim.detector_tree.calculate_expected_accuracy(sim.detector_tree.root, debug=True)
    print(results.mean())
    # sim = SystemSimulator(data, ood_test_shift=CVCCLINIC, maximum_loss=0.5, estimator=estimator)
    # results = sim.uniform_rate_sim(0.5, 10000)
    # print(results.mean())

def collect_tpr_tnr_sensitivity_data(data):
    dfs = []
    bins = 4
    total_num_tpr_tnr = np.sum([i+j/2>=0.5 for i in np.linspace(0, 1, bins) for j in np.linspace(0, 1, bins)])
    with tqdm(total=6*bins*total_num_tpr_tnr) as pbar:
        for test_set in [CVCCLINIC, ETISLARIB, ENDOCV]:
            for val_set in [CVCCLINIC, ETISLARIB, ENDOCV]:
                if test_set == val_set:
                    continue
                for rate in np.linspace(0, 1, bins):
                    for tpr in np.linspace(0, 1, bins):
                        for tnr in np.linspace(0, 1, bins):
                            if tnr+tpr/2 < 0.5:
                                continue
                            sim = SystemSimulator(data, ood_test_shift=test_set, ood_val_shift=val_set, maximum_loss=0.5, estimator=BernoulliEstimator, dsd_tpr=tpr, dsd_tnr=tnr)
                            results = sim.uniform_rate_sim(rate, 1000)
                            # results = results.mean()
                            results["tpr"] = tpr
                            results["tnr"] = tnr
                            results["rate"] = rate
                            results["test_set"] = test_set
                            results["val_set"] = val_set
                            results = results.groupby(["tpr", "tnr", "rate", "test_set", "val_set", "Tree"]).mean().reset_index()
                            # print(results)
                            results = results.to_dict()
                            dfs.append(results)
                            pbar.update(1)
    df_final = pd.DataFrame(dfs)
    df_final.to_csv("tpr_tnr_sensitivity.csv")

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

def plot_tpr_tnr_sensitivity():
    # Load and preprocess data
    df = pd.read_csv("tpr_tnr_sensitivity.csv").groupby(["tpr", "tnr", "rate", "test_set", "val_set"]).mean().reset_index()
    df["tpr"] = df["tpr"].round(2)
    df["tnr"] = df["tnr"].round(2)
    df["Error"] = np.abs(df['E[f(x)=y]'] - df['correct_prediction'])


    df = df.sort_values(by=["tpr", "tnr"])
    df["rate"] = df["rate"].round(2)
    df["ba"] = round((df["tpr"] + df["tnr"]) / 2, 2)
    df = df[df["ba"] >= 0.5]

    # Prepare data for heatmaps
    facet = df.groupby(["ba", "rate", "val_set", "test_set"])[["E[f(x)=y]", "correct_prediction", "Error"]].mean().reset_index()
    # facet = facet.pivot(index=["val_set", "test_set"], columns="rate", values="Error")

    # Define heatmap function
    def draw_heatmap(data, **kws):
        # Extract numeric data for the heatmap
        heatmap_data = data.pivot(index="ba", columns="rate", values="Error")
        heatmap_data = heatmap_data.loc[::-1]

        sns.heatmap(heatmap_data, **kws, cmap="mako", vmin=0, vmax=(df["ind_acc"]-df["ood_val_acc"]).mean())

    # Create FacetGrid and plot heatmaps
    g = sns.FacetGrid(facet.reset_index(), col="test_set", row="val_set", col_order=[CVCCLINIC, ETISLARIB, ENDOCV], row_order=[CVCCLINIC, ETISLARIB, ENDOCV], margin_titles=True)
    g.map_dataframe(draw_heatmap)
    plt.savefig("cross_validated_accuracy_estimation_error.eps")
    plt.show()

    # Additional analysis and plotting
    print(df[df["ba"] == 1].groupby(["ba", "rate"])[["E[f(x)=y]", "correct_prediction", "Error"]].mean().reset_index())
    df = df.groupby(["ba", "rate", "val_set", "test_set"]).mean().reset_index()
    df = df.groupby(["ba", "rate"])["Error"].mean().reset_index()
    pivot_table = df.pivot(index="ba", columns="rate", values="Error")
    pivot_table = pivot_table.loc[::-1]
    sns.heatmap(pivot_table, cmap="mako")
    plt.legend()
    plt.savefig("tpr_tnr_sensitivity.eps")
    plt.show()



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

def fetch_dsd_accuracies(dsd):
        df = load_pra_df(dsd, batch_size=1, samples=100)
        ind_val = df[df["fold"]=="ind_val"]
        ood_val = df[df["fold"]==ETISLARIB]
        ood_test = df[df["fold"]==ENDOCV]
        ind_test = df[df["fold"]=="ind_test"]
        val = pd.concat([ind_val, ood_val])
        ba = 0
        best_tpr = 0
        best_tnr = 0
        for i in np.linspace(val["feature"].min(), val["feature"].max(), 100):
            if ind_val["feature"].mean()<i<ood_val["feature"].mean():
                tnr = (ind_val["feature"]<i).mean()
                tpr = (ood_val["feature"]>i).mean()
                this_ba = (tpr + tnr) / 2
                if this_ba>ba:
                    ba = this_ba
                    best_tpr = tpr
                    best_tnr = tnr
            else:
                tnr = (ind_val["feature"]>i).mean()
                tpr = (ood_val["feature"]<i).mean()
                this_ba = (tpr + tnr) / 2
                if this_ba>ba:
                    ba = this_ba
                    best_tpr = tpr
                    best_tnr = tnr
        return ba


def eval_rate_estimator():
    data = []
    with tqdm(total=26*26*26*9) as pbar:
        for rate in tqdm(np.linspace(0, 1, 26)):
            for tpr in np.linspace(0, 1, 26):
                for tnr in np.linspace(0, 1, 26):
                    for tl in [10, 20, 30, 50, 60, 100, 200, 500, 1000]:
                        pbar.update(1)
                        re = BernoulliEstimator(prior_rate=rate, tpr=tpr, tnr=tnr)
                        sample = re.sample(10_000, rate)
                        dsd = get_dsd_verdicts_given_true_trace(sample, tpr, tnr)
                        for i in np.array_split(dsd, int(10_000//tl)):
                            re.update(i)
                            rate_estimate = re.get_rate()
                            data.append({"rate": rate, "tpr": tpr, "tnr": tnr, "tl": tl, "rate_estimate": rate_estimate})
    df = pd.DataFrame(data)
    df.to_csv("rate_estimator_eval.csv")
    print(df)

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



if __name__ == '__main__':
    data = load_pra_df("knn", batch_size=1, samples=1000)
    data["correct_prediction"] = data["loss"] < 0.5
    print(data[data["fold"]==CVCCLINIC]["correct_prediction"].mean())
    print(data[data["fold"]=="ind_val"]["correct_prediction"].mean())
    # print("asdadsa")
    # single_run(data)
    # uniform_bernoulli(data, load = False)
    collect_tpr_tnr_sensitivity_data(data)
    # plot_tpr_tnr_sensitivity()
    # plot_ba_rate_sensitivity(uniform_bernoulli(data, load=False))
    # eval_rate_estimator()
    #plot_rate_estimates()
    # print(fetch_dsd_accuracies())

