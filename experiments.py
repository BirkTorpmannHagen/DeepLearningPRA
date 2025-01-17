import numpy as np
import pandas as pd

from simulations import *
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
import seaborn as sns
from tqdm import tqdm

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
            sim = SystemSimulator(data, ood_test_shift=ENDOCV, maximum_loss=0.5, estimator=estimator)
            results = sim.uniform_rate_sim(i, 10000)
            re, ndsdre, tr, ndsdtr = results.mean()
            result_list.append({"p": i, "Risk":re, "DSD":True, "True Risk": False, "error":np.abs(tr-re)})
            result_list.append({"p": i, "Risk":tr, "DSD":True, "True Risk": True, "error":np.abs(tr-re)})
            result_list.append({"p": i, "Risk":ndsdre, "DSD":False, "True Risk": False, "error":np.abs(ndsdre-ndsdtr)})
            result_list.append({"p": i, "Risk":ndsdtr, "DSD":False, "True Risk": True, "error":np.abs(ndsdre-ndsdtr)})
        results = pd.DataFrame(result_list)
        results.to_csv("risk_uniform_bernoulli.csv")
        results = pd.read_csv("risk_uniform_bernoulli.csv")


    results = results[results["DSD"]==True]
    sns.lineplot(results, x="p", y="error")
    plt.savefig("risk_uniform_bernoulli.eps")
    plt.show()
    return results

def single_run(data, estimator=BernoulliEstimator):
    sim = SystemSimulator(data, ood_test_shift=ENDOCV, maximum_loss=0.5, estimator=estimator)
    results = sim.uniform_rate_sim(0.5, 10000)
    print(results.mean())
    sim = SystemSimulator(data, ood_test_shift=CVCCLINIC, maximum_loss=0.5, estimator=estimator)
    results = sim.uniform_rate_sim(0.5, 10000)
    print(results.mean())

def collect_tpr_tnr_sensitivity_data(data):
    dfs = []
    for rate in np.linspace(0, 1, 26):
        for tpr in np.linspace(0, 1, 26):
            for tnr in np.linspace(0, 1, 26):

                sim = SystemSimulator(data, ood_test_shift=ENDOCV, maximum_loss=0.5, estimator=BernoulliEstimator, dsd_tpr=tpr, dsd_tnr=tnr)
                results = sim.uniform_rate_sim(rate, 1000)
                results = results.mean()
                results["tpr"] = tpr
                results["tnr"] = tnr
                results["rate"] = rate
                results = results.to_dict()
                dfs.append(results)
    df_final = pd.DataFrame(dfs)
    df_final.to_csv("tpr_tnr_sensitivity.csv")


def plot_tpr_tnr_sensitivity():
    df = pd.read_csv("tpr_tnr_sensitivity.csv").groupby(["tpr", "tnr", "rate"]).mean().reset_index()
    df["tpr"] = df["tpr"].round(2)
    df["tnr"] = df["tnr"].round(2)

    df["Error"] = np.abs(df["Risk Estimate"] - df["True Risk"])
    df = df[df["rate"] == 1]

    # Ensure `tpr` and `tnr` are sorted properly
    df = df.sort_values(by=["tpr", "tnr"])
    pivot_table = df.pivot(index="tpr", columns="tnr", values="Error")

    # Reverse the `tpr` axis (y-axis) order
    pivot_table = pivot_table.loc[::-1]

    sns.heatmap(pivot_table)
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

def fetch_dsd_accuracies():
    data = []
    for dsd in ["knn", "energy", "cross_entropy","typicality", "grad_magnitude"]:
        df = load_pra_df(dsd, batch_size=10, samples=100)
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
                tpr = (ind_val["feature"]<i).mean()
                tnr = (ood_val["feature"]>i).mean()
                this_ba = (tpr + tnr) / 2
                if this_ba>ba:
                    ba = this_ba
                    best_tpr = tpr
                    best_tnr = tnr
            else:
                tpr = (ind_val["feature"]>i).mean()
                tnr = (ood_val["feature"]<i).mean()
                this_ba = (tpr + tnr) / 2
                if this_ba>ba:
                    ba = this_ba
                    best_tpr = tpr
                    best_tnr = tnr
        data.append({"dsd":dsd, "ba":ba, "tpr":best_tpr, "tnr":best_tnr})
    return pd.DataFrame(data)


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



if __name__ == '__main__':
    # data = load_pra_df("knn", batch_size=10, samples=1000)
    # print("asdadsa")
    # single_run(data)
    # uniform_bernoulli(data, load = False)
    # collect_tpr_tnr_sensitivity_data(data)
    # plot_tpr_tnr_sensitivity()
    # plot_ba_rate_sensitivity(uniform_bernoulli(data, load=False))
    # eval_rate_estimator()
    #plot_rate_estimates()
    print(fetch_dsd_accuracies())

