# from albumentations.random_utils import normal
from scipy.stats import ks_2samp
from seaborn import FacetGrid
import warnings


warnings.filterwarnings("ignore")
from experiments.accuracy_prediction import *
from experiments.dataset_analysis import *
from experiments.pra import *
from experiments.loss_regression import *
from experiments.appendices import *

def simulate_dsd_accuracy_estimation(data, rate, val_set, test_set, ba, tpr, tnr, dsd):
    sim = UniformBatchSimulator(data, ood_test_shift=test_set, ood_val_shift=val_set, estimator=ErrorAdjustmentEstimator,
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
        data = ood_detector_correctness_prediction_accuracy(32)
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









    # g = sns.FacetGrid(data.reset_index(), col="Dataset", row="D(ood)")
    # g.map_dataframe(sns.lineplot, x="batch_size", y="correct_prediction", hue="feature_name")




def examine_feature_distributions():
    df = load_all(1)
    df = df[df["fold"]!="train"]

    p_value_data = []
    for dataset, feature_name in itertools.product(DATASETS, DSDS):
        subdf_dataset = df[(df["Dataset"]==dataset)&(df["feature_name"]==feature_name)]
        for fold in subdf_dataset["fold"].unique():
            subdf = subdf_dataset[subdf_dataset["fold"]==fold]
            if subdf.empty:
                print(f"Empty dataframe for {dataset} {feature_name}")
                continue

            subdf["idx_quant"] =  pd.qcut(subdf["idx"], q=50, labels=False)
            unique_classes = subdf["class"].unique()
            unique_idx = subdf["idx_quant"].unique()

            for idx in unique_idx:
                data1 = subdf[subdf["idx_quant"] == idx]["feature"]
                data2 = subdf["feature"].sample(len(data1))

                p_value = ks_2samp(data1, data2).pvalue
                p_value_data.append(
                    {"Dataset": dataset, "feature_name": feature_name, "Bias": "Temporal", "x":idx,
                     "p": p_value})
            if dataset!="Polyp":
                for cls in unique_classes:
                    data1 = subdf[subdf["class"]==cls]["feature"]
                    data2 = subdf["feature"].sample(len(data1))
                    p_value = ks_2samp(data1,data2).pvalue
                    p_value_data.append({"Dataset": dataset, "feature_name": feature_name, "Bias":"Class", "x":cls, "p": p_value})

    data = pd.DataFrame(p_value_data)
    print(data.groupby(["Dataset", "Bias", "feature_name"])["p"].agg(["mean"]))


def examine_aggregated_feature_distributions(batch_size):
    dfs = load_all_biased("old_debiased_data", filter_batch=batch_size)
    dfs = dfs[dfs["fold"]!="train"]
    dfs["feature"] = dfs["feature"].astype(float)
    p_value_data = []
    # g = sns.FacetGrid(dfs, col="Dataset", row="feature_name", sharex=False, sharey=False)
    # g.map_dataframe(sns.kdeplot, x="feature", hue="bias", common_norm = False, fill=True)
    # plt.tight_layout()
    # plt.show()
    for dataset, feature in itertools.product(DATASETS, DSDS):
        data_feature_df = dfs[(dfs["Dataset"]==dataset) & (dfs["feature_name"]==feature)]

        for fold in data_feature_df["fold"].unique():
            subdf = data_feature_df[data_feature_df["fold"]==fold]
            if feature == "rabanser":
                subdf = subdf[subdf["k"] == 0]
            else:
                subdf = subdf[subdf["k"] == -1]
            if subdf.empty:
                continue
            random_rabanser = subdf[subdf["bias"] == "RandomSampler"]
            for bias in SAMPLERS:
                if bias=="RandomSampler":
                    continue
                by_bias = subdf[subdf["bias"]==bias]
                if by_bias.empty:
                    print(f"Empty for {dataset} {bias}")
                    continue
                p_val = ks_2samp(by_bias["feature"], random_rabanser["feature"]).pvalue
                p_value_data.append({"Dataset": dataset, "feature_name":feature, "fold":fold, "bias": SAMPLER_LUT[bias], "p": p_val})
    df = pd.DataFrame(p_value_data)
    print(df.groupby(["Dataset", "feature_name", "bias"])["p"].mean())







def polyp_tables_and_plots():
    df = load_polyp_data()

    #data passthrough

    ood_data = df # unseen ood data

    ood_data =ood_data[ood_data["fold"]!=ood_data["Calibration Fold"]]
    print(ood_data.head(10))
    print(ood_data.groupby(["model","feature_name", "correct_prediction"])["verdict"].mean())

    plot_data = ood_data.groupby(["model", "fold", "feature_name", "correct_prediction"])["verdict"].mean()
    print(plot_data)
    #accuracy table
    # print(df.groupby(["fold", "model"])[["correct_prediction", "IoU"]].mean())

    #ood detector effect


    # risk plot
    # get_datasetwise_risk()
    # get_risk_tables()



    # print(df)

def run_methodological_experiments():
    accuracy_table()
    dataset_summaries()


def run_acc_prediction_experiments():
    # test_generalization_gap_estimation(1)
    # get_acc_prediction_results(1)
    acc_prediction_table()
    error_heatmap()
    # error_per_accuracy()


if __name__ == '__main__':
    run_acc_prediction_experiments()



