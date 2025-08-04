# from albumentations.random_utils import normal

from seaborn import FacetGrid
import warnings

from experiments.runtime_classification import *

warnings.filterwarnings("ignore")
from simulations import *
from utils import *
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


if __name__ == '__main__':
    #accuracies on each dataset
    from experiments.dataset_analysis import *
    from experiments.runtime_classification import *

    # accuracy_table()
    #class distribution
    # dataset_summaries()
    """
        Runtime Verification
    """
    # loss_correctness_test()
    for batch_size in BATCH_SIZES[1:]:
        print(f"Running batch size {batch_size}")
        debiased_ood_detector_correctness_prediction_accuracy(batch_size)
        # ood_detector_correctness_prediction_accuracy(batch_size, shift="")
        # ood_verdict_accuracy_table(batch_size)
    #simple batching
    # ood_verdict_plots_batched()

    # ood_detector_correctness_prediction_accuracy(64)
    # ood_verdict_accuracy_table(32)
    # ood_verdict_accuracy_table(batch)



    # for batch_size in [1, 8, 16]:
    #     ood_detector_correctness_prediction_accuracy(batch_size)


    #runtime verification
    # plot_batching_effect("NICO", "entropy")
    # eval_debiased_ood_detectors()
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