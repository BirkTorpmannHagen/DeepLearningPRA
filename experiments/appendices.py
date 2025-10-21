import matplotlib.pyplot as plt

from experiments.runtime_classification import *


def investigate_training_wise_thresholding(batch_size, shift="normal"):
    df = load_all(prefix="fine_data", batch_size=batch_size, shift=shift, samples=100)
    data = []
    for dataset, feature in itertools.product(DATASETS, DSDS):
        data_filtered = df[(df["Dataset"] == dataset) & (df["feature_name"] == feature)]
        for ood_val_fold in data_filtered["shift"].unique():
            data_copy = data_filtered.copy()
            if ood_val_fold in ["train", "ind_val", "ind_test"] or ood_val_fold in SYNTHETIC_SHIFTS:
                # dont calibrate on ind data or synthetic ood data
                continue
            data_train = data_copy[
                (data_copy["shift"] == ood_val_fold) | (data_copy["shift"] == "train")]
            dsd = OODDetector(data_train, ood_val_fold, threshold_method="val_optimal")
            # dsd.kde()
            for ood_test_fold in data_filtered["fold"].unique():
                if ood_test_fold in ["train", "ind_val", "ind_test"]:
                    continue
                for ind_test_fold in ["ind_test", "ind_val"]:
                    data_test = data_copy[(data_copy["fold"] == ood_test_fold) | (data_copy["fold"] == ind_test_fold)]
                    shift = ood_test_fold.split("_")[0]
                    shift_intensity = ood_test_fold.split("_")[-1] if "_" in ood_test_fold else "Organic"
                    data_copy["ood"] = ~data_copy["correct_prediction"]
                    tpr, tnr, ba = dsd.get_metrics(data_test)
                    if np.isnan(ba):
                        continue
                    data.append({"Dataset": dataset, "feature_name": feature, "Threshold Method": "val_optimal",
                                      "OoD==f(x)=y": True, "Performance Calibrated": False,
                                      "OoD Val Fold": ood_val_fold, "InD Val Fold":"Train",
                                      "OoD Test Fold": ood_test_fold, "InD Test Fold": ind_test_fold, "Shift": shift,
                                      "Shift Intensity": shift_intensity, "tpr": tpr, "tnr": tnr, "ba": ba})
    df = pd.DataFrame(data)

    trainingwise = df.groupby(["Dataset", "feature_name", "InD Test Fold"])["tnr"].mean().reset_index()
    ood_results = get_all_ood_detector_data(1, filter_thresholding_method=True, filter_correctness_calibration=True,
                                            filter_organic=True, filter_best=False)

    regular = ood_results[ood_results["OoD==f(x)=y"]].groupby(["Dataset", "feature_name", "InD Test Fold"])[
        "tnr"].mean().reset_index()
    trainingwise.replace(DSD_PRINT_LUT, inplace=True)
    # print(regular)
    # print(trainingwise)
    merged = pd.merge(trainingwise, regular, on=["Dataset", "feature_name", "InD Test Fold"],
                      suffixes=("_training", "_regular"))
    merged["tnr_diff"] = merged["tnr_training"] - merged["tnr_regular"]
    print(merged.groupby(["Dataset", "feature_name"])[["tnr_training", "tnr_regular", "tnr_diff"]].mean().reset_index())

def plot_ind_correctness_by_ood_feature():
    df = load_all(prefix="coarse_data", batch_size=1, shift="normal", samples=100)
    df = df[(df["fold"]=="ind_val")&(df["Dataset"]=="OfficeHome")]
    print(df)
    g = sns.FacetGrid(df, col="feature_name", margin_titles=True, sharey=False, sharex=False, col_wrap=3)
    g.map_dataframe(sns.kdeplot, x="feature", hue="correct_prediction", common_norm=False)
    plt.savefig("figures/officehome_ind_correctness_by_ood_feature.png", dpi=300, bbox_inches='tight')
    plt.show()
