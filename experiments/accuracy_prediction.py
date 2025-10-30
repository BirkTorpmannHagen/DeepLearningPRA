import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pygam import LinearGAM
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

from sklearn.metrics import mean_absolute_error
from matplotlib.patches import Patch

from components import OODDetector
from experiments.runtime_classification import get_all_ood_detector_data
from utils import SHIFT_PRINT_LUT, load_all, DSD_PRINT_LUT, SYNTHETIC_SHIFTS, DATASETS, load_pra_df, DSD_LUT, SHIFT_LUT


def test_generalization_gap_estimation(batch_size):
    df_raw = load_all(batch_size, shift="", model="resnet")
    print(df_raw.groupby(["Dataset", "fold"])["correct_prediction"].mean().reset_index())

    df = get_all_ood_detector_data(batch_size, filter_thresholding_method=True, filter_ood_correctness=False,
                                   filter_correctness_calibration=True, filter_organic=False, filter_best=True, )
    df = df[df["OoD==f(x)=y"] == False]  # only OOD performance

    df_synth = df[df["Shift Intensity"]!="Organic"]
    df_synth.replace(SHIFT_PRINT_LUT, inplace=True)
    unique_shifts  = df_synth["Shift"].unique().tolist()

    acc_by_dataset_and_shift = df_raw.groupby(["Dataset", "fold"])["correct_prediction"].mean().reset_index()

    ood_accs = df.groupby(["Dataset", "OoD Test Fold", "OoD==f(x)=y"])["tpr"].mean().reset_index()
    ind_accs = df.groupby(["Dataset", "InD Test Fold", "OoD==f(x)=y"])["tnr"].mean().reset_index()
    ind_accs["tnr"]=1-ind_accs["tnr"]
    ind_accs.rename(columns={"InD Test Fold":"fold", "tnr":"Detection Rate"}, inplace=True)
    ood_accs.rename(columns={"OoD Test Fold":"fold", "tpr":"Detection Rate"}, inplace=True)

    merged = pd.concat([ood_accs, ind_accs], ignore_index=True)
    merged = merged.merge(acc_by_dataset_and_shift, on=["Dataset", "fold"])
    merged["Shift"] = merged["fold"].apply(lambda x: x.split("_")[0] if "_" in x else "Organic")
    merged["Organic"] = merged["Shift"].apply(lambda x: "Synthetic" if x in SYNTHETIC_SHIFTS else "Organic")
    acc = merged.groupby(["Dataset", "fold"], as_index=False)["correct_prediction"].mean()

    # pull the per-dataset ind_val baseline
    ind = (acc.loc[acc["fold"] == "ind_val", ["Dataset", "correct_prediction"]]
           .rename(columns={"correct_prediction": "ind_val_acc"}))
    # join baseline back to every shift of the same dataset
    acc = acc.merge(ind, on="Dataset", how="left")
    # absolute and relative differences vs ind_val
    acc["Generalization Gap"] = acc["correct_prediction"] - acc["ind_val_acc"]
    acc["Accuracy"] = acc["correct_prediction"]
    # acc["Generalization Gap"] = acc["acc_diff"] / acc["ind_val_acc"]  # e.g., 0.10 == +10%
    # acc["Generalization Gap"] = - acc["Generalization Gap"] * 100  # convert to percentage
    merged = merged.merge(acc, on=["Dataset", "fold"], how="left")

    g = sns.FacetGrid(merged, col="Dataset", col_wrap=3)
    g.map_dataframe(sns.scatterplot, x="Detection Rate", y="Generalization Gap", hue="Shift", hue_order=["Organic", "Synthetic"], alpha=0.7, edgecolor=None)

    merged["shift"] = merged.replace(SHIFT_PRINT_LUT, inplace=True)
    gam_data = []
    for dataset in DATASETS:
        for_dataset = merged[merged["Dataset"]==dataset]
        for shift in for_dataset["Shift"].unique():

            train = for_dataset[(for_dataset["Shift"]!=shift)&(for_dataset["Shift"]!="FGSM")]

            test = for_dataset[for_dataset["Shift"]==shift]

            # model = LinearGAM(constraints="monotonic_dec")
            model = LinearRegression()
            model.fit(train["Detection Rate"].values.reshape(-1,1), train["Generalization Gap"].values.reshape(-1,1))
            mae = mean_absolute_error(test["Generalization Gap"].values.reshape(-1,1), model.predict(test["Detection Rate"].values.reshape(-1,1)))
            baseline = mean_absolute_error(test["Generalization Gap"], [0]*len(test))
            # score = model.score(test["Detection Rate"], test["Generalization Gap"])
            print(f"{dataset:<15} {shift:<20} {mae:>10.4f} {baseline:>10.4f}k")
            gam_data.append({"Dataset":dataset, "Shift":shift, "mae":mae, "baseline mae": baseline,  "x":np.linspace(0,1,2), "y":model.predict(np.linspace(0,1,2).reshape(-1,1))})
    gam_df = pd.DataFrame(gam_data)

    print(gam_df.groupby(["Dataset"])[["mae", "baseline mae"]].mean()) # print simple evaluation

    def plot_gam_fits(data, color=None, **kwargs):
        dataset = data["Dataset"].unique()[0]
        fit_to_plot = gam_df[(gam_df["Shift"]=="Organic")&(gam_df["Dataset"]==dataset)]
        plt.plot(fit_to_plot["x"].values[0], fit_to_plot["y"].values[0], color="black", linestyle="--", label="GAM Fit (Organic)")

    g.map_dataframe(plot_gam_fits)
    plt.show()

            # print(model.summary())

def get_acc_prediction_results(batch_size, model="resnet"):
    df = get_all_ood_detector_data(batch_size, filter_thresholding_method=True, filter_ood_correctness=False,
                                   filter_correctness_calibration=True, filter_organic=False, filter_best=False, model=model)
    df = df[df["OoD==f(x)=y"] == False]  # only OOD performance

    df_synth = df[df["Shift Intensity"]!="Organic"]
    df_synth.replace(SHIFT_PRINT_LUT, inplace=True)
    unique_shifts  = df_synth["Shift"].unique().tolist()
    max_intensity = df_synth["Shift Intensity"].max()
    df_raw = load_all(batch_size, shift="", model=model)

    acc_by_dataset_and_shift = df_raw.groupby(["Dataset", "feature_name", "fold"])["correct_prediction"].mean().reset_index()
    acc_by_dataset_and_shift.replace(DSD_PRINT_LUT, inplace=True)

    ood_accs = df.groupby(["Dataset","feature_name", "OoD Test Fold", "OoD==f(x)=y"])["tpr"].mean().reset_index()
    ind_accs = df.groupby(["Dataset","feature_name", "InD Test Fold", "OoD==f(x)=y"])["tnr"].mean().reset_index()
    ind_accs["tnr"]=1-ind_accs["tnr"]
    ind_accs.rename(columns={"InD Test Fold":"fold", "tnr":"Detection Rate"}, inplace=True)
    ood_accs.rename(columns={"OoD Test Fold":"fold", "tpr":"Detection Rate"}, inplace=True)

    merged = pd.concat([ood_accs, ind_accs], ignore_index=True)

    merged = merged.merge(acc_by_dataset_and_shift, on=["Dataset", "feature_name", "fold"])

    merged["Shift"] = merged["fold"].apply(lambda x: x.split("_")[0] if "_" in x else "Organic")
    merged["Organic"] = merged["Shift"].apply(lambda x: "Synthetic" if x in SYNTHETIC_SHIFTS else "Organic")

    acc = merged.groupby(["Dataset","feature_name", "fold"], as_index=False)["correct_prediction"].mean()
    # pull the per-dataset ind_val baseline
    ind = (acc.loc[acc["fold"] == "ind_val", ["Dataset","feature_name", "correct_prediction"]]
           .rename(columns={"correct_prediction": "ind_val_acc"}))

    # join baseline back to every shift of the same dataset
    acc = acc.merge(ind, on=["Dataset", "feature_name"], how="left")
    # absolute and relative differences vs ind_val
    acc["Generalization Gap"] = acc["correct_prediction"] - acc["ind_val_acc"]
    acc["Accuracy"] = acc["correct_prediction"]
    merged = merged.merge(acc, on=["Dataset","feature_name", "fold"], how="left")

    g = sns.FacetGrid(merged, col="Dataset", row="feature_name", sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x="Detection Rate", y="Generalization Gap", hue="Organic", hue_order=["Organic", "Synthetic"], alpha=0.7, edgecolor=None)

    merged["shift"] = merged.replace(SHIFT_PRINT_LUT, inplace=True)
    model_data = []

    for dataset in DATASETS:
        for feature_name in merged["feature_name"].unique():
            for_dataset = merged[(merged["Dataset"]==dataset)&(merged["feature_name"]==feature_name)]
            if for_dataset.empty:
                continue
            raw_data = load_classifier_data(dataset_name=dataset, feature_name=DSD_LUT[feature_name], batch_size=batch_size,
                                            shift="", model=model)
            for shift in list(for_dataset["Shift"].unique())+["all"]:
                if shift=="adv" or shift=="ind":
                    continue
                train = for_dataset[(for_dataset["Shift"]!=shift)&(for_dataset["Shift"]!="FGSM")]
                # model = LinearGAM(constraints="monotonic_dec")
                reg_model = LinearRegression()

                reg_model.fit(train["Detection Rate"].values.reshape(-1, 1), train["Generalization Gap"].values.reshape(-1, 1))
                # print(raw_data.head(10))

                if shift=="Organic":
                    print(raw_data.head(10))
                    test = raw_data[raw_data["Organic"]==True]
                elif shift=="all":
                    test = raw_data.sample(len(raw_data[raw_data["Organic"]==True]))
                else:
                    test = raw_data[raw_data["shift"]==SHIFT_LUT[shift]]

                test_organic = raw_data[raw_data["Organic"]==True]


                test = test[test["fold"]!="train"]
                test_organic = test_organic[test_organic["fold"]!="train"]

                for ood_folds in test_organic["fold"].unique():
                    if ood_folds in ["train", "ind_val", "ind_test"]:
                        continue

                    calib_ood = test_organic[(test_organic["fold"]==ood_folds)|(test_organic["fold"]=="ind_val")]
                    if shift=="Organic":
                        test_ood = test[(test["fold"]!=ood_folds)&(test["ood"]==True)] #makes sure no overlap when normal
                    else:

                        test_ood = test[(test["shift_intensity"]==max_intensity)&(test["ood"]==True)]


                    test_ind = test_organic[test_organic["fold"]=="ind_test"]
                    assert not test_ind.empty
                    ood_detector = OODDetector(calib_ood, "val_optimal")
                    ood_dr = test_ood.apply(lambda row: ood_detector.predict(row), axis=1).mean()
                    ind_dr =  test_ind.apply(lambda row: ood_detector.predict(row), axis=1).mean()
                    ind_acc = test_organic[test_organic["fold"]=="ind_val"]["correct_prediction"].mean()
                    test_ood["Generalization Gap"] = test_ood["correct_prediction"]-ind_acc
                    test_ind["Generalization Gap"] = test_ind["correct_prediction"]-ind_acc

                    for proportion in np.linspace(0,1,11):

                        detection_rate = proportion*ood_dr + (1-proportion)*ind_dr

                        gap = (proportion*test_ood["Generalization Gap"].mean() + (1-proportion)*test_ind["Generalization Gap"].mean())

                        mae = np.abs(gap - reg_model.predict(detection_rate.reshape(1, -1)))[0][0]
                        baseline =np.abs(gap - 0)
                        model_data.append({"Dataset":dataset, "feature_name":feature_name, "Shift":shift,
                                         "mae":mae, "naive baseline mae": baseline})

    model_df = pd.DataFrame(model_data)
    model_df.to_csv(f"{model}_ood_detector_data/acc_prediction_results.csv", index=False)
    print(model_df.groupby(["Dataset", "feature_name", "Shift"])[["mae", "naive baseline mae"]].mean())

            # print(gam.summary())

def error_heatmap():
    df = pd.concat([pd.read_csv(f"acc_estimation_data/{dataset}_acc_prediction_gam_results.csv") for dataset in DATASETS])
    ood_detector_data = get_all_ood_detector_data(batch_size=1, filter_thresholding_method=True, filter_ood_correctness=False,
                                      filter_correctness_calibration=True, filter_organic=True, filter_best=True)
    ood_detector_data = ood_detector_data[ood_detector_data["OoD==f(x)=y"]==False]
    best_detectors_per_dataset = ood_detector_data.groupby(["Dataset", "feature_name"])["ba"].mean().reset_index()
    df = df.merge(best_detectors_per_dataset, on=["Dataset", "feature_name"], how="inner", suffixes=("", "_detector"))
    df = df.round(2)
    df["intensity"] = df["intensity"].apply(lambda x: str(round(float(x), 2)) if x!="OoD" else x)
    # df["mae"] = df["mae"].apply(lambda x: float(x.strip("[]")))
    # Mean per (Dataset, feature_name, proportion, intensity)

    df_grouped = (
        df.groupby(["Dataset", "feature_name", "proportion", "intensity"])[["mae", "naive baseline mae"]]
        .mean()
        .reset_index()
    )

    g = sns.FacetGrid(df_grouped, col="Dataset", sharex=True, sharey=True, margin_titles=False, col_wrap=2)

    def heatmap(data, **kwargs):
        ax = plt.gca()

        mae_p = data.pivot(index="intensity", columns="proportion", values="mae").sort_index().sort_index(axis=1)
        base_p = data.pivot(index="intensity", columns="proportion", values="naive baseline mae").reindex_like(mae_p)

        baseline_better = base_p < mae_p
        vmin = float(mae_p.min().min())
        vmax = float(mae_p.max().max())
        norm = Normalize(vmin=vmin, vmax=vmax)

        # draw both heatmaps WITHOUT colorbars
        hm_mako = sns.heatmap(
            mae_p, cmap="mako", norm=norm,
            mask=baseline_better, cbar=False, ax=ax, **kwargs
        )
        hm_magma = sns.heatmap(
            mae_p, cmap="magma", norm=norm,
            mask=~baseline_better, cbar=False, ax=ax, **kwargs
        )

        # make two dedicated colorbar axes, very close together
        divider = make_axes_locatable(ax)
        # append the INNER one first (this will sit closest to the main axes)
        cax_magma = divider.append_axes("right", size="5%", pad=0.015)
        # then the OUTER one (sits to the right of the first)
        cax_mako = divider.append_axes("right", size="5%", pad=0.030)

        # create independent mappables for each cmap
        sm_mako = sns.color_palette("mako", as_cmap=True)
        sm_magma = sns.color_palette("magma", as_cmap=True)

        mappable_mako = plt.cm.ScalarMappable(norm=norm, cmap=sm_mako)
        mappable_magma = plt.cm.ScalarMappable(norm=norm, cmap=sm_magma)
        mappable_mako.set_array([])
        mappable_magma.set_array([])

        # draw the colorbars
        cb_magma = ax.figure.colorbar(mappable_magma, cax=cax_magma)
        cb_mako = ax.figure.colorbar(mappable_mako, cax=cax_mako)

        # hide ticks/numbers for magma; keep them for mako
        cb_magma.ax.set_yticklabels([])
        cb_magma.ax.tick_params(length=0)

        # optional: thin outlines off
        for c in (cb_mako, cb_magma):
            c.outline.set_visible(False)

        # legend once per facet
        handles = [Patch(label="Model ≤ Baseline (mako)"), Patch(label="Baseline < Model (magma)")]
        if not getattr(ax, "_legend_added", False):
            ax.legend(handles, ["Model ≤ Baseline (mako)", "Baseline < Model (magma)"],
                      frameon=False, loc="upper right", fontsize="small")
            ax._legend_added = True

    g.map_dataframe(heatmap)
    g.set_axis_labels("Proportion OoD", "Shift Intensity")
    plt.tight_layout()
    plt.show()

def error_per_accuracy():
    data = pd.concat([pd.read_csv(f"acc_estimation_data/{dataset}_acc_prediction_gam_results.csv") for dataset in DATASETS])
    fits = pd.read_csv("gam_fits.csv")
    data = data[data["proportion"]==1]
    print(data.columns)
    data["relative_error"] = data["prediction"] - data["gap"]

    g = sns.FacetGrid(fits, col="Dataset", row="feature_name", sharex=False, sharey=False)
    g.map_dataframe(sns.lineplot, x="x", y="y", color="black", linestyle="--")
    # g.map_dataframe(sns.scatterplot, x="gap", y="relative_error", hue="intensity")

    # g.map_dataframe(sns.lineplot, x="gap", y="gap", color="black", linestyle="--")
    plt.show()

