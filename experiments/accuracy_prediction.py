import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

from sklearn.metrics import mean_absolute_error
from matplotlib.patches import Patch

from components import OODDetector
from experiments.runtime_classification import get_all_ood_detector_data
from utils import *


def test_generalization_gap_estimation(batch_size):
    df_raw = load_all(batch_size, shift="")
    df = get_all_ood_detector_data(batch_size, filter_organic=False, filter_best=True)
    df_synth = df[df["Shift Intensity"]!="Organic"]
    df_synth.replace(SHIFT_PRINT_LUT, inplace=True)
    print(df.columns)

    acc_by_dataset_and_shift = df_raw.groupby(["Dataset", "fold"])["correct_prediction"].mean().reset_index()
    print(df["Dataset"].unique())
    ood_accs = df.groupby(["Dataset", "OoD Test Fold"])["tpr"].mean().reset_index()
    ind_accs = df.groupby(["Dataset", "InD Test Fold"])["tnr"].mean().reset_index()

    ind_accs["tnr"]=1-ind_accs["tnr"]
    ind_accs.rename(columns={"InD Test Fold":"fold", "tnr":"Detection Rate"}, inplace=True)
    ood_accs.rename(columns={"OoD Test Fold":"fold", "tpr":"Detection Rate"}, inplace=True)

    merged = pd.concat([ood_accs, ind_accs], ignore_index=True)
    merged = merged.merge(acc_by_dataset_and_shift, on=["Dataset", "fold"], how="left")
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
    print(merged.groupby(["Dataset", "Shift"])[["Generalization Gap"]].mean())
    g = sns.FacetGrid(merged, col="Dataset", col_wrap=3)
    g.map_dataframe(sns.scatterplot, x="Detection Rate", y="Generalization Gap", hue="Shift", alpha=0.7, edgecolor=None)

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
            print(f"{dataset:<15} {shift:<20} {mae:>10.4f} {baseline:>10.4f}")
            gam_data.append({"Dataset":dataset, "Shift":shift, "mae":mae, "baseline mae": baseline,  "x":np.linspace(0,1,2), "y":model.predict(np.linspace(0,1,2).reshape(-1,1))})
    gam_df = pd.DataFrame(gam_data)

    print(gam_df.groupby(["Dataset"])[["mae", "baseline mae"]].mean()) # print simple evaluation

    def plot_gam_fits(data, color=None, **kwargs):
        dataset = data["Dataset"].unique()[0]
        fit_to_plot = gam_df[(gam_df["Shift"]=="Organic")&(gam_df["Dataset"]==dataset)]
        plt.plot(fit_to_plot["x"].values[0], fit_to_plot["y"].values[0], color="black", linestyle="--", label="GAM Fit (Organic)")

    g.map_dataframe(plot_gam_fits)
    for ax in g.axes.flatten():
        ax.set_ylim(0.1,-1)
        ax.set_xlim(0,1)
    plt.savefig("figures/da_vs_generalization.pdf")
    plt.show()

            # print(model.summary())

def get_acc_prediction_results(batch_size, model="resnet"):
    df = get_all_ood_detector_data(batch_size, filter_organic=False, filter_best=False)
    df = df[df["OoD==f(x)=y"] == False]  # only OOD performance
    df = df[df["Model"]==model]
    print(df)
    assert df["Model"].unique()[0]==model
    df_synth = df[df["Shift Intensity"]!="Organic"]
    df_synth.replace(SHIFT_PRINT_LUT, inplace=True)
    unique_shifts  = df_synth["Shift"].unique().tolist()
    max_intensity = df_synth["Shift Intensity"].max()
    df_raw = load_all(batch_size, shift="")

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

    for dataset in DATASETS:
        model_data = []
        for feature_name in merged["feature_name"].unique():
            for_dataset = merged[(merged["Dataset"]==dataset)&(merged["feature_name"]==feature_name)]
            if for_dataset.empty:
                continue
            raw_data = load_data(dataset_name=dataset, feature_name=DSD_LUT[feature_name], batch_size=batch_size,
                                 shift="", model=model)
            if raw_data.empty:
                continue
            for shift in list(for_dataset["Shift"].unique())+["all"]:
                if shift=="adv" or shift=="ind":
                    continue
                train = for_dataset[(for_dataset["Shift"]!=shift)&(for_dataset["Shift"]!="FGSM")]
                # model = LinearGAM(constraints="monotonic_dec")
                reg_model = LinearRegression()

                reg_model.fit(train["Detection Rate"].values.reshape(-1, 1), train["Generalization Gap"].values.reshape(-1, 1))
                # print(raw_data.head(10))

                if shift=="Organic":
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
                        test_ood = test[test["ood"]==True]

                    test_ind = test_organic[test_organic["fold"] == "ind_test"]
                    assert not test_ind.empty

                    for intensity in test_ood["shift_intensity"].unique():
                        print(intensity)
                        test_ood_filt = test_ood[test_ood["shift_intensity"]==intensity]
                        if test_ood.empty:
                            continue
                        ood_detector = OODDetector(calib_ood, "val_optimal")
                        ood_dr = test_ood.apply(lambda row: ood_detector.predict(row), axis=1).mean()
                        ind_dr =  test_ind.apply(lambda row: ood_detector.predict(row), axis=1).mean()
                        ind_acc = test_organic[test_organic["fold"]=="ind_val"]["correct_prediction"].mean()
                        test_ood_filt["Generalization Gap"] = test_ood_filt["correct_prediction"]-ind_acc
                        test_ind["Generalization Gap"] = test_ind["correct_prediction"]-ind_acc


                        for proportion in np.linspace(0, 1, 11):
                            detection_rate = proportion * ood_dr + (1 - proportion) * ind_dr

                            gap = (proportion*test_ood_filt["Generalization Gap"].mean() + (1-proportion)*test_ind["Generalization Gap"].mean())
                            pred = reg_model.predict(detection_rate.reshape(1, -1))
                            mae = np.abs(gap - pred)[0][0]
                            baseline =np.abs(gap - 0)
                            model_data.append({"Dataset":dataset, "feature_name":feature_name, "Shift":shift,
                                               "intensity":intensity, "proportion":proportion,
                                             "mae":mae, "naive baseline mae": baseline, "pred": pred[0][0], "gap": gap,
                                               "detection_rate": detection_rate, "m":reg_model.coef_[0][0], "b":reg_model.intercept_[0]})

        model_df = pd.DataFrame(model_data)
        if model_df.empty:
            continue
        model_df.to_csv(f"data/{model}/ood_detector_data/{dataset}_acc_prediction_results.csv", index=False)
        print(model_df.groupby(["Dataset", "feature_name", "Shift"])[["mae", "naive baseline mae"]].mean())

        # print(gam.summary())


def get_all_pre_data():
    all_data = []
    for model, dataset in itertools.product(MODELS, DATASETS):
        try:
            df = pd.read_csv(f"data/{model}/pra_data/{dataset}_pre_results.csv")
            df["Dataset"] = dataset
            all_data.append(df)
        except FileNotFoundError:
            print(f"No data for model {model} dataset {dataset} ")
            continue
        print(df.head(3))
    all_df = pd.concat(all_data, ignore_index=True)
    print(all_df.groupby(["Dataset", "Model"])["Accuracy Error"].mean())
    return all_df



def acc_prediction_table():
    dfs = []
    for model, dataset  in itertools.product(MODELS, DATASETS):
        try:
            df = pd.read_csv(f"data/{model}/ood_detector_data/{dataset}_acc_prediction_results.csv")
            dfs.append(df)
            df["Model"]=model
        except FileNotFoundError:
            print(f"No data for model {model}")
            continue
        except pd.errors.EmptyDataError:
            print(f"Empty data for model {model} dataset {dataset}")
            continue
    df = pd.concat(dfs, ignore_index=True)

    print(df.groupby(["Dataset", "Model", "feature_name"])[["mae", "naive baseline mae"]].mean())

def get_all_acc_prediction_results():
    dfs = []
    for model, dataset in itertools.product(MODELS, DATASETS):
        try:
            df = pd.read_csv(f"data/{model}/ood_detector_data/{dataset}_acc_prediction_results.csv")
        except FileNotFoundError:
            print(f"No data for model {model}")
            continue
        df["Model"] = model
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df

def error_heatmap():
    df = get_all_acc_prediction_results()
    ood_detector_data = []
    for model in MODELS:
        for_model = get_all_ood_detector_data(batch_size=1, filter_organic=True, filter_best=True)
        if for_model.empty:
            continue
        for_model = for_model[for_model["Model"]==model]
        ood_detector_data.append(for_model)
    ood_detector_data = pd.concat(ood_detector_data, ignore_index=True)

    print(ood_detector_data["Dataset"].unique())
    pre_data = get_all_pre_data()
    pre_data = pre_data.groupby(["Dataset", "dsd", "val_set", "rate", "Model"])["Accuracy Error"].mean().reset_index()
    pre_data["intensity"] = pre_data["val_set"].apply(lambda x: x.split("_")[-1] if "_" in x else "OoD")
    pre_data.rename(columns={"dsd":"feature_name", "Accuracy Error":"mae", "rate":"proportion"}, inplace=True)
    pre_data = pre_data.groupby(["Dataset", "feature_name", "proportion", "intensity", "Model"])[["mae"]].mean().reset_index()

    ood_detector_data = ood_detector_data[ood_detector_data["OoD==f(x)=y"]==False]
    best_detectors_per_dataset = ood_detector_data.groupby(["Dataset", "feature_name", "Model"])["ba"].mean().reset_index()
    print(df["Dataset"].unique())

    df = df.merge(best_detectors_per_dataset, on=["Model", "Dataset", "feature_name"], how="inner", suffixes=("", "_detector"))
    df = df.round(2)
    print(df["Dataset"].unique())

    pre_data = pre_data.round(2)
    df["intensity"] = df["intensity"].apply(lambda x: str(round(float(x), 2)) if x!="OoD" else x)
    # df["mae"] = df["mae"].apply(lambda x: float(x.strip("[]")))
    # Mean per (Dataset, feature_name, proportion, intensity)

    df_grouped = (
        df.groupby(["Dataset", "Model", "feature_name", "proportion", "intensity"])[["mae", "naive baseline mae"]]
        .mean()
        .reset_index()
    )

    g = sns.FacetGrid(df_grouped, col="Dataset", sharex=True, sharey=True, margin_titles=False, col_wrap=3)
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    from matplotlib.colors import hsv_to_rgb
    def saturation_only_cmap(hue, value=0.75, n=256):
        """
        Colormap that holds brightness (V) constant and varies only saturation (S):
        t=0 -> fully saturated color, t=1 -> gray with same brightness.
        """
        t = np.linspace(0, 1, n)
        sats = 1.0 - t  # high saturation for low error
        hsv = np.stack([np.full(n, hue), sats, np.full(n, value)], axis=1)
        rgb = hsv_to_rgb(hsv)
        return LinearSegmentedColormap.from_list(f"sats_only_{hue:.2f}", rgb)

    palette = {
        "model": saturation_only_cmap(hue=0.33, value=0.75),  # green
        "baseline": saturation_only_cmap(hue=0.00, value=0.75),  # red
        "pre": saturation_only_cmap(hue=0.60, value=0.75),  # blue
    }

    def sort_mixed_index(df):
        """Sort numeric-like intensities ascending, keep 'OoD' or non-numeric last."""

        def parse_key(x):
            try:
                return (0, float(x))  # numeric part
            except ValueError:
                return (1, x)  # non-numeric last

        return df.sort_index(key=lambda idx: [parse_key(x) for x in idx])

    def draw_min_config_heatmap(data, **kwargs):
        ax = plt.gca()
        dataset = data["Dataset"].unique()[0]

        model = data["Model"].unique()[0]
        pre_data_filt = pre_data[(pre_data["Dataset"]==dataset)&(pre_data["Model"]==model)]
        # split by source


        # pivots (align grids)
        model_p = data.pivot(index="intensity", columns="proportion", values="mae")
        base_p = data.pivot(index="intensity", columns="proportion", values="naive baseline mae")
        pre_p = pre_data_filt.pivot(index="intensity", columns="proportion", values="mae")

        # union index/columns
        all_idx = model_p.index.union(base_p.index if base_p is not None else model_p.index).union(
            pre_p.index if pre_p is not None else model_p.index)
        all_col = model_p.columns.union(base_p.columns if base_p is not None else model_p.columns).union(
            pre_p.columns if pre_p is not None else model_p.columns)
        model_p = model_p.reindex(index=all_idx, columns=all_col)
        base_p = base_p.reindex(index=all_idx, columns=all_col)
        pre_p = pre_p.reindex(index=all_idx, columns=all_col)

        # 3D stack to get mins/argmins (order: model, baseline, pre)
        stack = np.stack([
            model_p.values,
            base_p.values,
            pre_p.values
        ], axis=0)
        # nan-aware min/argmin
        min_vals = np.nanmin(stack, axis=0)
        # for argmin with nans: replace nans with +inf so they won't win
        stack_for_arg = np.where(np.isnan(stack), np.inf, stack)
        winners = np.argmin(stack_for_arg, axis=0)  # 0=model,1=baseline,2=pre
        # cells where all nan -> mask out entirely
        all_nan_mask = np.all(np.isnan(stack), axis=0)

        # Shared normalization over min values (exclude nans)
        valid_min = min_vals[~np.isnan(min_vals)]
        if valid_min.size == 0:
            return
        vmin = float(np.nanmin(valid_min))
        vmax = float(np.nanmax(valid_min))
        norm = PowerNorm(gamma=0.6, vmin=vmin, vmax=vmax)  # lower gamma → more separation at low end

        # masks per config: True means "hide"; we invert below
        mask_model = ~(winners == 0) | all_nan_mask
        mask_baseline = ~(winners == 1) | all_nan_mask
        mask_pre = ~(winners == 2) | all_nan_mask

        # We'll plot the same numeric matrix (min_vals) three times, each with its own mask & cmap
        # Use consistent DataFrame for seaborn
        min_df = pd.DataFrame(min_vals, index=all_idx, columns=all_col)

        # PRE (Blues)
        sns.heatmap(
            min_df, cmap=palette["pre"], norm=norm,
            mask=mask_pre, cbar=False, ax=ax, **kwargs
        )
        # BASELINE (Reds)
        sns.heatmap(
            min_df, cmap=palette["baseline"], norm=norm,
            mask=mask_baseline, cbar=False, ax=ax, **kwargs
        )
        # MODEL (Greens)
        hm = sns.heatmap(
            min_df, cmap=palette["model"], norm=norm,
            mask=mask_model, cbar=False, ax=ax, **kwargs
        )

        # --- replace the Greys colorbar section with this ---
        divider = make_axes_locatable(ax)

        # three slim, stacked colorbars (inner→outer)
        cax_model = divider.append_axes("right", size="3%", pad=0.015)
        cax_base = divider.append_axes("right", size="3%", pad=0.030)
        cax_pre = divider.append_axes("right", size="3%", pad=0.045)

        sm_model = plt.cm.ScalarMappable(norm=norm, cmap=palette["model"])
        sm_base = plt.cm.ScalarMappable(norm=norm, cmap=palette["baseline"])
        sm_pre = plt.cm.ScalarMappable(norm=norm, cmap=palette["pre"])
        for sm in (sm_model, sm_base, sm_pre):
            sm.set_array([])

        cb_model = ax.figure.colorbar(sm_model, cax=cax_model)
        cb_base = ax.figure.colorbar(sm_base, cax=cax_base)
        cb_pre = ax.figure.colorbar(sm_pre, cax=cax_pre)

        # aesthetics: show ticks only on the outermost (blue/PRE) bar
        for cb in (cb_model, cb_base):
            cb.ax.set_yticklabels([])
            cb.ax.tick_params(length=0)
        for cb in (cb_model, cb_base, cb_pre):
            cb.outline.set_visible(False)

        cb_pre.ax.set_ylabel("Min error (low = saturated)", rotation=270, labelpad=10)

        # Legend: which color == which configuration
        handles = [
            Patch(color=palette["model"](0.0), label="Model"),  # t=0 → saturated
            Patch(color=palette["baseline"](0.0), label="Baseline"),
            Patch(color=palette["pre"](0.0), label="PRE"),
        ]
        if not getattr(ax, "_legend_added", False):
            ax.legend(handles=handles, frameon=False, loc="upper right", fontsize="small", title="Winner")
            ax._legend_added = True

        # Tidy axes
        ax.set_xlabel("Proportion OoD")
        ax.set_ylabel("Shift Intensity")
        ax.invert_yaxis()

    g.map_dataframe(draw_min_config_heatmap)
    g.set_axis_labels("Proportion OoD", "Shift Intensity")
    plt.tight_layout()
    plt.savefig("figures/accuracy_prediction_error_heatmap.pdf")
    plt.show()

def error_per_accuracy():
    data = get_all_acc_prediction_results()

    # Compute mean MAE per (Dataset, feature_name, Model)
    mae_means = (
        data.groupby(["Dataset", "feature_name", "Model"], as_index=False)["mae"]
        .mean()
    )

    # Find (feature_name, Model) combination with lowest MAE per Dataset
    best_combinations = (
        mae_means.loc[mae_means.groupby("Dataset")["mae"].idxmin(),
        ["Dataset", "feature_name", "Model"]]
    )

    # Filter data to keep only the best combinations per dataset
    filtered_data = data.merge(best_combinations, on=["Dataset", "feature_name", "Model"])
    # gam_
    fits = pd.read_csv("gam_fits.csv")
    filtered_data["relative_error"] = data["pred"] - data["gap"]
    filtered_data = filtered_data[filtered_data["proportion"]==1]

    def scatter_with_sliding_mean(data, x, y, window=31, **kwargs):
        """
        Overlay a centered rolling (sliding-window) mean on top of scatter points.
        `window` is the number of points in the rolling window (use an odd number).
        """
        ax = plt.gca()

        # scatter
        sns.scatterplot(data=data, x=x, y=y, alpha=0.5, ax=ax)

        # rolling mean (by count, not by x-width)
        df = data[[x, y]].dropna().sort_values(x)
        if len(df) >= window:
            m = df[y].rolling(window=window, center=True).mean()
            ax.plot(df[x], m, linewidth=2, linestyle='-', color='red', label='Sliding Mean')
        else:
            # if too few points for rolling mean, just connect them
            ax.plot(df[x], df[y], linewidth=2, linestyle='-', color='red', label='Line')

    g = sns.FacetGrid(filtered_data, col="Dataset", sharex=False, sharey=False, col_wrap=3)

    # scatter + sliding window average over the scatter points
    g.map_dataframe(scatter_with_sliding_mean, x="gap", y="mae", window=31)

    # keep the baseline as a dashed line
    g.map_dataframe(sns.lineplot, x="gap", y="naive baseline mae", color="black", linestyle="--")

    g.set_axis_labels("Generalization Gap", "Accuracy Prediction Error (MAE)")
    g.set_titles(col_template="{col_name}")
    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.set_ylim(1e-3,1)
    plt.savefig("figures/accuracy_prediction_error_per_gap.pdf", bbox_inches="tight")
    plt.show()

