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

def get_merged(batch_size=1, pretrain=True, filter_best=True):
    df = get_all_ood_detector_data(batch_size, filter_organic=False, filter_best=filter_best, pretrain=pretrain)

    df_synth = df[df["Shift Intensity"]!="Organic"]
    df_synth.replace(SHIFT_PRINT_LUT, inplace=True)

    df_raw = load_all(batch_size, shift="", pretrain=pretrain)

    acc_by_dataset_and_shift = df_raw.groupby(["Dataset", "Model", "fold"])["correct_prediction"].mean().reset_index()
    acc_by_dataset_and_shift.replace(DSD_PRINT_LUT, inplace=True)
    df.rename(columns={"Fold":"fold"}, inplace=True)
    merged = df.merge(acc_by_dataset_and_shift, on=["Dataset", "fold", "Model"], how="left")
    merged["Shift"] = merged["fold"].apply(lambda x: x.split("_")[0] if "_" in x else "Organic")
    merged["Organic"] = merged["Shift"].apply(lambda x: "Synthetic" if x in SYNTHETIC_SHIFTS else "Organic")
    acc = merged.groupby(["Dataset", "fold", "Model"], as_index=False)["correct_prediction"].mean()

    # pull the per-dataset ind_val baseline
    ind = (acc.loc[acc["fold"] == "ind_val", ["Dataset", "correct_prediction"]]
           .rename(columns={"correct_prediction": "ind_val_acc"}))

    # join baseline back to every shift of the same dataset
    acc = acc.merge(ind, on="Dataset", how="left")

    # absolute and relative differences vs ind_val
    acc["Generalization Gap"] = acc["correct_prediction"] - acc["ind_val_acc"]
    acc["Accuracy"] = acc["correct_prediction"]
    merged = merged.merge(acc, on=["Dataset", "fold", "Model"], how="left")
    return merged

def test_generalization_gap_estimation(batch_size, pretrain=False):
    """Figure 1: DR vs generalization gap, one panel per dataset, single row.
    Colorblind-safe palette, explicit shift legend, marker outlines."""
    merged = get_merged(batch_size, pretrain=pretrain)
    merged.replace(SHIFT_PRINT_LUT, inplace=True)
    # Drop pseudo-shifts that aren't covariate-shift augmentations
    merged = merged[~merged["Shift"].isin(["adv", "ind"])]
    hue_order = sorted([s for s in merged["Shift"].unique() if pd.notna(s)])
    palette = sns.color_palette("colorblind", n_colors=len(hue_order))

    gam_data = []
    for dataset in DATASETS:
        for_dataset = merged[merged["Dataset"] == dataset]
        if for_dataset.empty:
            continue
        for shift in for_dataset["Shift"].unique():
            train = for_dataset[(for_dataset["Shift"] != shift) & (for_dataset["Shift"] != "FGSM")]
            test = for_dataset[for_dataset["Shift"] == shift]
            if train.empty or test.empty:
                continue
            reg_model = LinearRegression()
            reg_model.fit(train["DR"].values.reshape(-1, 1), train["Generalization Gap"].values.reshape(-1, 1))
            mae = mean_absolute_error(test["Generalization Gap"].values.reshape(-1, 1),
                                      reg_model.predict(test["DR"].values.reshape(-1, 1)))
            baseline = mean_absolute_error(test["Generalization Gap"], [0] * len(test))
            gam_data.append({"Dataset": dataset, "Model": for_dataset["Model"].unique()[0],
                             "Shift": shift, "mae": mae, "baseline mae": baseline,
                             "x": np.linspace(0, 1, 2),
                             "y": reg_model.predict(np.linspace(0, 1, 2).reshape(-1, 1))})
    gam_df = pd.DataFrame(gam_data)

    g = sns.FacetGrid(merged, col="Dataset", col_order=DATASETS, col_wrap=3,
                      sharex=True, sharey=True, height=2.4, aspect=1.0)
    g.map_dataframe(sns.scatterplot, x="DR", y="Generalization Gap",
                    hue="Shift", hue_order=hue_order, palette=palette,
                    alpha=0.7, s=22, edgecolor="black", linewidth=0.3)
    g.set_titles(col_template="{col_name}")

    def plot_gam_fits(data, color=None, **kwargs):
        dataset = data["Dataset"].unique()[0]
        model = data["Model"].unique()[0]
        fit_to_plot = gam_df[(gam_df["Shift"] == "Organic") &
                             (gam_df["Dataset"] == dataset) & (gam_df["Model"] == model)]
        if not fit_to_plot.empty:
            plt.plot(fit_to_plot["x"].values[0], fit_to_plot["y"].values[0],
                     color="black", linestyle="--", linewidth=1.0, label="Linear Fit")

    g.map_dataframe(plot_gam_fits)
    for ax in g.axes.flatten():
        ax.set_ylim(0.1, -1)
        ax.set_xlim(0, 1)
    g.add_legend(title="Shift Type", bbox_to_anchor=(1.0, 0.5), loc="center left",
                 frameon=True, fontsize="x-small")
    plt.savefig("figures/da_vs_generalization.pdf", bbox_inches="tight")
    plt.show()


def get_acc_prediction_results(batch_size, pretrain=False):
    merged = get_merged(batch_size, pretrain=pretrain, filter_best=False)
    prefix = "data/pretrain" if pretrain else "data/nopretrain"
    g = sns.FacetGrid(merged, col="Dataset", row="feature_name", sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x="DR", y="Generalization Gap", hue="Organic", hue_order=["Organic", "Synthetic"], alpha=0.7, edgecolor=None)

    merged["shift"] = merged.replace(SHIFT_PRINT_LUT, inplace=True)
    for model in MODELS:
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
                    reg_model = LinearRegression()
                    reg_model.fit(train["DR"].values.reshape(-1, 1), train["Generalization Gap"].values.reshape(-1, 1))

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
            model_df.to_csv(f"{prefix}/{model}/ood_detector_data/{dataset}_acc_prediction_results.csv", index=False)
            print(model_df.groupby(["Dataset", "feature_name", "Shift"])[["mae", "naive baseline mae"]].mean())


def get_all_pre_data(pretrain=True):
    all_data = []
    prefix = "data/pretrain" if pretrain else "data/nopretrain"

    for model, dataset in itertools.product(MODELS, DATASETS):
        try:
            df = pd.read_csv(f"{prefix}/{model}/pra_data/{dataset}_pre_results.csv")
            df["Dataset"] = dataset
            all_data.append(df)
        except FileNotFoundError:
            print(f"No data for {prefix}/{model}/pra_data/{dataset}_pre_results.csv")
            continue
    all_df = pd.concat(all_data, ignore_index=True)
    return all_df



def acc_prediction_table(pretrain):
    dfs = []
    prefix = "data/pretrain" if pretrain else "data/nopretrain"
    for model, dataset  in itertools.product(MODELS, DATASETS):
        try:
            df = pd.read_csv(f"{prefix}/{model}/ood_detector_data/{dataset}_acc_prediction_results.csv")
            dfs.append(df)
            df["Model"]=model
        except FileNotFoundError:
            print(f"No data in {prefix}/{model}/ood_detector_data/{dataset}_acc_prediction_results.csv")
            continue
        except pd.errors.EmptyDataError:
            print(f"Empty data for model {model} dataset {dataset}")
            continue
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["Shift"]!="all")&(df["proportion"]==1)]
    meaned = df.groupby(["Dataset", "Model", "feature_name"])[["mae", "naive baseline mae"]].mean().reset_index()
    print(meaned.groupby(["Dataset",  "feature_name"])[["mae", "naive baseline mae"]].min())

def get_all_acc_prediction_results(pretrain=True):
    prefix = "data/pretrain" if pretrain else "data/nopretrain"
    dfs = []

    for model, dataset in itertools.product(MODELS, DATASETS):
        try:
            df = pd.read_csv(f"{prefix}/{model}/ood_detector_data/{dataset}_acc_prediction_results.csv")
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

    pre_data = get_all_pre_data(pretrain=True)

    pre_data = pre_data.groupby(["Dataset", "dsd", "val_set", "rate", "Model"])["Accuracy Error"].mean().reset_index()
    pre_data["intensity"] = pre_data["val_set"].apply(lambda x: x.split("_")[-1] if "_" in x else "OoD")
    pre_data.rename(columns={"dsd":"feature_name", "Accuracy Error":"mae", "rate":"proportion"}, inplace=True)
    pre_data = pre_data.groupby(["Dataset", "feature_name", "proportion", "intensity", "Model"])[["mae"]].mean().reset_index()

    scores = df.groupby(["Dataset", "feature_name", "Model"])["mae"].mean().reset_index()

    idx = scores.groupby("Dataset")["mae"].idxmin()
    best_configs = scores.loc[idx, ["Dataset", "Model", "feature_name"]]
    df = df.merge(
        best_configs,
        on=["Dataset", "Model", "feature_name"],
        how="inner"
    )
    df = df.round(2)

    pre_data = pre_data.round(2)
    pre_data = pre_data.groupby(["Dataset", "proportion", "intensity"])[["mae"]].mean().reset_index()
    df["intensity"] = df["intensity"].apply(lambda x: str(round(float(x), 2)) if x!="OoD" else x)

    df_grouped = (
        df.groupby(["Dataset", "Model", "feature_name", "proportion", "intensity"])[["mae", "naive baseline mae"]]
        .mean()
        .reset_index()
    )

    print(df_grouped.groupby(["Dataset", "Model", "feature_name"])[["mae", "naive baseline mae"]].mean())
    print(pre_data.groupby(["Dataset"])[["mae"]].mean())

    # Single-row layout: one panel per dataset
    g = sns.FacetGrid(df_grouped, col="Dataset", col_order=DATASETS,
                      sharex=True, sharey=False, margin_titles=False,
                      col_wrap=3, height=2.6, aspect=1.0)
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    def saturation_cmap_from_rgb(rgb_color, n=256):
        """Colormap from light-grey -> saturated colorblind-safe color (low error = saturated)."""
        t = np.linspace(0, 1, n)[:, None]
        base = np.array(rgb_color)[None, :]
        grey = np.array([0.85, 0.85, 0.85])[None, :]
        rgb = (1.0 - t) * base + t * grey
        return LinearSegmentedColormap.from_list("cb_sat", rgb)

    # Colorblind-safe distinct hues (Wong palette): blue, vermillion, bluish-green
    cb = sns.color_palette("colorblind")
    palette = {
        "model": saturation_cmap_from_rgb(cb[2]),     # bluish-green for "Ours"
        "baseline": saturation_cmap_from_rgb(cb[3]),  # vermillion for baseline
        "pre": saturation_cmap_from_rgb(cb[0]),       # blue for PRE
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
        pre_data_filt = pre_data[(pre_data["Dataset"]==dataset) ]
        # print(pre_data_filt.groupby(["Dataset", "Model", "feature_name", "proportion", "intensity"])[["mae"]].value_counts())
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

        # Single greyscale colorbar (winner colour communicates which method wins;
        # darkness shows the magnitude of the min error, on a shared scale).
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.04)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Greys)
        sm.set_array([])
        cb = ax.figure.colorbar(sm, cax=cax)
        cb.outline.set_visible(False)
        cb.ax.set_ylabel("Min error", rotation=270, labelpad=10, fontsize=8)
        cb.ax.tick_params(labelsize=7)

        # Tidy axes (legend drawn once at figure level below)
        ax.set_xlabel("Proportion OoD")
        ax.set_ylabel("Shift Intensity")
        ax.invert_yaxis()

    g.map_dataframe(draw_min_config_heatmap)
    g.set_axis_labels("Proportion OoD", "Shift Intensity")
    # single shared legend (winner color key)
    handles = [
        Patch(color=palette["model"](0.0), label="Ours"),
        Patch(color=palette["baseline"](0.0), label="Baseline"),
        Patch(color=palette["pre"](0.0), label="PRE"),
    ]
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    g.figure.legend(handles=handles, loc="lower center", ncol=3,
                    bbox_to_anchor=(0.5, -0.02), frameon=False, title="Winner (lowest MAE)")
    plt.savefig("figures/accuracy_prediction_error_heatmap.pdf", bbox_inches="tight")
    plt.show()

def error_per_accuracy():
    """Figure 3: MAE vs generalization gap, colorblind-safe; Q-Q residual inset per dataset."""
    from scipy import stats
    data = get_all_acc_prediction_results()

    mae_means = (
        data.groupby(["Dataset", "feature_name", "Model"], as_index=False)["mae"].mean()
    )
    best_combinations = (
        mae_means.loc[mae_means.groupby("Dataset")["mae"].idxmin(),
                      ["Dataset", "feature_name", "Model"]]
    )

    filtered_data = data.merge(best_combinations, on=["Dataset", "feature_name", "Model"])
    filtered_data["residual"] = filtered_data["pred"] - filtered_data["gap"]
    filtered_data = filtered_data[filtered_data["proportion"] == 1]

    cb = sns.color_palette("colorblind")
    scatter_color = cb[0]   # blue
    mean_color = cb[3]      # vermillion
    baseline_color = "black"

    def scatter_with_sliding_mean(data, x, y, window=31, **kwargs):
        ax = plt.gca()
        sns.scatterplot(data=data, x=x, y=y, alpha=0.45, ax=ax,
                        color=scatter_color, s=12, edgecolor="none")
        df = data[[x, y]].dropna().sort_values(x)
        if len(df) >= window:
            m = df[y].rolling(window=window, center=True).mean()
            ax.plot(df[x], m, linewidth=2, linestyle='-', color=mean_color, label='Sliding mean')
        else:
            ax.plot(df[x], df[y], linewidth=2, linestyle='-', color=mean_color, label='Line')

    g = sns.FacetGrid(filtered_data, col="Dataset", col_order=DATASETS,
                      sharex=False, sharey=False, col_wrap=3,
                      height=2.6, aspect=1.0)
    g.map_dataframe(scatter_with_sliding_mean, x="gap", y="mae", window=31)
    g.map_dataframe(sns.lineplot, x="gap", y="naive baseline mae",
                    color=baseline_color, linestyle="--", label="Baseline")
    g.set_axis_labels("Generalization Gap", "MAE")
    g.set_titles(col_template="{col_name}")
    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1)

    # Residual Q-Q inset per panel — quick visual answer to the "non-Gaussian residuals" critique
    for ax, dataset in zip(g.axes.flat, DATASETS):
        sub = filtered_data[filtered_data["Dataset"] == dataset]["residual"].dropna()
        if sub.empty:
            continue
        inset = ax.inset_axes([0.05, 0.06, 0.32, 0.32])
        stats.probplot(sub.values, dist="norm", plot=inset)
        inset.set_title("")
        inset.set_xlabel("")
        inset.set_ylabel("")
        inset.set_xticks([])
        inset.set_yticks([])
        for line in inset.get_lines():
            line.set_markersize(1.5)
            line.set_linewidth(0.6)
        inset.set_facecolor((1, 1, 1, 0.9))
        for spine in inset.spines.values():
            spine.set_linewidth(0.4)
        inset.text(0.04, 0.92, "Q-Q (resid.)", transform=inset.transAxes,
                   fontsize=5.5, va="top", ha="left")

    plt.savefig("figures/accuracy_prediction_error_per_gap.pdf", bbox_inches="tight")
    plt.show()


def dr_gap_correlation_distribution(batch_size=1, pretrain=True):
    """
    Robustness figure addressing the 'cherry-picking' critique.

    For every (Dataset x Architecture x Detector) configuration, computes the
    Spearman rank-correlation between OOD detection rate (DR) and the induced
    generalization gap, aggregating across all shifts and intensities. Renders
    the distribution as a swarm + box per dataset, with each architecture
    coloured. Demonstrates that the DR-gap relationship is broadly positive,
    not an artefact of the single best (detector, architecture) pair.
    """
    from scipy.stats import spearmanr

    merged = get_merged(batch_size=batch_size, pretrain=pretrain, filter_best=False)
    rows = []
    for (dataset, model, feature_name), grp in merged.groupby(["Dataset", "Model", "feature_name"]):
        sub = grp[["DR", "Generalization Gap"]].dropna()
        if len(sub) < 4 or sub["DR"].nunique() < 2 or sub["Generalization Gap"].nunique() < 2:
            continue
        rho, _ = spearmanr(sub["DR"], sub["Generalization Gap"])
        if np.isnan(rho):
            continue
        # |rho|: detector orientation can flip sign per (detector, architecture) — magnitude
        # captures monotonic coupling between DR and gap regardless of sign.
        rows.append({"Dataset": dataset, "Model": model,
                     "Detector": feature_name, "abs_rho": abs(float(rho))})
    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        print("No correlations could be computed; skipping figure.")
        return corr_df

    print("Correlation summary (|Spearman rho| per dataset):")
    print(corr_df.groupby("Dataset")["abs_rho"]
                 .agg(["median", "mean", "min", "max", "count"]))

    cb = sns.color_palette("colorblind")
    models_present = sorted(corr_df["Model"].unique().tolist())
    palette = {m: cb[i % len(cb)] for i, m in enumerate(models_present)}

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    sns.boxplot(data=corr_df, x="Dataset", y="abs_rho", order=DATASETS,
                color="lightgrey", fliersize=0, ax=ax, width=0.55)
    sns.swarmplot(data=corr_df, x="Dataset", y="abs_rho", order=DATASETS,
                  hue="Model", palette=palette, size=4, ax=ax,
                  edgecolor="black", linewidth=0.4)
    ax.axhline(0.5, color="black", linewidth=0.6, linestyle="--")
    ax.set_ylabel(r"$|$Spearman $\rho|$ (DR vs. gap)")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Architecture", bbox_to_anchor=(1.02, 1.0),
              loc="upper left", frameon=False, fontsize="x-small")
    plt.tight_layout()
    plt.savefig("figures/dr_gap_correlation_distribution.pdf", bbox_inches="tight")
    plt.show()
    return corr_df


def threshold_method_comparison(batch_size=1, pretrain=True):
    """
    Deployability figure addressing the 'threshold protocol' critique.

    Re-runs the accuracy-prediction pipeline (linear regression of gap on DR)
    for two thresholding regimes:
      - val_optimal: threshold tuned against an OOD calibration partition
      - id_quantile: threshold set as the 95th-percentile of InD scores only
    Reports per-dataset MAE for each regime alongside the naive baseline so
    the reader can see whether the linear model degrades when no OOD
    calibration data is available.
    """
    rows = []
    for tm in ("val_optimal", "id_quantile"):
        try:
            df_tm = get_all_ood_detector_data(batch_size, filter_organic=False,
                                              filter_best=True, pretrain=pretrain,
                                              threshold_method=tm)
        except Exception as e:
            print(f"[threshold_method_comparison] no rows for threshold_method={tm}: {e}")
            continue
        if df_tm.empty:
            print(f"[threshold_method_comparison] empty data for threshold_method={tm}; "
                  f"re-run ood_detector_correctness_prediction_accuracy (delete the existing "
                  f"data/pretrain/<model>/ood_detector_data/*.csv first) to populate id_quantile rows.")
            continue

        df_raw = load_all(batch_size, shift="", pretrain=pretrain)
        acc = df_raw.groupby(["Dataset", "Model", "fold"])["correct_prediction"].mean().reset_index()
        df_tm.rename(columns={"Fold": "fold"}, inplace=True)
        merged_tm = df_tm.merge(acc, on=["Dataset", "fold", "Model"], how="left")
        ind = (merged_tm[merged_tm["fold"] == "ind_val"]
                  .groupby("Dataset")["correct_prediction"].mean()
                  .rename("ind_val_acc").reset_index())
        merged_tm = merged_tm.merge(ind, on="Dataset", how="left")
        merged_tm["Generalization Gap"] = merged_tm["correct_prediction"] - merged_tm["ind_val_acc"]
        merged_tm.replace(SHIFT_PRINT_LUT, inplace=True)
        merged_tm["Shift"] = merged_tm["fold"].apply(
            lambda x: x.split("_")[0] if "_" in x else "Organic")

        for dataset in DATASETS:
            sub = merged_tm[merged_tm["Dataset"] == dataset].dropna(subset=["DR", "Generalization Gap"])
            if sub["Shift"].nunique() < 2 or len(sub) < 5:
                continue
            for shift in sub["Shift"].unique():
                train = sub[(sub["Shift"] != shift) & (sub["Shift"] != "FGSM")]
                test = sub[sub["Shift"] == shift]
                if len(train) < 2 or test.empty:
                    continue
                reg = LinearRegression()
                reg.fit(train["DR"].values.reshape(-1, 1),
                        train["Generalization Gap"].values.reshape(-1, 1))
                pred = reg.predict(test["DR"].values.reshape(-1, 1)).ravel()
                mae = float(np.mean(np.abs(test["Generalization Gap"].values - pred)))
                base = float(np.mean(np.abs(test["Generalization Gap"].values)))
                rows.append({"Dataset": dataset, "Threshold": tm,
                             "Shift": shift, "MAE": mae, "Baseline MAE": base})

    if not rows:
        print("[threshold_method_comparison] no results — id_quantile data not yet collected.")
        return pd.DataFrame()
    res = pd.DataFrame(rows)
    summary = (res.groupby(["Dataset", "Threshold"])[["MAE", "Baseline MAE"]]
                  .mean().reset_index())
    print("Threshold-method MAE summary:")
    print(summary)

    cb = sns.color_palette("colorblind")
    palette = {"val_optimal": cb[0], "id_quantile": cb[1]}
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    sns.barplot(data=summary, x="Dataset", y="MAE", hue="Threshold",
                order=DATASETS, palette=palette, ax=ax)
    methods_present = sorted(summary["Threshold"].unique().tolist())
    if "id_quantile" not in methods_present:
        ax.set_title("id_quantile rows missing — re-collect OOD detector data "
                     "(delete data/pretrain/*/ood_detector_data/*.csv) to populate",
                     fontsize=8, color="grey")
    base_line = (res.groupby("Dataset")["Baseline MAE"].mean()
                    .reindex(DATASETS).values)
    for i, b in enumerate(base_line):
        if not np.isnan(b):
            ax.hlines(b, i - 0.4, i + 0.4, colors="black",
                      linestyles="--", linewidth=1.2,
                      label="Baseline" if i == 0 else None)
    ax.set_ylabel("Accuracy-prediction MAE")
    ax.set_xlabel("")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize="x-small",
              bbox_to_anchor=(1.02, 1.0), loc="upper left", title="Threshold")
    plt.tight_layout()
    plt.savefig("figures/threshold_method_comparison.pdf", bbox_inches="tight")
    plt.show()
    return summary

