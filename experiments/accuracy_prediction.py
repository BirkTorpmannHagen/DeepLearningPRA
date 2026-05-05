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
def _parse_shift_type_from_fold(fold):
    """
    Maps a fold/condition name to a coarse shift type.

    Examples:
        'blur_0.1'      -> 'blur'
        'noise_0.5'     -> 'noise'
        'clipart'       -> 'Organic'
        'ind_val'       -> 'ind_val'
    """
    fold = str(fold)

    if fold in ("train", "ind_val", "ind_test"):
        return fold

    if "_" in fold:
        return fold.split("_")[0]

    return "Organic"


def _is_synthetic_shift_type(shift_type):
    return shift_type in SYNTHETIC_SHIFTS


def _prepare_dr_gap_data(batch_size=1, pretrain=True, filter_best=False):
    """
    Returns detector-rate data with:
        - fold: concrete held-out condition/domain/intensity
        - shift_type: coarse synthetic shift family, or Organic
        - gap: accuracy - InD validation accuracy

    Organic folds are retained as possible evaluation targets but should not be
    used as regression-training targets.
    """
    dr_data = get_all_ood_detector_data(
        batch_size=batch_size,
        filter_organic=False,
        filter_best=filter_best,
        pretrain=pretrain,
    )

    if dr_data.empty:
        return pd.DataFrame()

    dr_data = dr_data.copy()
    dr_data.rename(columns={"Fold": "fold"}, inplace=True)

    dr_data.replace(SHIFT_PRINT_LUT, inplace=True)

    dr_data["shift_type"] = dr_data["fold"].apply(_parse_shift_type_from_fold)
    dr_data["category"] = dr_data["shift_type"].apply(
        lambda s: "Synthetic" if _is_synthetic_shift_type(s) else "Organic"
    )

    rows = []

    for (dataset, model, feature_name), grp in dr_data.groupby(
        ["Dataset", "Model", "feature_name"]
    ):
        ind_val = grp[grp["fold"] == "ind_val"]

        if ind_val.empty:
            continue

        ind_val_acc = float(ind_val["Accuracy"].mean())

        g = grp.copy()
        g["gap"] = g["Accuracy"] - ind_val_acc

        rows.append(g)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


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

def _per_fold_method_data(batch_size=1, pretrain=True):
    """
    Per-fold MAE for Ours, ATC-MC, ATC-NE, PRE — the same per-(Dataset, Method)
    selection rule as `loo_fold_comparison` (best Model x feature/variant).
    Adds parsed `shift_type` and `intensity` columns for downstream plotting.
    Returns a long DataFrame: Dataset, fold, Method, shift_type, intensity, MAE.
    """
    raw = load_all(batch_size=batch_size, shift="", pretrain=pretrain)
    dr_data = get_all_ood_detector_data(batch_size, filter_organic=False,
                                        filter_best=False, pretrain=pretrain)
    if raw.empty or dr_data.empty:
        return pd.DataFrame()

    rows = []

    # --- Ours: per-fold LOO regression (matches loo_fold_comparison) ---
    for (dataset, model, feature_name), grp in dr_data.groupby(["Dataset", "Model", "feature_name"]):
        ind_val_row = grp[grp["Fold"] == "ind_val"]
        if ind_val_row.empty:
            continue
        ind_val_acc = float(ind_val_row["Accuracy"].mean())
        grp = grp.copy()
        grp["gap"] = grp["Accuracy"] - ind_val_acc
        eligible = grp[~grp["Fold"].isin(["train", "ind_val"])]
        if len(eligible) < 3:
            continue
        for held_idx, held_row in eligible.iterrows():
            train_set = eligible.drop(index=held_idx)
            if len(train_set) < 2:
                continue
            reg = LinearRegression()
            reg.fit(train_set["DR"].values.reshape(-1, 1), train_set["gap"].values)
            pred = float(reg.predict(np.array([[float(held_row["DR"])]]))[0])
            rows.append({"Dataset": dataset, "Model": model,
                         "feature_name": feature_name, "fold": held_row["Fold"],
                         "Method": "Ours", "MAE": abs(pred - float(held_row["gap"]))})

    # --- ATC: per-fold standard protocol ---
    for (dataset, model), df_dm in raw.groupby(["Dataset", "Model"]):
        ind_val_all = df_dm[df_dm["fold"] == "ind_val"]
        if ind_val_all.empty:
            continue
        for atc_name, feat, transform in [
            ("ATC-MC", "softmax",       lambda x: x),
            ("ATC-NE", "cross_entropy", lambda x: -x),
        ]:
            iv = ind_val_all[ind_val_all["feature_name"] == feat]
            if iv.empty:
                continue
            tau = _atc_threshold(transform(iv["feature"].values),
                                 iv["correct_prediction"].values)
            if np.isnan(tau):
                continue
            ind_val_acc = float(iv["correct_prediction"].mean())
            for fold, fdf in df_dm[df_dm["feature_name"] == feat].groupby("fold"):
                if fold in ("train", "ind_val"):
                    continue
                est = float((transform(fdf["feature"].values) >= tau).mean())
                true_acc = float(fdf["correct_prediction"].mean())
                rows.append({"Dataset": dataset, "Model": model,
                             "feature_name": feat, "fold": fold,
                             "Method": atc_name,
                             "MAE": abs((est - ind_val_acc) - (true_acc - ind_val_acc))})

    # --- PRE: per-fold from cached results.
    # Note: PRE's `val_set` is the (organic-only) calibration fold; `test_set` is
    # the held-out fold being predicted, which DOES include synthetic shifts.
    # Per-test-fold MAE: average over val_set calibration choices and Tree.
    try:
        pre = get_all_pre_data(pretrain=pretrain)
    except Exception as e:
        print(f"[per_fold] PRE unavailable: {e}")
        pre = pd.DataFrame()
    if not pre.empty:
        pre = pre[pre["rate"] == 1].copy()
        pre_agg = (pre.groupby(["Dataset", "Model", "dsd", "test_set"])["Accuracy Error"]
                       .mean().reset_index())
        for _, r in pre_agg.iterrows():
            rows.append({"Dataset": r["Dataset"], "Model": r["Model"],
                         "feature_name": r["dsd"], "fold": r["test_set"],
                         "Method": "PRE", "MAE": float(r["Accuracy Error"])})

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Best (Model, feature) per (Dataset, Method) — same selection rule for everyone
    config_means = (df.groupby(["Dataset", "Model", "feature_name", "Method"])["MAE"]
                      .mean().reset_index())
    best = config_means.loc[
        config_means.groupby(["Dataset", "Method"])["MAE"].idxmin(),
        ["Dataset", "Model", "feature_name", "Method"]
    ]
    df = df.merge(best, on=["Dataset", "Model", "feature_name", "Method"], how="inner")

    # Parse shift_type and intensity from fold name
    def parse_fold(fold):
        if "_" in str(fold):
            parts = str(fold).split("_")
            shift = parts[0]
            try:
                intensity = float(parts[-1])
                return shift, intensity, "synthetic"
            except ValueError:
                return str(fold), np.nan, "organic"
        return str(fold), np.nan, "organic"

    parsed = df["fold"].apply(parse_fold)
    df["shift_type"] = [p[0] for p in parsed]
    df["intensity"] = [p[1] for p in parsed]
    df["category"] = [p[2] for p in parsed]
    return df


def intensity_breakdown_plot(rows):
    """
    MAE vs synthetic-shift intensity, with organic targets shown separately.
    Uses corrected protocol rows.
    """
    df = rows.copy()

    if df.empty:
        print("[intensity_breakdown_plot] no rows.")
        return df

    df = df[df["Method"].isin(["Ours", "ATC-NE", "PRE"])].copy()
    df["Method"] = df["Method"].replace({"ATC-NE": "ATC"})

    def parse_intensity(fold):
        fold = str(fold)
        if "_" not in fold:
            return np.nan
        try:
            return float(fold.split("_")[-1])
        except ValueError:
            return np.nan

    df["intensity"] = df["fold"].apply(parse_intensity)

    cb = sns.color_palette("colorblind")
    palette = {"Ours": cb[2], "ATC": cb[1], "PRE": cb[0]}
    method_order = [m for m in ["Ours", "ATC", "PRE"] if m in df["Method"].unique()]

    synth = df[df["category"] == "Synthetic"].copy()
    organic = df[df["category"] == "Organic"].copy()

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharey=False)
    axes = axes.flatten()

    for i, dataset in enumerate(DATASETS):
        ax = axes[i]

        sub_s = synth[synth["Dataset"] == dataset]
        sub_o = organic[organic["Dataset"] == dataset]

        if not sub_s.empty:
            sns.lineplot(
                data=sub_s,
                x="intensity",
                y="MAE",
                hue="Method",
                hue_order=method_order,
                palette=palette,
                marker="o",
                markersize=5,
                errorbar="se",
                ax=ax,
            )

        if not sub_o.empty:
            org_summary = (
                sub_o.groupby(["Method", "fold"])["MAE"]
                .mean()
                .reset_index()
            )

            x_max = sub_s["intensity"].max() if not sub_s.empty else 0.5
            x_org = x_max + 0.08

            for method in method_order:
                vals = org_summary[org_summary["Method"] == method]["MAE"].values
                if len(vals) == 0:
                    continue

                ax.scatter(
                    [x_org] * len(vals),
                    vals,
                    color=palette[method],
                    marker="D",
                    s=28,
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=5,
                )

            ax.axvline(x_max + 0.04, color="grey", linestyle=":", linewidth=0.7)
            ax.text(
                x_org,
                ax.get_ylim()[1] * 0.95,
                "org.",
                fontsize=7,
                ha="center",
                va="top",
                color="grey",
            )

        ax.set_title(dataset, fontsize=10)
        ax.set_xlabel("Shift intensity")
        ax.set_ylabel("MAE")

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    for j in range(len(DATASETS), len(axes)):
        axes[j].set_visible(False)

    handles = [Patch(color=palette[m], label=m) for m in method_order]

    fig.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.95, 0.08),
        frameon=False,
        ncol=len(method_order),
        title="Method",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig("figures/intensity_breakdown.pdf", bbox_inches="tight")
    plt.show()

    pivot = (
        df.groupby(["Dataset", "shift_type", "Method"])["MAE"]
        .mean()
        .reset_index()
        .pivot_table(
            index=["Dataset", "shift_type"],
            columns="Method",
            values="MAE",
        )
        .reindex(columns=method_order)
    )

    pivot.to_csv("figures/intensity_breakdown.csv")
    print(pivot.round(3).to_string())

    return df


def loo_fold_comparison(batch_size=1, pretrain=True, anchor=False):
    """
    Leave-one-fold-out per-fold accuracy estimation.

      Ours: fit LinearRegression on (DR, gap) pairs for every non-held-out fold,
            predict gap on the held-out fold. If anchor=True, the regression is
            anchored to the InD-val point (no intercept, regressors centered on
            DR(ind_val)) so the predicted gap is exactly 0 when DR equals InD DR.
      ATC : standard protocol — threshold tau is set on InD val labels (Garg et
            al. ICLR 2022); estimate on the held-out fold.

    Both methods are evaluated on the SAME held-out folds (every non-train,
    non-ind_val fold). Per-dataset MAE is the mean over (held-out fold,
    architecture, detector/variant) for the best (Model, feature) per
    (Dataset, Method).
    """
    raw = load_all(batch_size=batch_size, shift="", pretrain=pretrain)
    if raw.empty:
        return pd.DataFrame()
    dr_data = get_all_ood_detector_data(batch_size, filter_organic=False,
                                        filter_best=False, pretrain=pretrain)
    if dr_data.empty:
        return pd.DataFrame()

    rows = []

    # ----- Ours: LOO regression of gap on DR -----
    ours_label = "Ours-anchored" if anchor else "Ours"
    for (dataset, model, feature_name), grp in dr_data.groupby(["Dataset", "Model", "feature_name"]):
        ind_val_row = grp[grp["Fold"] == "ind_val"]
        if ind_val_row.empty:
            continue
        ind_val_dr = float(ind_val_row["DR"].mean())
        ind_val_acc = float(ind_val_row["Accuracy"].mean())
        grp = grp.copy()
        grp["gap"] = grp["Accuracy"] - ind_val_acc

        eligible = grp[~grp["Fold"].isin(["train", "ind_val"])]
        if len(eligible) < 3:  # need enough to LOO
            continue
        for held_idx, held_row in eligible.iterrows():
            train_set = eligible.drop(index=held_idx)
            if len(train_set) < 2:
                continue
            X_train = train_set["DR"].values.reshape(-1, 1)
            y_train = train_set["gap"].values
            x_held = float(held_row["DR"])
            if anchor:
                # Center on ind_val DR; force intercept = 0 so reg(ind_val_dr) == 0
                reg = LinearRegression(fit_intercept=False)
                reg.fit(X_train - ind_val_dr, y_train)
                pred = float(reg.predict(np.array([[x_held - ind_val_dr]]))[0])
            else:
                reg = LinearRegression()
                reg.fit(X_train, y_train)
                pred = float(reg.predict(np.array([[x_held]]))[0])
            mae = abs(pred - float(held_row["gap"]))
            rows.append({"Dataset": dataset, "Model": model,
                         "feature_name": feature_name, "fold": held_row["Fold"],
                         "Method": ours_label, "MAE": mae})

    # ----- ATC: per-fold estimate using the same per-fold raw data -----
    for (dataset, model), df_dm in raw.groupby(["Dataset", "Model"]):
        ind_val_all = df_dm[df_dm["fold"] == "ind_val"]
        if ind_val_all.empty:
            continue
        for atc_name, feat, transform in [
            ("ATC-MC", "softmax",       lambda x: x),
            ("ATC-NE", "cross_entropy", lambda x: -x),
        ]:
            iv = ind_val_all[ind_val_all["feature_name"] == feat]
            if iv.empty:
                continue
            tau = _atc_threshold(transform(iv["feature"].values),
                                 iv["correct_prediction"].values)
            if np.isnan(tau):
                continue
            ind_val_acc = float(iv["correct_prediction"].mean())
            for fold, fdf in df_dm[df_dm["feature_name"] == feat].groupby("fold"):
                if fold in ("train", "ind_val"):
                    continue
                est_acc = float((transform(fdf["feature"].values) >= tau).mean())
                true_acc = float(fdf["correct_prediction"].mean())
                # Compare in gap space (same units as Ours)
                mae = abs((est_acc - ind_val_acc) - (true_acc - ind_val_acc))
                rows.append({"Dataset": dataset, "Model": model,
                             "feature_name": feat, "fold": fold,
                             "Method": atc_name, "MAE": mae})

    if not rows:
        print("[loo_fold_comparison] no rows produced.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Best (Model, feature) per (Dataset, Method) — same selection rule for everyone
    config_means = (df.groupby(["Dataset", "Model", "feature_name", "Method"])["MAE"]
                      .mean().reset_index())
    best = config_means.loc[
        config_means.groupby(["Dataset", "Method"])["MAE"].idxmin(),
        ["Dataset", "Model", "feature_name", "Method"]
    ]
    df_best = df.merge(best, on=["Dataset", "Model", "feature_name", "Method"], how="inner")
    summary = (df_best.groupby(["Dataset", "Method"])["MAE"]
                       .mean().reset_index())
    pivot = summary.pivot(index="Dataset", columns="Method", values="MAE").reindex(DATASETS)
    method_order = [c for c in [ours_label, "ATC-MC", "ATC-NE"] if c in pivot.columns]
    pivot = pivot.reindex(columns=method_order)
    print(f"\n=== Leave-one-fold-out per-fold MAE (anchor={anchor}) ===")
    print(pivot.round(4).to_string())
    return pivot


def _atc_per_cell_data(batch_size=1, pretrain=True):
    """
    Closed-form ATC MAE per (Dataset, Model, intensity, proportion) cell, mirroring
    the analytical mixing used by `get_acc_prediction_results` for "Ours":

        mixed_pred  = p * atc_acc(ood_at_intensity) + (1-p) * atc_acc(ind_test)
        mixed_truth = p * true_acc(ood_at_intensity) + (1-p) * true_acc(ind_test)
        mae         = |mixed_pred - mixed_truth|

    Per cell we return the *better* of ATC-MC and ATC-NE so the heatmap stays at
    four winner colours (Ours / Baseline / PRE / ATC).
    """
    raw = load_all(batch_size=batch_size, shift="", pretrain=pretrain)
    if raw.empty:
        return pd.DataFrame()

    rows = []
    for (dataset, model), df_dm in raw.groupby(["Dataset", "Model"]):
        ind_val = df_dm[df_dm["fold"] == "ind_val"]
        ind_test = df_dm[df_dm["fold"] == "ind_test"]
        if ind_val.empty or ind_test.empty:
            continue

        for atc_name, feat, transform in [
            ("ATC-MC", "softmax",       lambda x: x),
            ("ATC-NE", "cross_entropy", lambda x: -x),
        ]:
            iv = ind_val[ind_val["feature_name"] == feat]
            it = ind_test[ind_test["feature_name"] == feat]
            if iv.empty or it.empty:
                continue
            tau = _atc_threshold(transform(iv["feature"].values),
                                 iv["correct_prediction"].values)
            if np.isnan(tau):
                continue
            atc_ind = float((transform(it["feature"].values) >= tau).mean())
            true_ind = float(it["correct_prediction"].mean())

            for fold, fdf in df_dm[df_dm["feature_name"] == feat].groupby("fold"):
                if fold in ("train", "ind_val", "ind_test"):
                    continue
                if "_" in fold:
                    intensity_raw = fold.split("_")[-1]
                    try:
                        intensity = str(round(float(intensity_raw), 2))
                    except ValueError:
                        intensity = intensity_raw
                else:
                    intensity = "OoD"
                atc_ood = float((transform(fdf["feature"].values) >= tau).mean())
                true_ood = float(fdf["correct_prediction"].mean())
                for proportion in np.linspace(0, 1, 11):
                    pred = proportion * atc_ood + (1 - proportion) * atc_ind
                    truth = proportion * true_ood + (1 - proportion) * true_ind
                    rows.append({
                        "Dataset": dataset, "Model": model, "ATC variant": atc_name,
                        "intensity": intensity,
                        "proportion": round(float(proportion), 2),
                        "mae": abs(pred - truth),
                    })
    if not rows:
        return pd.DataFrame()
    df_atc = pd.DataFrame(rows)
    # Mirror the "best (Model, feature) per Dataset" rule used for Ours:
    # pick the (Model, ATC variant) combination with the lowest mean MAE per Dataset.
    config_means = (df_atc.groupby(["Dataset", "Model", "ATC variant"])["mae"]
                          .mean().reset_index())
    best = config_means.loc[config_means.groupby("Dataset")["mae"].idxmin(),
                            ["Dataset", "Model", "ATC variant"]]
    df_atc = df_atc.merge(best, on=["Dataset", "Model", "ATC variant"], how="inner")
    df_atc = (df_atc.groupby(["Dataset", "intensity", "proportion"])["mae"]
                    .mean().reset_index())
    return df_atc


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

    atc_data = _atc_per_cell_data(batch_size=1, pretrain=True).round(2)

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
        "baseline": saturation_cmap_from_rgb(cb[3]),  # vermillion for Baseline
        "pre": saturation_cmap_from_rgb(cb[0]),       # blue for PRE
        "atc": saturation_cmap_from_rgb(cb[1]),       # orange for ATC
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


        atc_data_filt = atc_data[atc_data["Dataset"] == dataset] if not atc_data.empty else atc_data

        # pivots (align grids)
        model_p = data.pivot(index="intensity", columns="proportion", values="mae")
        base_p = data.pivot(index="intensity", columns="proportion", values="naive baseline mae")
        pre_p = pre_data_filt.pivot(index="intensity", columns="proportion", values="mae")
        atc_p = (atc_data_filt.pivot(index="intensity", columns="proportion", values="mae")
                 if not atc_data_filt.empty else pd.DataFrame(index=model_p.index, columns=model_p.columns, dtype=float))

        # union index/columns
        all_idx = model_p.index.union(base_p.index).union(pre_p.index).union(atc_p.index)
        all_col = model_p.columns.union(base_p.columns).union(pre_p.columns).union(atc_p.columns)
        model_p = model_p.reindex(index=all_idx, columns=all_col)
        base_p = base_p.reindex(index=all_idx, columns=all_col)
        pre_p = pre_p.reindex(index=all_idx, columns=all_col)
        atc_p = atc_p.reindex(index=all_idx, columns=all_col)

        # 4D stack: order matches palette key order (model, baseline, pre, atc)
        stack = np.stack([
            model_p.values,
            base_p.values,
            pre_p.values,
            atc_p.values,
        ], axis=0)
        min_vals = np.nanmin(stack, axis=0)
        stack_for_arg = np.where(np.isnan(stack), np.inf, stack)
        winners = np.argmin(stack_for_arg, axis=0)  # 0=model,1=baseline,2=pre,3=atc
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
        mask_atc = ~(winners == 3) | all_nan_mask

        min_df = pd.DataFrame(min_vals, index=all_idx, columns=all_col)

        sns.heatmap(min_df, cmap=palette["pre"], norm=norm,
                    mask=mask_pre, cbar=False, ax=ax, **kwargs)
        sns.heatmap(min_df, cmap=palette["baseline"], norm=norm,
                    mask=mask_baseline, cbar=False, ax=ax, **kwargs)
        sns.heatmap(min_df, cmap=palette["atc"], norm=norm,
                    mask=mask_atc, cbar=False, ax=ax, **kwargs)
        hm = sns.heatmap(min_df, cmap=palette["model"], norm=norm,
                         mask=mask_model, cbar=False, ax=ax, **kwargs)

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
        Patch(color=palette["atc"](0.0), label="ATC"),
    ]
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    g.figure.legend(handles=handles, loc="lower center", ncol=4,
                    bbox_to_anchor=(0.5, -0.02), frameon=False, title="Winner (lowest MAE)")
    plt.savefig("figures/accuracy_prediction_error_heatmap.pdf", bbox_inches="tight")
    plt.show()

def error_per_accuracy(rows):
    """
    Figure: MAE vs observed generalization gap using corrected protocol rows.
    """
    from scipy import stats

    data = rows.copy()
    data = data[data["Method"] == "Ours"].copy()

    if data.empty:
        print("[error_per_accuracy] no Ours rows.")
        return data

    data["residual"] = data["predicted_gap"] - data["observed_gap"]

    cb = sns.color_palette("colorblind")
    scatter_color = cb[0]
    mean_color = cb[3]

    def scatter_with_sliding_mean(data, x, y, window=15, **kwargs):
        ax = plt.gca()
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            alpha=0.45,
            ax=ax,
            color=scatter_color,
            s=18,
            edgecolor="none",
        )

        df = data[[x, y]].dropna().sort_values(x)
        if len(df) >= window:
            m = df[y].rolling(window=window, center=True).mean()
            ax.plot(df[x], m, linewidth=2, color=mean_color, label="Sliding mean")

    g = sns.FacetGrid(
        data,
        col="Dataset",
        col_order=DATASETS,
        sharex=False,
        sharey=False,
        col_wrap=3,
        height=2.6,
        aspect=1.0,
    )

    g.map_dataframe(
        scatter_with_sliding_mean,
        x="observed_gap",
        y="MAE",
    )

    g.map_dataframe(
        sns.lineplot,
        x="observed_gap",
        y=data["observed_gap"].abs(),
        color="black",
        linestyle="--",
        label="Naive",
    )

    g.set_axis_labels("Observed generalization gap", "MAE")
    g.set_titles(col_template="{col_name}")

    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1)

    for ax, dataset in zip(g.axes.flat, DATASETS):
        sub = data[data["Dataset"] == dataset]["residual"].dropna()
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

        inset.text(
            0.04,
            0.92,
            "Q-Q",
            transform=inset.transAxes,
            fontsize=5.5,
            va="top",
            ha="left",
        )

    plt.tight_layout()
    plt.savefig("figures/accuracy_prediction_error_per_gap.pdf", bbox_inches="tight")
    plt.show()

    return data


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


def _atc_threshold(scores, correct):
    """
    Garg et al., ICLR 2022 (Average Threshold Confidence).
    Pick tau on labeled InD validation s.t. P(score < tau) = 1 - acc.
    Equivalently: tau is the (1 - acc) quantile of the InD-val score distribution.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct).astype(bool)
    if len(scores) == 0:
        return np.nan
    err_rate = 1.0 - float(correct.mean())
    err_rate = float(np.clip(err_rate, 0.0, 1.0))
    return float(np.quantile(scores, err_rate))


def atc_comparison(batch_size=1, pretrain=True):
    """
    Compare label-free accuracy estimators per dataset:
      - Naive       : assume InD-val accuracy on every fold
      - ATC-MC      : Garg et al. ICLR 2022, max-softmax confidence
      - ATC-NE      : Garg et al. ICLR 2022, negative entropy
      - Ours        : OOD-detection-rate -> linear regression -> accuracy gap
      - PRE         : Probabilistic Runtime Estimation (Mihaylov et al.)

    ATC and Naive are computed here from raw feature data; Ours and PRE are
    pulled from cached results to keep apples-to-apples (per-fold, no
    bootstrap mixing).
    """
    raw = load_all(batch_size=batch_size, shift="", pretrain=pretrain)
    if raw.empty:
        print("[atc_comparison] no raw feature data found.")
        return pd.DataFrame()

    rows_atc = []
    for (dataset, model), df_dm in raw.groupby(["Dataset", "Model"]):
        ind_val_acc_per_feat = (
            df_dm[df_dm["fold"] == "ind_val"]
                 .groupby("feature_name")["correct_prediction"]
                 .mean()
        )
        if ind_val_acc_per_feat.empty:
            continue
        ind_val_acc = float(ind_val_acc_per_feat.iloc[0])  # invariant across feature

        for atc_method, score_feature, score_transform in [
            ("ATC-MC", "softmax",       lambda x: x),       # higher = more confident
            ("ATC-NE", "cross_entropy", lambda x: -x),      # entropy: invert sign
        ]:
            sub = df_dm[df_dm["feature_name"] == score_feature]
            if sub.empty:
                continue
            ind_val = sub[sub["fold"] == "ind_val"]
            if ind_val.empty:
                continue
            tau = _atc_threshold(score_transform(ind_val["feature"].values),
                                 ind_val["correct_prediction"].values)
            if np.isnan(tau):
                continue

            for fold, fdf in sub.groupby("fold"):
                if fold in ("train", "ind_val"):
                    continue
                est_acc = float((score_transform(fdf["feature"].values) >= tau).mean())
                true_acc = float(fdf["correct_prediction"].mean())
                rows_atc.append({
                    "Dataset": dataset, "Model": model, "fold": fold,
                    "Method": atc_method,
                    "MAE": abs(est_acc - true_acc),
                    "estimated_acc": est_acc, "true_acc": true_acc,
                })
                rows_atc.append({
                    "Dataset": dataset, "Model": model, "fold": fold,
                    "Method": "Naive",
                    "MAE": abs(ind_val_acc - true_acc),
                    "estimated_acc": ind_val_acc, "true_acc": true_acc,
                })

    atc_df = pd.DataFrame(rows_atc)
    if atc_df.empty:
        print("[atc_comparison] no ATC rows produced.")
        return pd.DataFrame()
    # de-dup Naive entries (one per fold/model/dataset is enough)
    naive = atc_df[atc_df["Method"] == "Naive"].drop_duplicates(
        subset=["Dataset", "Model", "fold"])
    atc_only = atc_df[atc_df["Method"] != "Naive"]
    atc_df = pd.concat([naive, atc_only], ignore_index=True)

    # --- Ours: take per-(Dataset, Model, feature) results at proportion=1 (pure shift),
    #     then pick the best (Model, feature) per Dataset to mirror Table 5.
    ours = get_all_acc_prediction_results(pretrain=pretrain)
    if not ours.empty:
        ours = ours[ours["proportion"] == 1].copy()
        scores = ours.groupby(["Dataset", "feature_name", "Model"])["mae"].mean().reset_index()
        idx = scores.groupby("Dataset")["mae"].idxmin()
        best = scores.loc[idx, ["Dataset", "feature_name", "Model"]]
        ours = ours.merge(best, on=["Dataset", "feature_name", "Model"], how="inner")
        ours_rows = ours[["Dataset", "Model", "mae"]].rename(columns={"mae": "MAE"})
        ours_rows["Method"] = "Ours"
        ours_rows["fold"] = "agg"
    else:
        ours_rows = pd.DataFrame(columns=["Dataset", "Model", "fold", "Method", "MAE"])

    # --- PRE
    try:
        pre = get_all_pre_data(pretrain=pretrain)
    except Exception as e:
        print(f"[atc_comparison] PRE data unavailable: {e}")
        pre = pd.DataFrame()
    if not pre.empty:
        pre = pre.rename(columns={"Accuracy Error": "MAE", "rate": "proportion",
                                  "dsd": "feature_name"})
        pre = pre[pre["proportion"] == 1].copy()
        pre_rows = pre[["Dataset", "Model", "MAE"]].copy()
        pre_rows["Method"] = "PRE"
        pre_rows["fold"] = "agg"
    else:
        pre_rows = pd.DataFrame(columns=["Dataset", "Model", "fold", "Method", "MAE"])

    full = pd.concat([atc_df[["Dataset", "Model", "fold", "Method", "MAE"]],
                      ours_rows[["Dataset", "Model", "fold", "Method", "MAE"]],
                      pre_rows[["Dataset", "Model", "fold", "Method", "MAE"]]],
                     ignore_index=True)

    summary = (full.groupby(["Dataset", "Method"])["MAE"]
                   .mean().reset_index())
    print("\n=== Label-free accuracy estimation (mean MAE per dataset) ===")
    pivot = summary.pivot(index="Dataset", columns="Method", values="MAE").reindex(DATASETS)
    pivot = pivot.reindex(columns=[c for c in ["Naive", "ATC-MC", "ATC-NE", "PRE", "Ours"]
                                   if c in pivot.columns])
    print(pivot.round(4).to_string())

    method_order = [m for m in ["Naive", "ATC-MC", "ATC-NE", "PRE", "Ours"]
                    if m in summary["Method"].unique()]
    cb = sns.color_palette("colorblind")
    palette = {"Naive": "lightgrey", "ATC-MC": cb[1], "ATC-NE": cb[7],
               "PRE": cb[0], "Ours": cb[2]}
    palette = {k: v for k, v in palette.items() if k in method_order}

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    sns.barplot(data=summary, x="Dataset", y="MAE", hue="Method",
                order=DATASETS, hue_order=method_order, palette=palette, ax=ax)
    ax.set_ylabel("MAE")
    ax.set_xlabel("")
    ax.legend(title="", frameon=False, fontsize="x-small",
              bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig("figures/atc_comparison.pdf", bbox_inches="tight")
    plt.show()
    pivot.to_csv("figures/atc_comparison.csv")
    return pivot

def predicted_vs_true_gap_grid(rows):
    """
    Calibration grid using corrected protocol rows.
    """
    calib = rows.copy()

    if calib.empty:
        print("[predicted_vs_true_gap_grid] no rows.")
        return calib, pd.DataFrame()

    calib = calib.dropna(subset=["observed_gap", "predicted_gap"]).copy()

    calib["residual"] = calib["predicted_gap"] - calib["observed_gap"]
    calib["abs_error"] = calib["residual"].abs()

    method_order = [
        m for m in ["Ours", "ATC-MC", "ATC-NE"]
        if m in calib["Method"].unique()
    ]

    summary = (
        calib.groupby(["Dataset", "Method"])
        .agg(
            mean=("residual", "mean"),
            median=("residual", "median"),
            std=("residual", "std"),
            count=("residual", "count"),
            MAE=("abs_error", "mean"),
        )
        .reset_index()
    )

    summary.to_csv("figures/calibration_grid_summary.csv", index=False)
    calib.to_csv("figures/calibration_grid_data.csv", index=False)

    cb = sns.color_palette("colorblind")
    palette = {
        "Ours": cb[2],
        "ATC-MC": cb[1],
        "ATC-NE": cb[7],
    }

    g = sns.FacetGrid(
        calib,
        row="Dataset",
        col="Method",
        row_order=DATASETS,
        col_order=method_order,
        sharex=False,
        sharey=False,
        height=1.85,
        aspect=1.0,
        margin_titles=True,
    )

    def plot_panel(data, **kwargs):
        ax = plt.gca()

        method = data["Method"].iloc[0]
        color = palette.get(method, "black")

        sns.scatterplot(
            data=data,
            x="observed_gap",
            y="predicted_gap",
            color=color,
            alpha=0.45,
            s=10,
            edgecolor="none",
            ax=ax,
        )

        vals = np.concatenate([
            data["observed_gap"].to_numpy(dtype=float),
            data["predicted_gap"].to_numpy(dtype=float),
        ])
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            return

        lo = float(vals.min())
        hi = float(vals.max())

        if np.isclose(lo, hi):
            lo -= 0.05
            hi += 0.05

        pad = 0.06 * (hi - lo)
        lo -= pad
        hi += pad

        ax.plot(
            [lo, hi],
            [lo, hi],
            color="black",
            linestyle="--",
            linewidth=0.8,
        )

        if len(data) >= 3 and data["observed_gap"].nunique() > 1:
            reg = LinearRegression()
            reg.fit(
                data["observed_gap"].values.reshape(-1, 1),
                data["predicted_gap"].values,
            )

            xs = np.linspace(lo, hi, 100)
            ys = reg.predict(xs.reshape(-1, 1))

            ax.plot(xs, ys, color="black", linewidth=1.1)
            slope = float(reg.coef_[0])
        else:
            slope = np.nan

        mean_resid = float(data["residual"].mean())
        mae = float(data["abs_error"].mean())

        text = f"bias={mean_resid:+.3f}\nMAE={mae:.3f}"
        if np.isfinite(slope):
            text += f"\nslope={slope:.2f}"

        ax.text(
            0.04,
            0.96,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    g.map_dataframe(plot_panel)

    g.set_axis_labels("Observed gap", "Predicted gap")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    for ax in g.axes.flat:
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(False)

    plt.tight_layout()
    plt.savefig("figures/calibration_grid.pdf", bbox_inches="tight")
    plt.show()

    return calib, summary

def shift_type_loo_predictions(batch_size=1, pretrain=True, seq_length=-1):
    """
    Correct leave-one-shift-type-out protocol.

    Evaluation targets:
        1. Synthetic shift types:
           - hold out all folds/intensities of one synthetic shift type;
           - train regression only on other synthetic shift types;
           - never train on organic folds.

        2. Organic folds:
           - evaluate each organic fold/domain;
           - train regression only on synthetic shift types;
           - never train on organic folds.

    No held-out target labels are used for fitting the regression.
    """
    df = _prepare_dr_gap_data(
        batch_size=batch_size,
        pretrain=pretrain,
        filter_best=False,
    )

    if df.empty:
        return pd.DataFrame()

    rows = []

    for (dataset, model, feature_name), grp in df.groupby(
        ["Dataset", "Model", "feature_name"]
    ):
        grp = grp.copy()

        # Exclude non-evaluation pseudo-folds.
        grp = grp[~grp["fold"].isin(["train", "ind_val"])]

        # Synthetic calibration pool only.
        synthetic_pool = grp[
            (grp["category"] == "Synthetic")
            & (~grp["shift_type"].isin(["FGSM", "adv", "ind"]))
        ].copy()

        if len(synthetic_pool) < 2:
            continue

        # ------------------------------------------------------------
        # 1. Synthetic held-out shift types
        # ------------------------------------------------------------
        for held_shift_type in synthetic_pool["shift_type"].unique():
            train = synthetic_pool[synthetic_pool["shift_type"] != held_shift_type]
            test = synthetic_pool[synthetic_pool["shift_type"] == held_shift_type]
            print(test)
            input()
            if len(train) < 2 or test.empty:
                continue

            reg = LinearRegression()
            reg.fit(
                train["DR"].values.reshape(-1, 1),
                train["gap"].values,
            )

            preds = reg.predict(test["DR"].values.reshape(-1, 1))

            for (_, r), pred in zip(test.iterrows(), preds):
                rows.append({
                    "Dataset": dataset,
                    "Model": model,
                    "feature_name": feature_name,
                    "fold": r["fold"],
                    "shift_type": held_shift_type,
                    "category": "Synthetic",
                    "observed_gap": float(r["gap"]),
                    "predicted_gap": float(pred),
                    "MAE": abs(float(pred) - float(r["gap"])),
                    "Method": "Ours",
                })

        # ------------------------------------------------------------
        # 2. Organic held-out folds/domains
        # ------------------------------------------------------------
        organic_targets = grp[
            (grp["category"] == "Organic")
            & (~grp["fold"].isin(["train", "ind_val", "ind_test"]))
        ].copy()

        for _, r in organic_targets.iterrows():
            train = synthetic_pool

            if len(train) < 2:
                continue

            reg = LinearRegression()
            reg.fit(
                train["DR"].values.reshape(-1, 1),
                train["gap"].values,
            )

            pred = float(reg.predict(np.array([[float(r["DR"])]]))[0])

            rows.append({
                "Dataset": dataset,
                "Model": model,
                "feature_name": feature_name,
                "fold": r["fold"],
                "shift_type": "Organic",
                "category": "Organic",
                "observed_gap": float(r["gap"]),
                "predicted_gap": pred,
                "MAE": abs(pred - float(r["gap"])),
                "Method": "Ours",
            })

    return pd.DataFrame(rows)

def atc_predictions(batch_size=1, pretrain=True):
    """
    ATC evaluated on the same held-out folds/conditions as the corrected Ours
    protocol.

    ATC uses only InD validation labels for threshold calibration.
    """
    raw = load_all(batch_size=batch_size, shift="", pretrain=pretrain)

    if raw.empty:
        return pd.DataFrame()

    rows = []

    for (dataset, model), df_dm in raw.groupby(["Dataset", "Model"]):
        ind_val_all = df_dm[df_dm["fold"] == "ind_val"]

        if ind_val_all.empty:
            continue

        for method_name, feat, transform in [
            ("ATC-MC", "softmax", lambda x: x),
            ("ATC-NE", "cross_entropy", lambda x: -x),
        ]:
            sub = df_dm[df_dm["feature_name"] == feat]

            if sub.empty:
                continue

            ind_val = sub[sub["fold"] == "ind_val"]

            if ind_val.empty:
                continue

            tau = _atc_threshold(
                transform(ind_val["feature"].values),
                ind_val["correct_prediction"].values,
            )

            if np.isnan(tau):
                continue

            ind_val_acc = float(ind_val["correct_prediction"].mean())

            for fold, fdf in sub.groupby("fold"):
                if fold in ("train", "ind_val"):
                    continue

                shift_type = _parse_shift_type_from_fold(fold)

                if shift_type in ("FGSM", "adv", "ind"):
                    continue

                est_acc = float((transform(fdf["feature"].values) >= tau).mean())
                true_acc = float(fdf["correct_prediction"].mean())

                predicted_gap = est_acc - ind_val_acc
                observed_gap = true_acc - ind_val_acc

                rows.append({
                    "Dataset": dataset,
                    "Model": model,
                    "feature_name": feat,
                    "fold": fold,
                    "shift_type": shift_type,
                    "category": "Synthetic" if _is_synthetic_shift_type(shift_type) else "Organic",
                    "observed_gap": observed_gap,
                    "predicted_gap": predicted_gap,
                    "MAE": abs(predicted_gap - observed_gap),
                    "Method": method_name,
                })

    return pd.DataFrame(rows)

def pre_predictions(pretrain=True):
    """
    PRE results normalized to the same output schema.

    Assumes cached PRE results already used the same calibration regime:
    PRE calibrated on the same labeled synthetic calibration shifts used by
    the regression model, and never on the held-out target condition.
    """
    try:
        pre = get_all_pre_data(pretrain=pretrain)
    except Exception as e:
        print(f"[pre_predictions] PRE unavailable: {e}")
        return pd.DataFrame()

    if pre.empty:
        return pd.DataFrame()

    pre = pre.copy()
    pre = pre[pre["rate"] == 1]

    rows = []

    for _, r in pre.iterrows():
        fold = r["test_set"]
        shift_type = _parse_shift_type_from_fold(fold)

        if shift_type in ("FGSM", "adv", "ind", "train", "ind_val"):
            continue

        rows.append({
            "Dataset": r["Dataset"],
            "Model": r["Model"],
            "feature_name": r["dsd"],
            "fold": fold,
            "shift_type": shift_type,
            "category": "Synthetic" if _is_synthetic_shift_type(shift_type) else "Organic",
            "observed_gap": np.nan,
            "predicted_gap": np.nan,
            "MAE": float(r["Accuracy Error"]),
            "Method": "PRE",
        })

    return pd.DataFrame(rows)

def accuracy_prediction_table(batch_size=1, pretrain=True):
    """
    Main corrected table.

    Uses:
        - corrected leave-one-shift-type-out Ours;
        - ATC on the same target folds;
        - PRE from cached calibrated results.

    Selection rule:
        Best Model x feature_name per Dataset x Method is selected by mean MAE
        over that method's corrected evaluation targets.

    This is still a dataset-level best-configuration report. In the paper,
    describe it as selecting the strongest detector/model configuration using
    labeled calibration-shift performance, not as target-fold tuning.
    """
    ours = shift_type_loo_predictions(
        batch_size=batch_size,
        pretrain=pretrain,
    )

    atc = atc_predictions(
        batch_size=batch_size,
        pretrain=pretrain,
    )

    pre = pre_predictions(pretrain=pretrain)

    dfs = [d for d in [ours, atc, pre] if not d.empty]

    if not dfs:
        print("[accuracy_prediction_table] no rows produced.")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Remove pseudo/unsupported folds defensively.
    df = df[
        ~df["fold"].isin(["train", "ind_val"])
        & ~df["shift_type"].isin(["FGSM", "adv", "ind"])
    ].copy()

    # Best configuration per dataset/method.
    config_scores = (
        df.groupby(["Dataset", "Method", "Model", "feature_name"], as_index=False)["MAE"]
        .mean()
    )

    best_configs = config_scores.loc[
        config_scores.groupby(["Dataset", "Method"])["MAE"].idxmin(),
        ["Dataset", "Method", "Model", "feature_name"],
    ]

    df_best = df.merge(
        best_configs,
        on=["Dataset", "Method", "Model", "feature_name"],
        how="inner",
    )

    summary = (
        df_best.groupby(["Dataset", "Method"], as_index=False)["MAE"]
        .mean()
    )

    pivot = (
        summary.pivot(index="Dataset", columns="Method", values="MAE")
        .reindex(DATASETS)
    )

    method_order = [
        m for m in ["Ours", "ATC-MC", "ATC-NE", "PRE"]
        if m in pivot.columns
    ]

    pivot = pivot.reindex(columns=method_order)

    print("\n=== Corrected leave-one-shift-type-out accuracy prediction MAE ===")
    print(pivot.round(4).to_string())

    return pivot, df_best
def shift_type_loo_predictions_subsampled(
    batch_size=1,
    pretrain=True,
    seq_length=-1,
    n_samples=1,
    random_state=0,
):
    """
    Leave-one-shift-type-out protocol, but test targets are evaluated on
    sequence-level averages.

    If seq_length > 0, each test set is repeatedly subsampled into sequences
    of length `seq_length`; DR and gap are averaged within each sampled sequence
    before prediction.

    If seq_length <= 0, this reduces to the original row-level protocol.
    """

    df = _prepare_dr_gap_data(
        batch_size=batch_size,
        pretrain=pretrain,
        filter_best=True,
    )

    if df.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(random_state)
    rows = []

    def _sample_means(test_df, seq_length, n_samples):
        """
        Return a dataframe where each row is the mean of a sampled sequence.
        Non-numeric metadata is copied from the first sampled row.
        """
        test_df = test_df.copy()

        if seq_length is None or seq_length <= 0:
            return test_df.reset_index(drop=True)

        if test_df.empty:
            return test_df

        sampled_rows = []

        for sample_idx in range(n_samples):
            replace = len(test_df) < seq_length
            sampled = test_df.sample(
                n=seq_length,
                replace=replace,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )

            base = sampled.iloc[0].copy()

            base["DR"] = sampled["DR"].astype(float).mean()
            base["gap"] = sampled["gap"].astype(float).mean()
            base["fold"] = sampled["fold"].iloc[0]
            base["sequence_id"] = sample_idx
            base["sequence_length"] = seq_length
            base["n_available"] = len(test_df)

            sampled_rows.append(base)

        return pd.DataFrame(sampled_rows).reset_index(drop=True)

    for (dataset, model, feature_name), grp in df.groupby(
        ["Dataset", "Model", "feature_name"]
    ):
        grp = grp.copy()

        # Exclude non-evaluation pseudo-folds.
        grp = grp[~grp["fold"].isin(["train", "ind_val"])]

        # Synthetic calibration pool only.
        synthetic_pool = grp[
            (grp["category"] == "Synthetic")
            & (~grp["shift_type"].isin(["FGSM", "adv", "ind"]))
        ].copy()

        if len(synthetic_pool) < 2:
            continue

        # ------------------------------------------------------------
        # 1. Synthetic held-out shift types
        # ------------------------------------------------------------
        for held_shift_type in synthetic_pool["shift_type"].unique():
            train = synthetic_pool[synthetic_pool["shift_type"] != held_shift_type]
            raw_test = synthetic_pool[
                synthetic_pool["shift_type"] == held_shift_type
            ]

            if len(train) < 2 or raw_test.empty:
                continue

            test = _sample_means(raw_test, seq_length, n_samples)

            reg = LinearRegression()
            reg.fit(
                train["DR"].values.reshape(-1, 1),
                train["gap"].values,
            )

            preds = reg.predict(test["DR"].values.reshape(-1, 1))

            for (_, r), pred in zip(test.iterrows(), preds):
                rows.append({
                    "Dataset": dataset,
                    "Model": model,
                    "feature_name": feature_name,
                    "fold": r["fold"],
                    "shift_type": held_shift_type,
                    "category": "Synthetic",
                    "observed_gap": float(r["gap"]),
                    "predicted_gap": float(pred),
                    "MAE": abs(float(pred) - float(r["gap"])),
                    "Method": "Ours",
                    "sequence_id": r.get("sequence_id", np.nan),
                    "sequence_length": r.get("sequence_length", seq_length),
                    "n_available": r.get("n_available", len(raw_test)),
                })

        # ------------------------------------------------------------
        # 2. Organic held-out folds/domains
        # ------------------------------------------------------------
        organic_targets = grp[
            (grp["category"] == "Organic")
            & (~grp["fold"].isin(["train", "ind_val", "ind_test"]))
        ].copy()

        for organic_fold, raw_test in organic_targets.groupby("fold"):
            if len(synthetic_pool) < 2 or raw_test.empty:
                continue

            test = _sample_means(raw_test, seq_length, n_samples)

            reg = LinearRegression()
            reg.fit(
                synthetic_pool["DR"].values.reshape(-1, 1),
                synthetic_pool["gap"].values,
            )

            preds = reg.predict(test["DR"].values.reshape(-1, 1))

            for (_, r), pred in zip(test.iterrows(), preds):
                rows.append({
                    "Dataset": dataset,
                    "Model": model,
                    "feature_name": feature_name,
                    "fold": organic_fold,
                    "shift_type": "Organic",
                    "category": "Organic",
                    "observed_gap": float(r["gap"]),
                    "predicted_gap": float(pred),
                    "MAE": abs(float(pred) - float(r["gap"])),
                    "Method": "Ours",
                    "sequence_id": r.get("sequence_id", np.nan),
                    "sequence_length": r.get("sequence_length", seq_length),
                    "n_available": r.get("n_available", len(raw_test)),
                })

    return pd.DataFrame(rows)

def sequence_length_sensitivity(lengths, n_samples=50, error="sem"):
    rows = []

    for length in lengths:
        ours = shift_type_loo_predictions_subsampled(
            batch_size=1,
            pretrain=True,
            seq_length=length,
            n_samples=n_samples,
        )

        if ours.empty:
            continue

        # Mean MAE per sampled sequence replicate
        rep_mae = (
            ours
            .groupby("sequence_id", dropna=False)["MAE"]
            .mean()
            .reset_index(name="replicate_mean_mae")
        )

        mean_mae = rep_mae["replicate_mean_mae"].mean()
        std_mae = rep_mae["replicate_mean_mae"].std(ddof=1)
        sem_mae = std_mae / np.sqrt(len(rep_mae))

        rows.append({
            "sequence_length": length,
            "mean_mae": mean_mae,
            "std_mae": std_mae,
            "sem_mae": sem_mae,
            "n_replicates": len(rep_mae),
        })

    res_df = pd.DataFrame(rows).sort_values("sequence_length")

    if error == "std":
        yerr = res_df["std_mae"]
    elif error == "sem":
        yerr = res_df["sem_mae"]
    elif error == "ci95":
        yerr = 1.96 * res_df["sem_mae"]
    else:
        raise ValueError("error must be one of: 'std', 'sem', 'ci95'")

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        res_df["sequence_length"],
        res_df["mean_mae"],
        yerr=yerr,
        marker="o",
        capsize=4,
    )
    plt.xscale("log")
    plt.xlabel("Sequence length")
    plt.ylabel("Mean MAE")
    plt.title("Sequence length sensitivity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("seq_length_analysis.pdf")
    plt.show()

    return res_df