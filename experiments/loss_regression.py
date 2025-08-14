import numpy as np
import pandas as pd
import pygam
import seaborn as sns
from matplotlib import pyplot as plt, patches as patches
from scipy.stats import spearmanr
import matplotlib.gridspec as gridspec

from components import OODDetector
from utils import *

def get_baseline_loss_estimate(df):
    loss_by_fold =  df.groupby(["Dataset", "shift", "shift_intensity"])["loss"].mean().reset_index()
    ind_val_loss = loss_by_fold[loss_by_fold["shift"]=="ind_val"][["Dataset", "loss"]]
    loss_by_fold = loss_by_fold.merge(ind_val_loss, on="Dataset", how="left")
    loss_by_fold["baseline_error"] = np.abs(loss_by_fold["loss_x"] - loss_by_fold["loss_y"])
    return loss_by_fold.groupby(["Dataset"])["baseline_error"].mean().reset_index()

def get_best_gam_data(batch_size=32):
    df = load_all(batch_size, shift="", samples=20)
    df = df[df["fold"] != "train"]
    shift_colors = dict(zip(df["shift"].unique(), sns.color_palette("pastel", len(df["shift"].unique()))))
    df = filter_max_loss(df)
    baseline_loss = get_baseline_loss_estimate(df)

    all_gam_results = pd.concat([
        pd.read_csv(f"gam_data/gam_prediction_errors_{dataset}_{feature_name}_{batch_size}.csv")
        for dataset in DATASETS for feature_name in DSDS if feature_name != "rabanser"
    ])

    all_gam_preds = pd.concat([
        pd.read_csv(f"gam_data/gam_fits_{dataset}_{feature_name}_{batch_size}.csv")
        for dataset in DATASETS for feature_name in DSDS if feature_name != "rabanser"
    ])

    # Compute mean MAE for each (Dataset, feature_name, train_shift)
    mean_mae = all_gam_results.groupby(
        ["Dataset", "Feature Name", "Train Shift"]
    )[["MAE", "MAPE"]].mean().reset_index()

    # Get index of best combination per Dataset
    best_combo = mean_mae.groupby("Dataset")[("MAE")].idxmin()
    best_metrics = mean_mae.iloc[best_combo].reset_index(drop=True)
    best_keys = best_metrics[["Dataset", "Feature Name", "Train Shift"]]

    # Filter all_gam_results
    filtered_all_gam_results = all_gam_results.merge(
        best_keys,
        on=["Dataset", "Feature Name", "Train Shift"],
        how="inner"
    )
    all_gam_preds = all_gam_preds.merge(
        best_keys,
        on=["Dataset", "Feature Name", "Train Shift"],
        how="inner"
    )
    df.rename(columns={"feature_name": "Feature Name"}, inplace=True)
    df = df.merge(
        best_keys,
        on=["Dataset", "Feature Name"],
        how="inner"
    )
    return filtered_all_gam_results, all_gam_preds, df

def plot_gam_errors(batch_size=32, q=21):
    gam_results, gam_preds, df = get_best_gam_data(batch_size=batch_size)

    # add numeric midpoint column
    def add_bins(g):
        bins = pd.qcut(g["Loss"], q=q, duplicates="drop")
        g = g.copy()
        g["loss_mid"] = bins.map(lambda iv: iv.mid)
        return g

    gr = gam_results.groupby("Dataset", group_keys=False).apply(add_bins)

    # aggregate mean & standard error per bin
    plot_df = (
        gr.groupby(["Dataset", "loss_mid"], as_index=False)
          .agg(
              MAE_mean=("MAE", "mean"),
              MAE_std=("MAE", "std"),
              n=("MAE", "size")
          )
    )
    plot_df["MAE_se"] = plot_df["MAE_std"] / np.sqrt(plot_df["n"])

    g = sns.FacetGrid(plot_df, col="Dataset", sharex=False, sharey=False,
                      height=3, col_wrap=3)

    def line_and_refs(data, color, **kwargs):
        data = data.sort_values("loss_mid")
        # plot mean ± error bars
        plt.errorbar(
            data["loss_mid"], data["MAE_mean"], yerr=data["MAE_se"] * 1.96,
            fmt="-o", color=color, capsize=3, label="Mean ± 95% CI"
        )

    def add_baseline_line(data, color, **kwargs):
        # facet's dataset name
        ds = data["Dataset"].iloc[0]

        # pick the correct loss column in df
        loss_col = "Loss" if "Loss" in df.columns else "loss"

        # baseline = mean loss on ind_val for this dataset
        base = df.loc[(df["Dataset"] == ds) & (df["shift"] == "ind_val"), loss_col].mean()

        # draw y = x - base across the current x-range (x is loss_mid)
        ax = plt.gca()
        xmin, xmax = ax.get_xlim()  # limits set by the errorbar plot already
        xx = np.linspace(xmin, xmax, 200)
        yy = np.abs(xx - base)
        plt.plot(xx, yy, linestyle="--", color=color, label="Baseline Estimate Error")


    g.map_dataframe(line_and_refs)
    g.map_dataframe(add_baseline_line, color="red")

    g.set_axis_labels("Loss (bin midpoint)", "Mean MAE")
    for ax in g.axes.flat:

        ax.legend(loc="best")
    plt.savefig("figures/gam_errors.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def gam_fits(batch_size=32):
    gam_results, gam_preds, df = get_best_gam_data(batch_size=batch_size)
    for dataframe in gam_results, gam_preds, df:
        for dataset in DATASETS:
            assert dataframe[dataframe["Dataset"]==dataset]["Feature Name"].nunique() == 1, "Expected only one feature name per dataset in the GAM dataframes"

    # 5 columns: 1 empty, 2 subplots, 1 empty — keeps bottom centered
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    gs = gridspec.GridSpec(2, 6, figure=fig)  # 2 rows, 6 cols

    # Top row: 3 same-size axes (each spans 2 cols)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])

    # Bottom row: 2 same-size axes, centered (each spans 2 cols)
    ax4 = fig.add_subplot(gs[1, 1:3])
    ax5 = fig.add_subplot(gs[1, 3:5])

    shift_colors = dict(zip(df["shift"].unique(), sns.color_palette(palette="pastel", n_colors=len(df["shift"].unique()))))
    ax = [ax1, ax2, ax3, ax4, ax5]
    for i, dataset in enumerate(df["Dataset"].unique()):
        feature_name = df[df["Dataset"]==dataset]["Feature Name"].unique()[0]
        subdf_preds = gam_preds[(gam_preds["Dataset"]==dataset)]
        subdf_scatter = df[(df["Dataset"]==dataset)&(df["Dataset"]==dataset) ]
        metric = gam_results[gam_results["Dataset"]==dataset]["MAE"].mean()
        # print(subdf.columns)
        for shift in subdf_scatter["shift"].unique():
            subdf_shift = subdf_scatter[subdf_scatter["shift"]==shift]
            ax[i].scatter(subdf_shift["feature"], subdf_shift["loss"], alpha=1, color=shift_colors[shift])
        correlation = spearmanr(subdf_preds["feature"], subdf_preds["monotonic_pred_loss"])[0]
        # ax[i,j].scatter(subdf["feature"], subdf["loss"], alpha=0.5)
        ax[i].plot(subdf_preds["feature"], subdf_preds["monotonic_pred_loss"], color="red")
        ax[i].fill_between(subdf_preds["feature"], subdf_preds["monotonic_pred_loss_lower"], subdf_preds["monotonic_pred_loss_upper"], color="red", alpha=0.3)
        ax[i].set_title(f"{dataset}|{DSD_PRINT_LUT[feature_name]}: MAE={round(metric,2)}, Rho={round(correlation,2)}")
        ax[i].set_ylim(bottom=0
                         )
        # ax[i].scatter(subdf_train["feature"], subdf_train["loss"], alpha=0.5, label="train")
    plt.tight_layout()
    plt.savefig("figures/gam_fits.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def get_gam_data(load=True):
    for batch_size in BATCH_SIZES:
        total_df = load_all(batch_size=batch_size, shift="", samples=10)
        total_df = total_df[total_df["fold"]!="train"] #exclude training data, to not skew results
        for feature_name in total_df["feature_name"].unique():
            for dataset in total_df["Dataset"].unique():
                metric_data = []
                pred_data = []
                df = total_df[(total_df["Dataset"]==dataset) & (total_df["feature_name"]==feature_name)]
                shifts = df["shift"].unique()

                df = filter_max_loss(df)
                # train_shifts = [s for s in shifts if s not in ["ind_val", "ind_test"]]
                # for regressor_training_shift in train_shift["shift"].unique():

                train = df
                regressor_training_shift ="all"
                if train.empty:
                    print("Skipping due to empty train or test set: ")
                    print("Dataset:", dataset, "Feature:", feature_name, "Shift:", regressor_training_shift)
                    continue

                X_train = train['feature']  # Ensure this is adjusted to your dataset
                y_train = train['loss']

                # Fit the GAM
                # combined_X = np.concatenate((X_train, X_test))
                print("Feature:", feature_name, "Shift:", regressor_training_shift, "Dataset:", dataset)
                assert len(X_train.unique()>1), "Unique values in training set must be greater than 1"
                spr = spearmanr(X_train, y_train)[0]
                print("\t", spr)

                if spr < 0:
                    gam_monotonic = pygam.LinearGAM(fit_intercept=False, constraints="monotonic_dec")
                    gam_monotonic.fit(X_train, y_train)
                else:
                    gam_monotonic = pygam.LinearGAM(fit_intercept=False, constraints="monotonic_inc")
                    gam_monotonic.fit(X_train, y_train)

                XX = gam_monotonic.generate_X_grid(term=0)
                monotonic_grid_preds = gam_monotonic.predict(XX)
                monotonic_grid_preds_conf = gam_monotonic.prediction_intervals(XX, width=0.95)

                for i, (x, ym, ym_c) in enumerate(zip(XX, monotonic_grid_preds, monotonic_grid_preds_conf)):
                    pred_data.append({"feature":x[0],
                                        "monotonic_pred_loss":ym, "monotonic_pred_loss_lower":ym_c[0], "monotonic_pred_loss_upper":ym_c[1],

                                      "Dataset":dataset, "Train Shift":regressor_training_shift, "Feature Name":feature_name, "Batch Size":batch_size})
                for test_shift in shifts:
                    test_shift_df = df[((df["shift"] == test_shift) | (df["shift"] == "ind_test"))]
                    for intensity in test_shift_df["shift_intensity"].unique():
                        test = test_shift_df[test_shift_df["shift_intensity"]==intensity]
                        if test.empty:
                            print("Skipping due to empty test set: ")
                            continue

                        X_test = test['feature']
                        y_test = test['loss']
                        y_test = y_test.apply(lambda x: x if x<= DATASETWISE_RANDOM_LOSS[dataset] else DATASETWISE_RANDOM_LOSS[dataset])  # clip to random guess loss
                        preds_monotonic = gam_monotonic.predict(X_test)
                        preds_monotonic = np.clip(preds_monotonic, 0, DATASETWISE_RANDOM_LOSS[dataset])  # clip to random guess loss
                        for y, yhat in zip(y_test, preds_monotonic):

                            mape_monotonic = np.abs((y-yhat)/y)
                            mae_monotonic = np.abs(y-yhat)
                            data = {"Dataset":dataset, "Feature Name":feature_name, "Train Shift":regressor_training_shift, "Test Shift":test_shift, "Shift Intensity":intensity, "Batch Size":batch_size, "Loss":y, "Prediction":yhat, "MAPE": mape_monotonic, "MAE": mae_monotonic}
                            metric_data.append(data)

                    # preds = gam.predict(X_test)
                    # smape = np.mean(np.abs(preds - y_test) / (np.abs(preds) + np.abs(y_test)))
                    # print(f"Dataset: {dataset} Feature: {feature_name} Shift: {shift}: {smape*100}")
                print("saving results....")
                pred_errors = pd.DataFrame(metric_data)
                pred_df = pd.DataFrame(pred_data)
                pred_df.to_csv(f"gam_data/gam_fits_{dataset}_{feature_name}_{batch_size}.csv")
                print("saved fits!")
                print(len(pred_errors))
                pred_errors.to_csv(f"gam_data/gam_prediction_errors_{dataset}_{feature_name}_{batch_size}.csv")
                print("saved errors!")

def regplots(batch_size):


    df = load_all(batch_size=batch_size, prefix="final_data", shift="", samples=40)
    df = df[df["fold"]!="train"] #exclude training data, to not skew results
    df = df[~df["shift"].isin(["contrast", "brightness", "smear"])]
    for shift in df["shift"].unique():
        if shift not in ["hue", "saltpepper", "noise", "multnoise", "smear", "contrast", "brightness", "ind_val", "ind_test", "train"]:
            print(shift)
            df.replace({shift: "Organic Shift"}, inplace=True)


    df.replace({"normal":"Organic Shift"}, inplace=True)
    hues = df["shift"].unique()
    def plot_threshold(data,color=None, **kwargs):
        threshold = OODDetector(data, ood_val_shift="Organic Shift", threshold_method="val_optimal").threshold
        plt.axvline(threshold, color=color, linestyle="--", label="Threshold")
    # def plot_max_loss(data, color=None, **kwargs):
    #     plt.axhline(DATASETWISE_RANDOM_LOSS[data["Dataset"].unique()[0]], color=color, linestyle="--", label="Random Guessing")
    def custom_scatter(data, **kwargs):
        kwargs.pop("color", None)  # Remove auto-passed color to prevent conflict
        sns.scatterplot(data=data[data["shift"].isin(SYNTHETIC_SHIFTS)], x="feature", y="loss", hue="shift", palette=sns.color_palette(n_colors=len(SYNTHETIC_SHIFTS)+2)[2:], **kwargs)
        sns.scatterplot(data=data[~data["shift"].isin(SYNTHETIC_SHIFTS)], x="feature", y="loss", hue="fold", marker="x",palette=sns.color_palette(n_colors=len(SYNTHETIC_SHIFTS)+2)[:2], alpha=1, **kwargs)
    df.replace(DSD_PRINT_LUT, inplace=True)
    df = filter_max_loss(df)
    g = sns.FacetGrid(df, row="feature_name", col="Dataset", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(custom_scatter)
    g.map_dataframe(plot_threshold)
    g.set_titles(row_template="Feature = {row_name}", col_template="{col_name}")

    for ax, row_val in zip(g.axes[:, 0], g.row_names):
        if row_val == "feature_name":
            ax.set_ylabel("Feature")
        else:
            ax.set_ylabel(row_val)
    for ax in g.axes.flat:
        ax.set_ylim(bottom=0)
    g.add_legend()
    # for ax in g.axes.flat:
        # ax.set_yscale("log")
    #     ax.set_xscale("log")
    plt.savefig(f"figures/regplots_{batch_size}.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def regplot_by_shift():
    print("Loading")
    df = load_all(batch_size=32, shift="", samples=100)
    df = df[df["fold"]!="train"]
    # df = filter_max_loss(df)
    df = df[~df["shift"].isin(["contrast", "brightness","smear"])]

    print("Loaded!")
    for dataset in DATASETS:
        subdf = df[df["Dataset"]==dataset]
        if dataset=="Polyp":
            subdf = subdf[subdf["shift"]!="saltpepper"]
        print(f"Plotting for {dataset}")
        special_intensities = ['InD', 'OoD']
        unique_intensities = subdf["shift_intensity"].unique()
        remaining_intensities = sorted([x for x in unique_intensities if x not in special_intensities])

        base_colors = sns.color_palette(n_colors=2)  # For 'InD' and 'OoD'
        mako_colors = sns.color_palette("mako", len(remaining_intensities))
        full_palette = base_colors + mako_colors
        hue_order = special_intensities + remaining_intensities
        palette = {k: c for k, c in zip(hue_order, full_palette)}
        subdf["shift"] = subdf["shift"].apply(lambda x: x if x in SYNTHETIC_SHIFTS else "Organic")
        def plot_max_loss(data,color=None, **kwargs):
            plt.axhline(DATASETWISE_RANDOM_LOSS[dataset], color=color, linestyle="--", label="Random Guessing")
        subdf.rename(columns=COLUMN_PRINT_LUT, inplace=True)
        subdf.replace(DSD_PRINT_LUT, inplace=True)
        g = sns.FacetGrid(subdf, row="Feature", col="Shift", margin_titles=True, sharex=False, sharey=False)
        g.map_dataframe(sns.scatterplot, x="Feature Value", y="Loss", hue="Shift Intensity", palette=palette, hue_order=hue_order)
        g.map_dataframe(plot_max_loss)
        # g.add_legend()
        plt.savefig(f"figures/regplot_by_shift_{dataset}.png", dpi=300, bbox_inches='tight')
        plt.show()

def filter_max_loss(df):
    # Clip the loss for 'Organic Shift' rows
    df.loc[df["shift"] == "Organic Shift", "loss"] = df[df["shift"] == "Organic Shift"].apply(
        lambda row: min(row["loss"], DATASETWISE_RANDOM_LOSS[row["Dataset"]]), axis=1
    )

    # Filter: keep 'Organic Shift' (already clipped), and non-Organic Shift rows only if their loss is below threshold
    filt = df[
        (df["shift"] == "Organic Shift") |
        ((df["shift"] != "Organic Shift") &
         (df.apply(lambda row: row["loss"] <= DATASETWISE_RANDOM_LOSS[row["Dataset"]], axis=1)))
    ]
    return filt


def plot_intensitywise_kdes():
    df = load_all(batch_size=1, shift="", prefix="final_data")
    g = sns.FacetGrid(df, row="Dataset", col="shift")
    g.map_dataframe(sns.kdeplot, x="feature", y="loss", hue="shift_intensity")
    plt.show()


def compare_gam_errors():
    df = get_gam_data()
    df = df[df["KS"]==False]
    print(df.groupby(["Dataset","train_shift","batch_size", "feature_name"])[[ "regular mape", "MAPE"]].mean())
    plot_df = df[df["train_shift"]=="noise"]
    plot_df = plot_df[plot_df["KS"]==False]
    g = sns.FacetGrid(plot_df, col="Dataset", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.lineplot, x="batch_size", y="MAPE", hue="feature_name", palette="pastel")
    for ax in g.axes.flat:
        ax.set_ylim(0, 1)
        ax.set_xscale("log")
    g.add_legend( title="Feature")
    plt.tight_layout()
    plt.show()
