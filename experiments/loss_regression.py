import itertools

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

def get_best_gam_data(batch_size=32, prefix="coarse_data"):
    df = load_all(batch_size, shift="", prefix=prefix, samples=20)
    df = df[df["fold"] != "train"]
    # df = filter_max_loss(df)
    all_gams = []

    for dataset, feature_name in itertools.product(DATASETS, DSDS):
        try:
            gam_result = pd.read_csv(f"gam_data/gam_prediction_errors_{dataset}_{feature_name}_{batch_size}.csv")
            all_gams.append(gam_result)
        except FileNotFoundError:
            continue
    all_gam_results = pd.concat(all_gams, ignore_index=True)

    all_gam_preds = []
    for dataset, feature_name in itertools.product(DATASETS, DSDS):
        try:
            gam_pred = pd.read_csv(f"gam_data/gam_fits_{dataset}_{feature_name}_{batch_size}.csv")
            all_gam_preds.append(gam_pred)
        except FileNotFoundError:
            continue
    all_gam_preds = pd.concat(all_gam_preds, ignore_index=True)

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
                      height=2.5, col_wrap=3)

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
    # g.add_legend()
    g.set_axis_labels("Loss (bin midpoint)", "Mean MAE")
    # g.fig.subplots_adjust(bottom=0.16)

    for i, ax in enumerate(g.axes.flat):
        ax.set_yscale("log")
    plt.tight_layout()
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
        ax[i].set_title(f"{dataset}|{DSD_PRINT_LUT[feature_name]}: MAE={round(metric,2)}")
        ax[i].set_ylim(bottom=0
                         )
        # ax[i].scatter(subdf_train["feature"], subdf_train["loss"], alpha=0.5, label="train")
    plt.tight_layout()
    plt.savefig("figures/gam_fits.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def plot_gam_errors_by_batch_size():
    all_gam_results = []
    all_gam_preds = []
    all_dfs = []
    for batch_size in BATCH_SIZES:
        gam_results, gam_preds, df = get_best_gam_data(batch_size=batch_size)
        all_gam_results.append(gam_results)
        all_gam_preds.append(gam_preds)
        all_dfs.append(df)
    all_gam_results = pd.concat(all_gam_results)
    all_gam_preds = pd.concat(all_gam_preds)
    all_dfs = pd.concat(all_dfs)
    def plot_baseline(data, **kwargs):
        dataset = data["Dataset"].iloc[0]
        df_dataset = all_dfs[all_dfs["Dataset"]==dataset]
        print(df_dataset.columns)

        ind_loss = df_dataset[df_dataset["fold"]=="ind_test"].groupby("batch_size")["loss"].mean().reset_index()
        df = df_dataset.groupby("batch_size")["loss"].mean().reset_index()
        merged = df.merge(ind_loss, on="batch_size", suffixes=("", "_ind_val"))
        print(merged)
        merged["Baseline Error"] = np.abs(merged["loss"] - merged["loss_ind_val"])
        sns.lineplot(data=merged, x=range(5), y="Baseline Error", color="red", label="Baseline Error", ax=plt.gca())
        plt.ylim(0, np.percentile(df_dataset["loss"],95))


    g = sns.FacetGrid(all_gam_results, col="Dataset", margin_titles=True, sharex=False, sharey=False, col_wrap=3, height=2.5)
    g.map_dataframe(sns.boxenplot, x="Batch Size", y="MAE", hue="Batch Size", palette=sns.color_palette())
    g.map_dataframe(plot_baseline)
    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-3)
    fig_width = g.fig.get_size_inches()[0]
    num_plots = len(g.axes.flat)
    num_cols = 3  # Top row columns
    last_row_plots = num_plots % num_cols
    # Compute total space occupied by the last row's plots
    last_row_width = (fig_width / num_cols) * last_row_plots

    # Compute left padding to center the row
    left_padding = (fig_width - last_row_width) / 2

    # Adjust position of the last row's plots
    for ax in g.axes[-last_row_plots:]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + left_padding / fig_width, pos.y0, pos.width, pos.height])

    plt.savefig("figures/batch_size_regression_errors.pdf")
    plt.show()


def get_gam_data():
    with tqdm(total = len(DATASETS) * len(DSDS) * len(BATCH_SIZES), desc="Loading GAM data") as pbar:
        for batch_size in BATCH_SIZES:
            print("working on batch size ", batch_size)
            total_df = load_all(batch_size=batch_size, shift="", samples=100)
            total_df = total_df[total_df["fold"]!="train"] #exclude training data, to not skew results
            total_df_random_grouping = []

            for strat in np.linspace(0,1, 11):
                df = load_all(batch_size=batch_size, shift="", samples=10, groupbyfolds=False, stratisfication=strat)
                df["Stratification Rate"]= strat
                total_df_random_grouping.append(df)
            total_df_random_grouping = pd.concat(total_df_random_grouping, ignore_index=True)

            for feature_name in total_df["feature_name"].unique():
                for dataset in total_df["Dataset"].unique():
                    metric_data = []
                    ungrouped_data = []
                    pred_data = []

                    df = total_df[(total_df["Dataset"]==dataset) & (total_df["feature_name"]==feature_name)]
                    # shifts = df["shift"].unique()

                    # df = filter_max_loss(df)
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
                    spr = spearmanr(X_train, y_train)[0]

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

                    pred_df = pd.DataFrame(pred_data)
                    pred_df.to_csv(f"gam_data/gam_fits_{dataset}_{feature_name}_{batch_size}.csv")
                    print("saved fits!")
                    shifts = df["shift"].unique()
                    for test_shift in shifts:
                        test_shift_df = df[((df["shift"] == test_shift) | (df["shift"] == "ind_test"))]
                        for intensity in test_shift_df["shift_intensity"].unique():
                            test = test_shift_df[test_shift_df["shift_intensity"]==intensity]
                            if test.empty:
                                print("Skipping due to empty test set: ")
                                continue

                            X_test = test['feature']
                            y_test = test['loss']
                            # y_test = y_test.apply(lambda x: x if x<= DATASETWISE_RANDOM_LOSS[dataset] else DATASETWISE_RANDOM_LOSS[dataset])  # clip to random guess loss
                            preds_monotonic = gam_monotonic.predict(X_test)
                            # preds_monotonic = np.clip(preds_monotonic, 0, DATASETWISE_RANDOM_LOSS[dataset])  # clip to random guess loss
                            for y, yhat in zip(y_test, preds_monotonic):
                                mape_monotonic = np.abs((y-yhat)/y)
                                mae_monotonic = np.abs(y-yhat)
                                data = {"Dataset":dataset, "Feature Name":feature_name, "Train Shift":regressor_training_shift,
                                        "Test Shift":test_shift, "Shift Intensity":intensity, "Batch Size":batch_size,
                                        "Loss":y, "Prediction":yhat, "MAPE": mape_monotonic, "MAE": mae_monotonic,
                                        "Baseline MAE": np.abs(y - df[df["shift"] == "ind_test"]["loss"].mean())
}  # baseline is the mean loss on ind_test}
                                metric_data.append(data)

                    pred_errors = pd.DataFrame(metric_data)
                    pred_errors.to_csv(f"gam_data/gam_prediction_errors_{dataset}_{feature_name}_{batch_size}.csv")
                    print("saved errors!")


                    #grouped
                    if batch_size!=1:
                        subdf_ungrouped = total_df_random_grouping[
                            (total_df_random_grouping["Dataset"] == dataset) & (total_df_random_grouping["feature_name"] == feature_name)
                        ]
                        for strat in subdf_ungrouped["Stratification Rate"].unique():
                            X_test = subdf_ungrouped[subdf_ungrouped["Stratification Rate"]==strat]['feature']
                            y_test = subdf_ungrouped[subdf_ungrouped["Stratification Rate"]==strat]["loss"]#.apply(lambda x: x if x <= DATASETWISE_RANDOM_LOSS[dataset] else DATASETWISE_RANDOM_LOSS[dataset])  # clip to random guess loss
                            preds_monotonic = gam_monotonic.predict(X_test)
                            # preds_monotonic = np.clip(preds_monotonic, 0,
                            #                           DATASETWISE_RANDOM_LOSS[dataset])  # clip to random guess loss
                            for y, yhat in zip(y_test, preds_monotonic):
                                mape_monotonic = np.abs((y - yhat) / y)
                                mae_monotonic = np.abs(y - yhat)
                                data = {"Dataset": dataset, "Feature Name": feature_name, "Train Shift": regressor_training_shift,
                                         "Batch Size": batch_size, "Loss": y, "Stratification Rate": strat,
                                        "Prediction": yhat, "MAPE": mape_monotonic, "MAE": mae_monotonic}
                                ungrouped_data.append(data)

                        ungrouped_data = pd.DataFrame(ungrouped_data)
                        ungrouped_data.to_csv(f"gam_data/gam_ungrouped_{dataset}_{feature_name}_{batch_size}.csv")
                        print("saved ungrouped data!")

                    pbar.update(1)

def assess_ungrouped_regression_errors():
    dfs = []
    baseline = []
    # with tqdm(total=len(DATASETS) * len(DSDS) * len(BATCH_SIZES[1:]), desc="Loading GAM data") as pbar:
    #     for dataset, feature_name, batch_size in itertools.product(DATASETS, DSDS, BATCH_SIZES[1:]):
    #         try:
    #             df = pd.read_csv(f"gam_data/gam_ungrouped_{dataset}_{feature_name}_{batch_size}.csv")
    #             dfs.append(df)
    #             baseline_df = load_pra_df(dataset, feature_name, batch_size=batch_size, shift="", samples=100)
    #             ind_loss = baseline_df[baseline_df["fold"] == "ind_val"]["loss"].mean()
    #             # guard against zero baseline for MAPE
    #             if ind_loss == 0:
    #                 b_mape = np.nan
    #             else:
    #                 b_mape = np.abs((baseline_df["loss"] - ind_loss) / ind_loss).mean()
    #
    #             baseline.append({
    #                 "Dataset": dataset,
    #                 "Feature Name": feature_name,
    #                 "Batch Size": batch_size,
    #                 "Baseline MAE": np.abs(baseline_df["loss"] - ind_loss).mean(),
    #                 "Baseline MAPE": b_mape,
    #             })
    #         except FileNotFoundError:
    #             print(f"File not found for {dataset}, {feature_name}, {batch_size}")
    #         finally:
    #             pbar.update(1)
    #
    # baseline_df = pd.DataFrame(baseline)
    # dfs = pd.concat(dfs, ignore_index=True)
    #
    # # Means only used for ranking; keep for annotation
    # means = (
    #     dfs.groupby(["Batch Size", "Dataset", "Feature Name"])[["MAE", "MAPE"]]
    #        .mean()
    #        .reset_index()
    #        .rename(columns={"MAE": "MAE_mean", "MAPE": "MAPE_mean"})
    # )
    #
    # # Pick best Feature Name per (Dataset, Batch Size) by lowest MAE_mean
    # best_idx = means.groupby(["Dataset", "Batch Size"])["MAE_mean"].idxmin()
    # best_meta = means.loc[best_idx, ["Dataset", "Batch Size", "Feature Name"]]
    #
    # # Pull ALL original rows for the best combos
    # best_rows = dfs.merge(best_meta, on=["Dataset", "Batch Size", "Feature Name"], how="inner")
    #
    # # Annotate with group means and baselines
    # best_rows = (
    #     best_rows
    #     .merge(means, on=["Dataset", "Batch Size", "Feature Name"], how="left")
    #     .merge(baseline_df, on=["Dataset", "Batch Size", "Feature Name"], how="left")
    #     .sort_values(["Dataset", "Batch Size", "Feature Name"])
    #     .reset_index(drop=True)
    # )
    # feature_names = best_rows["Feature Name"].unique()
    # palette = dict(zip(feature_names, sns.color_palette(n_colors=len(feature_names))))

    order = BATCH_SIZES[1:]

    def plot_baseline(data, **kwargs):
        ax = plt.gca()
        bs = np.sort(data["Batch Size"].unique())
        s = data.groupby("Batch Size")["Baseline MAE"].mean()
        y = [s.get(b, np.nan) for b in bs]
        x = pd.Index(order).get_indexer(bs)
        mask = ~np.isnan(y)
        ax.plot(x[mask], np.asarray(y)[mask], linestyle="--", marker="o", color="red", label="Baseline")

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
    #
    # # FacetGrid with col_wrap=3 for 3 plots per row
    # best_rows.to_csv("gam_data/ungrouped_regression_errors_summary.csv", index=False)
    best_rows = pd.read_csv("gam_data/ungrouped_regression_errors_summary.csv")
    g = sns.FacetGrid(
        best_rows,
        col="Dataset",
        margin_titles=True,
        sharex=True, sharey=False,
        col_wrap=3,
        height=2.5
    )

    g.map_dataframe(
        sns.boxenplot,
        x="Batch Size", y="MAE",
        order=order, palette=sns.color_palette()
    )
    g.map_dataframe(plot_baseline)
    # Legend in one row at the bottom

    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-3)

    g.add_legend(title="Feature Name", bbox_to_anchor=(0.8, 0.5), loc="upper left")
    fig_width = g.fig.get_size_inches()[0]
    num_plots = len(g.axes.flat)
    num_cols = 3  # Top row columns
    last_row_plots = num_plots % num_cols
    # Compute total space occupied by the last row's plots
    last_row_width = (fig_width / num_cols) * last_row_plots

    # Compute left padding to center the row
    left_padding = (fig_width - last_row_width) / 2

    # Adjust position of the last row's plots
    for ax in g.axes[-last_row_plots:]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + left_padding / fig_width, pos.y0, pos.width, pos.height])

    plt.savefig("figures/ungrouped_regression_errors.pdf", dpi=300, bbox_inches='tight')
    plt.show()







def regplots(batch_size):
    df = load_all(batch_size=batch_size, prefix="fine_data", shift="", samples=3)
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
    def plot_max_loss(data, color=None, **kwargs):
        plt.axhline(DATASETWISE_RANDOM_LOSS[data["Dataset"].unique()[0]], color=color, linestyle="--", label="Random Guessing")
    def custom_scatter(data, **kwargs):
        kwargs.pop("color", None)  # Remove auto-passed color to prevent conflict
        sns.scatterplot(data=data[data["Organic"]==False], x="feature", y="loss", hue="shift", s=12, palette=sns.color_palette(n_colors=len(SYNTHETIC_SHIFTS)+2)[2:], **kwargs)
        sns.scatterplot(data=data[data["Organic"]==True], x="feature", y="loss", hue="ood_label",s=20, marker="x",palette=sns.color_palette(n_colors=len(SYNTHETIC_SHIFTS)+2)[:2], alpha=1, **kwargs)
    df.replace(DSD_PRINT_LUT, inplace=True)
    # df = filter_max_loss(df)
    df["ood_label"] = df["ood"].apply(lambda x: "OoD" if x else "InD")
    # sns.set_context("notebook", font_scale=2)
    print(df.columns)
    g = sns.FacetGrid(df, row="feature_name", col="Dataset", margin_titles=True, sharex=False, sharey=False, height=1.5, aspect=1)
    g.map_dataframe(custom_scatter)
    # g.map_dataframe(plot_threshold)
    g.map_dataframe(plot_max_loss, color="red")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")


    for ax in g.axes.flat:
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        # ax.set_xscale("log")
        ax.set_yticks([])
        ax.set_xticks([])
        # Hide x labels/ticks for all but bottom row
        if ax not in g.axes[-1, :]:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        # Hide y labels/ticks for all but left column
        if ax not in g.axes[:, 0]:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    # g.add_legend(
    #     title="Shift",
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.05),  # centered below plot
    #     ncol=3  # number of columns in legend
    # )

    # plt.tight_layout(pad=2)
    # for ax in g.axes.flat:ood_accuracy_vs_pred_accuacy_plot
        # ax.set_yscale("log")
    #     ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(f"figures/regplots_{batch_size}.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def regplot_by_shift():
    print("Loading")
    df = load_all(batch_size=32, shift="", samples=30)
    df = df[df["fold"]!="train"]
    # df = filter_max_loss(df)

    df = df[~df["shift"].isin(["contrast", "brightness","smear"])]
    print(df["shift"].unique())
    df["shift"] = df["shift"].apply(lambda x: x if x in SYNTHETIC_SHIFTS else "Organic")
    print(df["shift"].unique())
    df.replace(DSD_PRINT_LUT, inplace=True)
    df.replace(SHIFT_PRINT_LUT, inplace=True)

    print(df["shift"].unique())


    for dataset in DATASETS:
        subdf = df[df["Dataset"]==dataset]
        print(f"Plotting for {dataset}")
        special_intensities = ['InD', 'OoD']
        unique_intensities = subdf["shift_intensity"].unique()
        remaining_intensities = sorted([x for x in unique_intensities if x not in special_intensities])

        base_colors = sns.color_palette(n_colors=2)  # For 'InD' and 'OoD'
        mako_colors = sns.color_palette("mako", len(remaining_intensities))
        full_palette = base_colors + mako_colors
        hue_order = special_intensities + remaining_intensities
        palette = {k: c for k, c in zip(hue_order, full_palette)}
        def plot_max_loss(data,color=None, **kwargs):
            plt.axhline(DATASETWISE_RANDOM_LOSS[dataset], color=color, linestyle="--", label="Random Guessing")

        g = sns.FacetGrid(subdf, row="feature_name", col="shift", margin_titles=True, sharex=False, sharey=False, height=1.75, aspect=1)
        g.map_dataframe(sns.scatterplot, x="feature", y="loss", hue="shift_intensity", palette=palette, hue_order=hue_order, s=30 )
        g.map_dataframe(plot_max_loss)
        g.set_titles(row_template="{row_name}", col_template="{col_name}", size=14)

        # g.add_legend()
        for ax in g.axes.flat:
            ax.set_xlabel("Feature Value", size=12)
            ax.set_yticks([])
            ax.set_xticks([])
        plt.tight_layout()
        plt.savefig(f"figures/regplot_by_shift_{dataset}.pdf", dpi=300, bbox_inches='tight')
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
    df = load_all(batch_size=1, shift="", prefix="coarse_data")
    g = sns.FacetGrid(df, row="Dataset", col="shift")
    g.map_dataframe(sns.kdeplot, x="feature", y="loss", hue="shift_intensity")
    plt.show()

