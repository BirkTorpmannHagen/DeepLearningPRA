import numpy as np
import pandas as pd
import pygam
import seaborn as sns
from matplotlib import pyplot as plt, patches as patches
from scipy.stats import spearmanr

from components import OODDetector
from utils import *


def get_gam_data(load=True):
    for sample_size in BATCH_SIZES:
        print("loading")
        total_df = load_all(batch_size=sample_size, shift="", samples=100)
        total_df = total_df[total_df["fold"]!="train"] #exclude training data, to not skew results
        for feature_name in total_df["feature_name"].unique():

            for dataset in total_df["Dataset"].unique():
                data = []
                pred_data = []
                df = total_df[(total_df["Dataset"]==dataset) & (total_df["feature_name"]==feature_name)]
                assert len(df["Dataset"].unique())==1, "Dataset should be unique in the filtered dataframe"
                assert len(df["feature_name"].unique())==1, "Feature name should be unique in the filtered dataframe"
                for shift in df["shift"].unique():
                    if shift in ["ind_val", "ind_test", "train"]:
                        continue
                    train = df[(df["feature_name"]==feature_name) & (df["Dataset"]==dataset) & ((df["shift"]==shift)|(df["shift"]=="ind_val")) ]
                    test = df[(df["feature_name"]==feature_name) & (df["Dataset"]==dataset) & ((df["shift"]==shift)|(df["shift"]!="ind_val"))]
                    if train.empty or test.empty:
                        print("Skipping due to empty train or test set: ")
                        print("Dataset:", dataset, "Feature:", feature_name, "Shift:", shift)

                        continue
                    X_train = train['feature']  # Ensure this is adjusted to your dataset
                    y_train = train['loss']
                    X_test = test['feature']
                    y_test = test['loss']#-train["loss"].mean()
                    # Fit the GAM
                    # combined_X = np.concatenate((X_train, X_test))

                    print("Feature:", feature_name, "Shift:", shift, "Dataset:", dataset)
                    print("\tX:", len(X_train.unique()))
                    print("\tY:", len(y_train.unique()))
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

                                          "Dataset":dataset, "train_shift":shift, "feature_name":feature_name, "sample_size":sample_size})

                    if spr>0.1:
                        plt.plot(XX[:, 0], gam_monotonic.predict(X=XX))
                        plt.scatter(X_train, y_train, c='b', alpha=0.1)
                        plt.scatter(X_test, y_test, c='r', alpha=0.1)
                        plt.show()
                    for test_shift in test["shift"].unique():
                        test_data = test[test["shift"] == test_shift]
                        for sev in test_data["shift_intensity"].unique():
                            test_data_fold = test_data[test_data["shift_intensity"] == sev]

                            preds_monotonic = gam_monotonic.predict(test_data_fold["feature"])
                            mape_monotonic = np.mean(np.abs(preds_monotonic - test_data_fold["loss"]) / np.abs(
                                test_data_fold["loss"]))
                            mae_monotonic = np.mean(np.abs(preds_monotonic - test_data_fold["loss"]))

                            data.append({"Dataset":dataset, "feature_name":feature_name, "train_shift":shift, "test_shift":test_shift, "shift_intensity":sev, "sample_size":sample_size, "monotonic mape": mape_monotonic, "monotonic mae": mae_monotonic})

                    # preds = gam.predict(X_test)
                    # smape = np.mean(np.abs(preds - y_test) / (np.abs(preds) + np.abs(y_test)))
                    # print(f"Dataset: {dataset} Feature: {feature_name} Shift: {shift}: {smape*100}")
                pred_errors = pd.DataFrame(data)
                pred_df = pd.DataFrame(pred_data)
                pred_df.to_csv(f"gam_data/gam_fits_{dataset}_{feature_name}_{sample_size}.csv")
                pred_errors.to_csv(f"gam_data/gam_prediction_errors_{dataset}_{feature_name}_{sample_size}.csv")


def regplots(sample_size):


    df = load_all(batch_size=sample_size, prefix="final_data", shift="", samples=40)
    df = df[df["fold"]!="train"] #exclude training data, to not skew results
    df["ind"]=df["fold"]=="ind"
    df = df[df["shift"]!="smear"]
    for shift in df["shift"].unique():
        if shift not in ["hue", "saltpepper", "noise", "multnoise", "smear", "contrast", "brightness", "ind_val", "ind_test", "train"]:
            print(shift)
            df.replace({shift: "Organic Shift"}, inplace=True)


    df.replace({"normal":"Organic Shift"}, inplace=True)
    hues = df["shift"].unique()
    def plot_threshold(data,color=None, **kwargs):
        threshold = OODDetector(data, ood_val_shift="Organic Shift", threshold_method="val_optimal").threshold
        plt.axvline(threshold, color=color, linestyle="--", label="Threshold")

    def custom_scatter(data, **kwargs):
        kwargs.pop("color", None)  # Remove auto-passed color to prevent conflict
        sns.scatterplot(data=data[data["ood"]], x="feature", y="loss", hue="shift", alpha=0.5, **kwargs)
        sns.scatterplot(data=data[~data["ood"]], x="feature", y="loss", color="black", marker="x", alpha=1, **kwargs, label="InD")
    df.replace(DSD_PRINT_LUT, inplace=True)
    df = df[(df["shift"] == "Organic Shift") |
            ((df["shift"] != "Organic Shift") &
             (df.apply(lambda row: row["loss"] <= DATASETWISE_RANDOM_LOSS[row["Dataset"]], axis=1)))]
    g = sns.FacetGrid(df, row="feature_name", col="Dataset", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(custom_scatter)
    g.map_dataframe(plot_threshold)
    g.set_titles(row_template="Feature = {row_name}", col_template="{col_name}")

    for ax, row_val in zip(g.axes[:, 0], g.row_names):
        if row_val == "feature_name":
            ax.set_ylabel("Feature")
        else:
            ax.set_ylabel(row_val)

    g.add_legend()
    # for ax in g.axes.flat:
        # ax.set_yscale("log")
    #     ax.set_xscale("log")
    plt.savefig(f"figures/regplots_{sample_size}.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def regplot_by_shift():
    print("Loading")
    df = load_all(batch_size=30, shift="", samples=100)
    df = df[df["fold"]!="train"]
    # df = filter_max_loss(df)
    df = df[~df["shift"].isin(["contrast", "brightness","smear"])]
    print("Loaded!")
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
        subdf["shift"] = subdf["shift"].apply(lambda x: x if x in SYNTHETIC_SHIFTS else "Organic")
        def plot_max_loss(data,color=None, **kwargs):
            plt.axhline(DATASETWISE_RANDOM_LOSS[dataset], color=color, linestyle="--", label="Random Guessing")
        g = sns.FacetGrid(subdf, row="shift", col="feature_name", margin_titles=True, sharex=False, sharey=False)
        g.map_dataframe(sns.scatterplot, x="feature", y="loss", hue="shift_intensity", palette=palette, hue_order=hue_order)
        g.map_dataframe(plot_max_loss)
        # g.add_legend()
        plt.savefig(f"figures/regplot_by_shift_{dataset}.png", dpi=300, bbox_inches='tight')
        plt.show()

def filter_max_loss(df):
    return df[(df["shift"] == "Organic Shift") |
            ((df["shift"] != "Organic Shift") &
             (df.apply(lambda row: row["loss"] <= DATASETWISE_RANDOM_LOSS[row["Dataset"]], axis=1)))]


def plot_intensitywise_kdes():
    df = load_all(batch_size=1, shift="", prefix="final_data")
    g = sns.FacetGrid(df, row="Dataset", col="shift")
    g.map_dataframe(sns.kdeplot, x="feature", y="loss", hue="shift_intensity")
    plt.show()


def compare_gam_errors():
    df = get_gam_data()
    df = df[df["KS"]==False]
    print(df.groupby(["Dataset","train_shift","sample_size", "feature_name"])[[ "regular mape", "monotonic mape"]].mean())
    plot_df = df[df["train_shift"]=="noise"]
    plot_df = plot_df[plot_df["KS"]==False]
    g = sns.FacetGrid(plot_df, col="Dataset", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.lineplot, x="sample_size", y="monotonic mape", hue="feature_name", palette="pastel")
    for ax in g.axes.flat:
        ax.set_ylim(0, 1)
        ax.set_xscale("log")
    g.add_legend( title="Feature")
    plt.tight_layout()
    plt.show()
