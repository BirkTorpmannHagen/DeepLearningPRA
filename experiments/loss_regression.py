import numpy
import pandas as pd
import pygam
import seaborn as sns
from matplotlib import pyplot as plt, patches as patches
from scipy.stats import spearmanr

from components import OODDetector
from plots import load_dfs
from utils import BATCH_SIZES, load_all


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
    def bin_Y(group, bins):
        group['feature_bin'] = pd.qcut(group['feature'], bins, labels=False, duplicates='drop')
        return group

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

    g = sns.FacetGrid(df, row="Dataset", col="feature_name", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x="feature", y="loss", hue="shift", hue_order=hues,  alpha=0.5)
    g.map_dataframe(plot_threshold)
    g.add_legend()
    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.set_xscale("log")
    plt.show()


def regplot_by_shift(sample_size, simulate=False):

    df = load_dfs(sample_size=sample_size, simulate=simulate)
    df["ind"]=df["fold"]=="ind"
    df["Shift Severity"]=df["fold"].apply(lambda x: round(float(x.split("_")[1]),2) if "_" in x else x)
    df.rename(columns={"feature_name":"Feature"}, inplace=True)
    df.replace({"typicality":"Typicality", "cross_entropy":"Cross Entropy", "knn":"KNN", "grad_magnitude":"GradNorm", "energy":"Energy", "softmax":"Softmax"}, inplace=True)
    hues = df["Shift Severity"].unique()
    if simulate:
        df = df[df["KS"]==False]
    df.replace({"normal":"Organic Shift"}, inplace=True)
    g = sns.FacetGrid(df, row="Shift", col="Feature", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x="feature", y="loss", hue="Shift Severity", hue_order=hues)
    g.add_legend()
    plt.show()


def plot_variances(df):
    sampled = load_dfs(10, simulate=True, samples=30)
    sampled = sampled[(sampled["KS"] == False) & (sampled["Dataset"] == "NICO")]
    df["Shift Severity"] = df["fold"].apply(
        lambda x: round(float(x.split("_")[1]), 2) if "_" in x else 0 if "ind" in x else 0.35)
    sampled["Shift Severity"] = sampled["fold"].apply(
        lambda x: round(float(x.split("_")[1]), 2) if "_" in x else 0 if "ind" in x else 0.35)

    data_feat = df.groupby(["Dataset", "feature_name", "Shift", "Shift Severity"])["feature"].std().reset_index()
    data_loss = df.groupby(["Dataset", "feature_name", "Shift", "Shift Severity"])["loss"].std().reset_index()
    data_mean_feat = df.groupby(["Dataset", "feature_name", "Shift", "Shift Severity"])["feature"].mean().reset_index()
    data_mean_loss = df.groupby(["Dataset", "feature_name", "Shift", "Shift Severity"])["loss"].mean().reset_index()
    data_feat.rename(columns={"feature": "Feature Variance"}, inplace=True)
    data_loss.rename(columns={"loss": "Loss Variance"}, inplace=True)
    data = pd.merge(data_feat, data_loss, on=["Dataset", "feature_name", "Shift", "Shift Severity"])
    data = pd.merge(data, data_mean_feat, on=["Dataset", "feature_name", "Shift", "Shift Severity"])
    data = pd.merge(data, data_mean_loss, on=["Dataset", "feature_name", "Shift", "Shift Severity"])
    name_map = {"grad_magnitude": "GradNorm", "cross_entropy": "Cross Entropy", "knn": "KNN", "softmax":"Softmax", "typicality":"Typicality", "energy":"Energy"}
    shift_map = {"normal": "Organic Shift", "noise": "Additive Noise", "multnoise":"Multiplicative Noise", "hue":"Hue Shift", "saltpepper": "Salt & Pepper Noise"}
    g = sns.FacetGrid(data, row="Dataset", col="feature_name", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.lineplot, x="Shift Severity", y="Feature Variance", hue="Shift", alpha=0.5)
    g.add_legend()
    plt.show()

    g = sns.FacetGrid(data, row="Dataset", col="feature_name", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.lineplot, x="Shift Severity", y="Loss Variance", hue="Shift", alpha=0.5)
    g.add_legend()
    plt.show()
    data = data[data["Dataset"] == "NICO"]

    fig, ax = plt.subplots(len(data["Shift"].unique()), len(data["feature_name"].unique()), figsize=(20, 20))
    color_map = dict(
        zip(sorted(data["Shift Severity"].unique()), sns.color_palette("magma", len(data["Shift Severity"].unique()))))

    # Add a color column to the sampled DataFrame
    sampled["color"] = sampled["Shift Severity"].map(color_map)

    for i, sev in enumerate(data["Shift"].unique()):
        for j, feature in enumerate(data["feature_name"].unique()):
            subdf = data[(data["Shift"] == sev) & (data["feature_name"] == feature)]
            sampled_subdf = sampled[(sampled["Shift"] == sev) & (sampled["feature_name"] == feature)]

            # Plot scatter points for the sampled data with a black outline
            ax[i, j].scatter(sampled_subdf['feature'], sampled_subdf['loss'],
                             color=sampled_subdf['color'], alpha=0.5,
                             edgecolors='black', linewidth=1.5)

            for row_n, (_, row) in enumerate(subdf.iterrows()):
                color = color_map[row['Shift Severity']]
                ellipse = patches.Ellipse((row['feature'], row['loss']), row['Feature Variance'], row['Loss Variance'],
                                          color=color, alpha=0.3)
                ax[i, j].add_patch(ellipse)

            ax[i, j].set_title(f"{shift_map[sev]}|{name_map[feature]}")
    plt.tight_layout()
    plt.savefig("variance_plot.pdf")
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
