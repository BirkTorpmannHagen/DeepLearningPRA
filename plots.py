import os

import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os import listdir
from tqdm import tqdm
from os.path import join

from utils import DATASETS, BATCH_SIZES, DSD_PRINT_LUT

pd.set_option("display.precision", 3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering

# def get_all_data(sample_size):
#     return pd.concat([load_old_dfs(sample_size), load_dfs(sample_size=sample_size)])

def merge_by_features(dataframe):
    # Reset index after filtering to avoid duplicate label issues
    dfs = []
    feature_names = dataframe["feature_name"].unique()
    base = dataframe[dataframe["feature_name"] == feature_names[0]].copy()
    base.rename({"feature": feature_names[0]}, axis=1, inplace=True)
    base.drop("feature_name", axis=1, inplace=True)
    base.reset_index(drop=True, inplace=True)  # Reset index here

    for feature_name in feature_names[1:]:  # Start from second element
        subdf = dataframe[dataframe["feature_name"] == feature_name].copy()
        subdf.rename({"feature": feature_name}, axis=1, inplace=True)
        subdf.drop("feature_name", axis=1, inplace=True)
        subdf.reset_index(drop=True, inplace=True)  # Reset index here

        base[feature_name] = subdf[feature_name]

    return base

    # dfs = []
    # for feature_name in dataframe["feature_name"].unique():
    #     subdf = dataframe[dataframe["feature_name"]==feature_name].copy()
    #     subdf.rename({"feature":feature_name}, axis=1, inplace=True)
    #     dfs.append(subdf)
    # [df.drop("feature_name", axis=1, inplace=True) for df in dfs]
    # for i, df in enumerate(dfs):
    #     # Identify the unique columns that are not part of the merge keys
    #     unique_cols = [col for col in df.columns if col not in ["Dataset", "Shift", "fold", "loss", "KS"]]
    #     # Rename these columns to ensure they are unique
    #     df.rename(columns={col: f"{col}" for col in unique_cols}, inplace=True)
    # dfs = [df.set_index(["fold", "Shift", "level_4", "Dataset", "loss", "KS"]).reset_index() for df in dfs]
    #
    # merged = pd.concat(dfs, axis=1)
    #
    # merged = merged.loc[:, ~merged.columns.duplicated()].copy()
    # merged.reset_index(inplace=True)
    # print(merged.head(10))
    # merged.drop("level_4", axis=1, inplace=True) #wtf pandas
    # # merged.drop("index", axis=1, inplace=True)
    # return merged


def simulate_sampling(df, samples, sample_size):
    def sample_loss_feature(group, n_samples, n_size):
        samples = []
        for i in range(n_samples):
            sample = group.sample(n=n_size, replace=True)  # Sampling with replacement
            mean_loss = sample['loss'].mean()
            mean_feature = sample['feature'].mean()
            samples.append({'loss': mean_loss, 'feature': mean_feature, "KS":False})
        return pd.DataFrame(samples)
        # Return a DataFrame of means with the original group keys
    cols = list(df.columns)
    cols.remove("loss")
    cols.remove("feature")
    cols.remove("Unnamed: 0")
    return df.groupby(cols).apply(sample_loss_feature, samples, sample_size).reset_index()




def load_dfs(sample_size, path="single_data/", simulate=False, samples=100):
    dfs = []
    for fname in tqdm(listdir(path)):
        if "ks" not in fname and not any(shift in fname for shift in ["dropout", "saturation", "brightness", "smear", "odin"]):
            df = pd.read_csv(join(path,fname))
            dataset= fname.split("_")[0]
            shift = fname.split("_")[1]
            df["Shift"]=shift
            df["Dataset"]=dataset
            if simulate:
                df=simulate_sampling(df,samples,sample_size)
            else:
                df["KS"] = False
            dfs.append(df)
    merged = pd.concat(dfs)
    merged = merged[merged["fold"]!="train"]
    merged["Shift Severity"] = merged["fold"].apply(
        lambda x: round(float(x.split("_")[1]), 2) if "_" in x else 0 if "ind" in x else "ood")
    return merged

def compare_ks_vs_no_ks(sample_size):

    df = load_dfs(sample_size=sample_size, simulate=True)
    df["ind"] = df["fold"] == "ind"
    df.replace({"normal": "Organic Shift"}, inplace=True)
    df.loc[df["fold"] == "ind", "Shift"] = "ind"
    hues = df["Shift"].unique()
    df = df[df["Dataset"] == "NICO"]
    g = sns.FacetGrid(df, row="KS", col="feature_name", margin_titles=True, sharex=False, sharey=True)
    g.map_dataframe(sns.scatterplot, x="feature", y="loss", hue="Shift", hue_order=hues, alpha=0.5)
    g.add_legend()
    plt.show()


def plot_tpr_tnr_sensitivity():
    # Load and preprocess data
    dfs = []
    for filename in os.listdir("pra_data"):
        df = pd.read_csv(f"pra_data/{filename}")
        dfs.append(df)
    df = pd.concat(dfs)

    print(df.head(10))
    df = df[df["ba"] >= 0.5]
    print(df.groupby(["ba", "Tree", "val_set", "test_set"])[["Rate Error", "Accuracy Error"]].mean())
    # input()
    df["ba"] = round(df["ba"], 2)
    df["rate"] = round(df["rate"], 2)
    # Prepare data for heatmaps
    facet = df.groupby(["ba", "rate", "val_set", "test_set"])[["Accuracy Error"]].mean().reset_index()
    # facet = facet.pivot(index=["val_set", "test_set"], columns="rate", values="Error")

    # Define heatmap function
    def draw_heatmap(data, **kws):
        # Extract numeric data for the heatmap
        heatmap_data = data.pivot(index="ba", columns="rate", values="Accuracy Error")
        heatmap_data = heatmap_data.loc[::-1]

        sns.heatmap(heatmap_data, **kws, cmap="mako", vmin=0, vmax=(df["ind_acc"]-df["ood_val_acc"]).mean())

    # Create FacetGrid and plot heatmaps
    g = sns.FacetGrid(facet.reset_index(), col="test_set", row="val_set", col_order=[CVCCLINIC, ETISLARIB, ENDOCV], row_order=[CVCCLINIC, ETISLARIB, ENDOCV], margin_titles=True)
    g.map_dataframe(draw_heatmap)
    plt.savefig("cross_validated_accuracy_estimation_error.eps")
    plt.show()

    # Additional analysis and plotting
    print(df[df["ba"] == 1].groupby(["ba", "rate"])[["E[f(x)=y]", "Accuracy Error"]].mean().reset_index())
    df = df.groupby(["ba", "rate"])["Accuracy Error"].mean().reset_index()
    pivot_table = df.pivot(index="ba", columns="rate", values="Accuracy Error")
    pivot_table = pivot_table.loc[::-1]
    sns.heatmap(pivot_table, cmap="mako")
    plt.legend()
    plt.savefig("tpr_tnr_sensitivity.eps")
    plt.show()

def plot_dsd_acc_errors():
    dfs = []
    for dataset in DATASETS:
        for batch_size in BATCH_SIZES:
            try:
                df = pd.read_csv(f"pra_data/dsd_results_{dataset}_{batch_size}.csv")
                # best_guess = (df["ind_acc"].mean() + df["ood_val_acc"].mean()) / 2
                best_guess = df["ind_acc"].mean()
                df["Dataset"]=dataset
                df["batch_size"]=batch_size
                df["lineplot_idx"]=BATCH_SIZES.index(batch_size)
                df["lineplot_rate_idx"] = pd.factorize(df['rate'])[0]
                print(dataset, " : ", best_guess)
                df["best_guess_error"] = np.abs(df["Accuracy"] - best_guess)
                dfs.append(df)
            except:
                print(f"No data found for {dataset} with batch size {batch_size}")
    df = pd.concat(dfs)
    df.replace(DSD_PRINT_LUT, inplace=True)
    df = df[df["Tree"]=="Base Tree"]
    # df = df[df["batch_size"]==1]
    g = sns.FacetGrid(df, col="Dataset", sharey=False, col_wrap=3)
    g.map_dataframe(sns.boxplot, x="rate", y="Accuracy Error", hue="test_set", showfliers=False, palette=sns.color_palette())
    g.map_dataframe(sns.lineplot, x="lineplot_rate_idx", y="best_guess_error", hue="test_set", linestyle="--", marker="o", palette=sns.color_palette(), legend=False)
    sorted_datasets = sorted(df["Dataset"].unique())

    for ax, dataset in zip(g.axes.flat, sorted_datasets):
        ax.set_title(dataset)
        ax.set_xlabel("P(E)")
        ax.set_ylabel("Accuracy Error")
        ax.set_xticklabels(df["rate"].unique())
        ax.set_xticks(range(len(df["rate"].unique())))
        ax.legend(title="Test Set", ncols=3, fontsize=8)
        ax.set_yscale("log")

    num_plots = len(g.axes.flat)
    num_cols = 3  # Top row columns
    last_row_plots = num_plots % num_cols

    if last_row_plots > 0:
        # Get figure width
        fig_width = g.fig.get_size_inches()[0]

        # Compute total space occupied by the last row's plots
        last_row_width = (fig_width / num_cols) * last_row_plots

        # Compute left padding to center the row
        left_padding = (fig_width - last_row_width) / 2

        # Adjust position of the last row's plots
        for ax in g.axes[-last_row_plots:]:
            pos = ax.get_position()
            ax.set_position([pos.x0 + left_padding / fig_width, pos.y0, pos.width, pos.height])
    plt.savefig("dsd_acc_erorrs_by_rate.pdf")
    plt.show()


    g = sns.FacetGrid(df, col="Dataset", height=3, aspect=1.5, col_wrap=3, sharey=False)

    df = df[df["val_set"]!=df["test_set"]]
    g.map_dataframe(sns.boxplot, x="batch_size", y="Accuracy Error", hue="test_set", showfliers=False, palette=sns.color_palette())
    g.map_dataframe(sns.lineplot, x="lineplot_idx", y="best_guess_error", hue="test_set", linestyle="--", marker="o", palette=sns.color_palette(), legend=False)
    # g.map_dataframe(sns.lineplot, x="batch_size", y="Accuracy Error", hue="test_set")
    for ax, dataset in zip(g.axes.flat, sorted_datasets):
        ax.set_title(dataset)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Accuracy Error")
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xticks(range(len(BATCH_SIZES)))
        ax.legend(title="Test Set", ncols=3, fontsize=8)
        ax.set_yscale("log")

    num_plots = len(g.axes.flat)
    num_cols = 3  # Top row columns
    last_row_plots = num_plots % num_cols

    if last_row_plots > 0:
        # Get figure width
        fig_width = g.fig.get_size_inches()[0]

        # Compute total space occupied by the last row's plots
        last_row_width = (fig_width / num_cols) * last_row_plots

        # Compute left padding to center the row
        left_padding = (fig_width - last_row_width) / 2

        # Adjust position of the last row's plots
        for ax in g.axes[-last_row_plots:]:
            pos = ax.get_position()
            ax.set_position([pos.x0 + left_padding / fig_width, pos.y0, pos.width, pos.height])
    plt.savefig("dsd_acc_errors.pdf")
    plt.show()

def plot_sensitivity_errors():
    dfs = []
    for dataset in DATASETS:
        try:
            df = pd.read_csv(f"pra_data/{dataset}_sensitivity_results.csv")
            best_guess = (df["ind_acc"].mean() + df["ood_val_acc"].mean()) / 2
            print(dataset, " : ", best_guess)
            df["Dataset"]=dataset
            df["best_guess_error"] = np.abs(df["Accuracy"] - best_guess)
            dfs.append(df)
        except:
            print(f"No data found for {dataset}")
    df = pd.concat(dfs)
    df = df[df["Tree"]=="Base Tree"]
    df = df[df["val_set"]!=df["test_set"]]
    df["rate"]=round(df["rate"], 2)
    df["ba"]=round(df["ba"], 2)
    df.replace(DSD_PRINT_LUT, inplace=True)
    print(df.columns)
    df = df.groupby(["Dataset", "rate", "ba"])[["Accuracy Error", "ind_acc", "ood_val_acc"]].mean().reset_index()
    df.rename(columns={"rate":"$P(E)$", "ba":"$p(D_{e}(x)=E)$"}, inplace=True)
    g = sns.FacetGrid(df, col="Dataset", col_wrap=3)
    #sort by ba increasing order
    def plot_heatmap(data, **kws):
        heatmap_data = data.pivot(index="$p(D_{e}(x)=E)$", columns="$P(E)$", values="Accuracy Error")
        heatmap_data = heatmap_data.loc[::-1] #higher ba is up
        sns.heatmap(heatmap_data, **kws, cmap="mako", vmin=0, vmax=(df["ind_acc"]-df["ood_val_acc"]).mean())
    g.map_dataframe(plot_heatmap)


    num_plots = len(g.axes.flat)
    num_cols = 3  # Top row columns
    last_row_plots = num_plots % num_cols

    if last_row_plots > 0:
        fig_width = g.fig.get_size_inches()[0]
        last_row_width = (fig_width / num_cols) * last_row_plots
        left_padding = (fig_width - last_row_width) / 2

        for ax in g.axes[-last_row_plots:]:
            pos = ax.get_position()
            ax.set_position([pos.x0 + left_padding / fig_width, pos.y0, pos.width, pos.height])
            cbar = ax.collections[0].colorbar
            cbar.ax.set_position([cbar.ax.get_position().x0 + left_padding / fig_width, cbar.ax.get_position().y0, cbar.ax.get_position().width, cbar.ax.get_position().height])
    plt.savefig("sensitivity_errors.pdf")
    plt.show()


def get_risk_tables():
    df = pd.read_csv("datasetwise_risk.csv")
    df.replace(DSD_PRINT_LUT, inplace=True)
    df.replace({"ind_test": "Kvasir"}, inplace=True)
    df_base = df[df["Tree"]=="Base Tree"]
    df_dsd = df[df["Tree"]!="Base Tree"]

    print(df_base.groupby(["Model", "Dataset"])["True Risk"].mean())
    print(df_dsd.groupby(["Model", "DSD", "Dataset"])["True Risk"].mean())


if __name__ == '__main__':
    pass
    # get_risk_tables()

    # data = load_pra_df(dataset_name="Polyp", feature_name="knn", batch_size=1, samples=1000)
    # plot_dsd_acc_errors()
    # plot_sensitivity_errors()
    # plot_loss_distributions(data)
    # plot_tpr_tnr_sensitivity()
    # plot_sample_size_effect()
    # show_thresholding_problems()
    # test = pd.read_csv("gam_results.csv")
    # print(test.groupby(["Dataset",  "feature_name"])["mape"].mean())
    # print(test.groupby(["Dataset", "train_shift",  "feature_name"])[["mape", "mae"]].mean())

    # print("Starting")
    # plot_variances(load_dfs(100, simulate=False))
    # build_gam_for_all(load_dfs(100, simulate=True), simulate=True)

    # get_gam_data(load=True)
    # find_best_train_shift()
    # gam_fits(KS=False)
    # compare_gam_errors()
    # build_gam_for_each(dfs)
    # regplots(100, simulate=True)
    # compare_ks_vs_no_ks(500)
    # correleations(100, simulate=True)
    # correleations(100, simulate=True)
    # regplot_by_shift(10, simulate=True)
    # regplot_by_shift(100, simulate=True)
    # correleations(100, simulate=False)
    # sanity_check()
    # classification_metrics(simulate=False)
    # correleations(100, simulate=True)
    # correleations(sample_size=100)
    # quantize_values_and_plot_kdes()
