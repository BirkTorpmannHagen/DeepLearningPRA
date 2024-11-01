import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch.nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from os import listdir
from tqdm import tqdm
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from os.path import join
import pygam
import matplotlib.patches as patches

from scipy.stats import spearmanr, pearsonr, kendalltau
from metrics import ba, spearman, pearson, kendall, auroc, variance

pd.set_option("display.precision", 3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
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

def build_gam_for_all(data, simulate=False):
    if simulate:
        df = data[data["KS"]==True]
    else:
        df = data
    df = merge_by_features(df)

    # print(df["Dataset"])
    # input()
    for dataset in df["Dataset"].unique():
        for shift in df["Shift"].unique():
            X_train = df[(df["Dataset"]==dataset) & (df["Shift"]==shift)]
            X_test = df[(df["Dataset"]==dataset) & (df["Shift"]!=shift)]

            y_train = X_train["loss"].to_numpy()
            y_test = X_test["loss"].to_numpy()

            X_train = X_train.drop(columns=["level_4", "Dataset", "Shift", "fold", "loss", "KS"]).to_numpy()
            X_test = X_test.drop(columns=["level_4", "Dataset", "Shift", "fold", "loss", "KS"]).to_numpy()

            # scaler = StandardScaler()
            # scaler.fit_transform(X_train)
            # scaler.transform(X_test)
            gam = pygam.LinearGAM()
            gam.fit(X_train, y_train)
            preds = gam.predict(X_test)
            mape = np.mean(np.abs(preds - y_test) / np.abs(y_test))

            fig, axs = plt.subplots(1, X_train.shape[1], figsize=(20, 5))
            for i, ax in enumerate(axs):
                XX = gam.generate_X_grid(term=i)
                ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
                ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
                ax.scatter(X_train[:, i], y_train, c='b', alpha=0.1)
                ax.scatter(X_test[:, i], y_test, c='r', alpha=0.1)
            fig.suptitle(f"Dataset: {dataset} Shift: {shift} MAPE: {mape}")
            plt.show()
            # print(f"Train: {shift}, {mape}")



def get_gam_data(load=True):
    if load:
        try:
            return pd.read_csv("gam_results_ks.csv")
        except FileNotFoundError:
            get_gam_data(load=False)
    else:
        data = []
        pred_data = []
        for sample_size in [ 10,20, 30, 50, 100, 200, 500, 1000]:
            df = load_dfs(sample_size=sample_size, simulate=True)
            for feature_name in df["feature_name"].unique():

                for dataset in df["Dataset"].unique():
                    for shift in df["Shift"].unique():
                        for ks in [False, True]:

                            train = df[(df["feature_name"]==feature_name) & (df["Dataset"]==dataset) & (df["Shift"]==shift) & (df["KS"]==ks)]
                            test = df[(df["feature_name"]==feature_name) & (df["Dataset"]==dataset) & (df["KS"]==ks)]
                            X_train = train['feature']  # Ensure this is adjusted to your dataset
                            y_train = train['loss']
                            X_test = test['feature']
                            y_test = test['loss']#-train["loss"].mean()
                            # Fit the GAM
                            # combined_X = np.concatenate((X_train, X_test))
                            gam_regular = pygam.LinearGAM(fit_intercept=False)
                            gam_regular.fit(X_train, y_train)
                            print("Feature:", feature_name, "Shift:", shift, "Dataset:", dataset, "KS:", ks)
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



                            XX = gam_regular.generate_X_grid(term=0)
                            grid_preds = gam_regular.predict(XX)
                            grid_preds_conf = gam_regular.prediction_intervals(XX, width=0.95)

                            monotonic_grid_preds = gam_monotonic.predict(XX)
                            monotonic_grid_preds_conf = gam_monotonic.prediction_intervals(XX, width=0.95)

                            for i, (x, y, y_c, ym, ym_c) in enumerate(zip(XX, grid_preds, grid_preds_conf, monotonic_grid_preds, monotonic_grid_preds_conf)):
                                pred_data.append({"feature":x[0], "pred_loss":y,
                                                  "pred_loss_lower":y_c[0], "pred_loss_upper":y_c[1],
                                                    "monotonic_pred_loss":ym, "monotonic_pred_loss_lower":ym_c[0], "monotonic_pred_loss_upper":ym_c[1],

                                                  "Dataset":dataset, "train_shift":shift, "feature_name":feature_name, "KS":ks, "sample_size":sample_size})


                            if not ks:
                                plt.plot(XX[:, 0], gam_monotonic.predict(X=XX))
                                plt.scatter(X_train, y_train, c='b', alpha=0.1)
                                plt.scatter(X_test, y_test, c='r', alpha=0.1)
                                plt.show()
                            for test_shift in test["Shift"].unique():
                                test_data = test[test["Shift"] == test_shift]
                                for sev in test_data["Shift Severity"].unique():
                                    test_data_fold = test_data[test_data["Shift Severity"] == sev]
                                    preds_regular = gam_regular.predict(test_data_fold["feature"])
                                    mape_regular = np.mean(np.abs(preds_regular - test_data_fold["loss"]) / np.abs(test_data_fold["loss"]))
                                    mae_regular = np.mean(np.abs(preds_regular - test_data_fold["loss"]))

                                    preds_monotonic = gam_monotonic.predict(test_data_fold["feature"])
                                    mape_monotonic = np.mean(np.abs(preds_monotonic - test_data_fold["loss"]) / np.abs(
                                        test_data_fold["loss"]))
                                    mae_monotonic = np.mean(np.abs(preds_monotonic - test_data_fold["loss"]))

                                    data.append({"Dataset":dataset, "feature_name":feature_name, "train_shift":shift, "test_shift":test_shift, "Shift Severity":sev, "sample_size":sample_size, "KS":ks, "regular mae":mae_regular, "regular mape": mape_regular,"monotonic mape": mape_monotonic, "monotonic mae": mae_monotonic})

                            # preds = gam.predict(X_test)
                            # smape = np.mean(np.abs(preds - y_test) / (np.abs(preds) + np.abs(y_test)))
                            # print(f"Dataset: {dataset} Feature: {feature_name} Shift: {shift}: {smape*100}")
        df = pd.DataFrame(data)
        pred_df = pd.DataFrame(pred_data)
        pred_df.to_csv("gam_preds.csv")
        df.to_csv(f"gam_results_ks.csv")
        return df

def chatterjee_ks(df, cols=("feature_name","KS")):
    def chatterjee_test(x, y):
        df = pd.DataFrame(zip(x, y), columns=["x", "y"])
        df["yranks"] = df["y"].rank()
        df.head()
        df = df.sort_values("x")  # Sort by the rank of X
        df.head()
        rank_series = df["yranks"].reset_index(drop=True)
        diff = []
        for i in range(len(rank_series) - 1):
            diff.append(abs(rank_series[i + 1] - rank_series[i]))

        xi = 1 - 3 * (sum(diff) / (len(y) ** 2 - 1))
        return xi

    return df.groupby(cols).apply(lambda x: chatterjee_test(x["feature"], x["loss"]))



def simulate_sampling(df, samples, sample_size):
    def sample_loss_feature(group, n_samples, n_size):
        samples = []
        for i in range(n_samples):
            sample = group.sample(n=n_size, replace=True)  # Sampling with replacement
            mean_loss = sample['loss'].mean()
            mean_feature = sample['feature'].mean()
            samples.append({'loss': mean_loss, 'feature': mean_feature, "KS":False})
            # samples.append({'loss': mean_loss, 'feature': ks_2samp(df[df["fold"]=="train"]["feature"], sample['feature'])[0], "KS":True})
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
def regplots(sample_size, simulate):
    def bin_Y(group, bins):
        group['feature_bin'] = pd.qcut(group['feature'], bins, labels=False, duplicates='drop')
        return group

    df = load_dfs(sample_size=sample_size, simulate=simulate)
    df["ind"]=df["fold"]=="ind"
    df.replace({"normal":"Organic Shift"}, inplace=True)
    df.loc[df["fold"]=="ind", "Shift"] = "ind"
    ind = df[df["fold"]=="ind"]
    hues = df["Shift"].unique()
    df = df[df["KS"]==False]

    g = sns.FacetGrid(df, row="Dataset", col="feature_name", margin_titles=True, sharex=False, sharey=True)
    g.map_dataframe(sns.scatterplot, x="feature", y="loss", hue="Shift", hue_order=hues,  alpha=0.5)
    g.add_legend()
    plt.show()


def show_thresholding_problems():
    df = load_dfs(100, simulate=True)
    df = df[df["KS"] == False]
    df.loc[df["fold"] == "ind", "Shift"] = "ind"
    df = df[df["fold"].isin(["ind", "ood", "noise_0.25", "dropout_0.25"])]
    df["ind"] = df["fold"] == "ind"

    # Explicit hue order
    # hue_order = ['ind', 'noise', 'normal', 'dropout']
    # hue_order=["ind", ""]
    g = sns.FacetGrid(df, col="feature_name", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.kdeplot, x="feature", common_norm=False)

    # Manually add legend if automatic is not working
    plt.legend(title="Shift Category")

    plt.show()


def classification_metrics(sample_size=100, simulate=True):
    df = load_dfs(sample_size, simulate=simulate)
    # auroc(df)
    variance(df)

def correleations(sample_size=100, simulate=False):
    df = load_dfs(sample_size, simulate=simulate)
    sprm = spearman(df, cols=["Dataset", "feature_name", "KS"])
    kndl = kendall(df, cols=["Dataset", "feature_name", "KS"])
    prs = pearson(df, cols=["Dataset", "feature_name", "KS"])
    chatterjee = chatterjee_ks(df, cols=["Dataset", "feature_name","KS"])
    merged = pd.concat([sprm, kndl, prs, chatterjee], axis=1)
    merged.rename(columns={0:"Spearman R", 1:"Kendall Tau", 2:"Pearson R", 3:"Chatterjee CC"}, inplace=True)
    return merged.reset_index()

def quantize_values_and_plot_kdes():
    df = load_dfs(100)
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    df = df[df["loss"]!=0]
    for dataset in df["Dataset"].unique():
        for fname in df["feature_name"].unique():
            subdf = df[(df["Dataset"]==dataset) &(df["feature_name"]==fname)]
            subdf["bin"] = pd.cut(subdf["feature"], bins=5)
            g = sns.FacetGrid(data=subdf, row="bin", aspect=15, height=1, palette=pal)
            g.map_dataframe(sns.kdeplot, x="loss", bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
            plt.show()


def regplot_by_shift(sample_size, simulate=False):

    df = load_dfs(sample_size=sample_size, simulate=simulate)
    df["ind"]=df["fold"]=="ind"
    df["Shift Severity"]=df["fold"].apply(lambda x: round(float(x.split("_")[1]),2) if "_" in x else x)
    df.rename(columns={"feature_name":"Feature"}, inplace=True)
    df.replace({"typicality":"Typicality", "cross_entropy":"Cross Entropy", "knn":"KNN", "odin":"ODIN", "grad_magnitude":"GradNorm", "energy":"Energy", "softmax":"Softmax"}, inplace=True)
    hues = df["Shift Severity"].unique()
    if simulate:
        df = df[df["KS"]==False]
    df.replace({"normal":"Organic Shift"}, inplace=True)
    g = sns.FacetGrid(df, row="Shift", col="Feature", margin_titles=True, sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x="feature", y="loss", hue="Shift Severity", hue_order=hues)
    g.add_legend()
    plt.show()


def sanity_check():
    df = load_dfs(100, simulate=False)
    g = sns.FacetGrid(df, row="Shift", margin_titles=True)
    g.map_dataframe(sns.kdeplot, x="feature", hue="feature_name", common_norm=False)
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


def plot_sample_size_effect(load=True):
    if load:
        corr_df = pd.read_csv("sample_size_effect.csv")
    else:
        corr_dfs_by_sample_size = [correleations(sample_size=i, simulate=True) for i in [10,30,50, 100, 200, 500, 1000]]

        for i, df in enumerate(corr_dfs_by_sample_size):
            df["Sample Size"] = [5,10,30,50,75, 100, 200, 500, 1000][i]


        corr_df = pd.concat(corr_dfs_by_sample_size)
    corr_df["Spearman R"] = corr_df["Spearman R"].abs()
    corr_df.to_csv("sample_size_effect.csv")
    g = sns.FacetGrid(corr_df, row="Dataset", col="feature_name", margin_titles=True, sharex=True, sharey=True)
    g.map_dataframe(sns.scatterplot, x="Sample Size", y="Spearman R", hue="KS")
    # g.map_dataframe(sns.lineplot, x="Sample Size", y="Spearman R", hue="KS")
    g.add_legend()
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

def gam_fits(metric="monotonic mae", KS=True):
    df = load_dfs(100, simulate=True)
    gam_results = get_gam_data()
    gam_results = gam_results[gam_results["KS"]==KS]
    gam_results = gam_results[gam_results["sample_size"]==100]
    gam_results_grouped = gam_results.groupby(["Dataset", "feature_name", "train_shift"])[["monotonic mae", "monotonic mape", "regular mae", "regular mape"]].mean().reset_index()
    idx = gam_results_grouped.groupby(["Dataset", "feature_name"])[[metric]].apply(lambda x: x.idxmin())
    min_mae_df = gam_results_grouped.iloc[idx[metric].values].reset_index(drop=True)
    print(min_mae_df)
    shift_colors = dict(zip(df["Shift"].unique(), sns.color_palette("pastel", len(df["Shift"].unique()))))
    print(shift_colors)
    min_mape_df = gam_results_grouped.iloc[idx[metric].values].reset_index(drop=True)
    df = df[df["KS"]==KS]
    df_preds = pd.read_csv("gam_preds.csv")
    df_preds = df_preds[df_preds["sample_size"]==100]
    df_preds = df_preds[df_preds["KS"]==KS]
    fig, ax = plt.subplots(len(df["Dataset"].unique()),len(df["feature_name"].unique()), figsize=(20, 10))
    for i, dataset in enumerate(df["Dataset"].unique()):
        for j, feature_name in enumerate(df["feature_name"].unique()):
            train_shift = min_mae_df[(min_mae_df["Dataset"]==dataset) & (min_mae_df["feature_name"]==feature_name)]["train_shift"].values[0]

            subdf = df[(df["Dataset"]==dataset) & (df["feature_name"]==feature_name) & (df["Shift"]!=train_shift)]
            subdf_train = df[(df["Dataset"]==dataset) & (df["feature_name"]==feature_name) & (df["Shift"]==train_shift)]

            subdf_preds = df_preds[(df_preds["Dataset"]==dataset) & (df_preds["feature_name"]==feature_name) & (df_preds["train_shift"]==train_shift)]
            # print(subdf.columns)
            for shift in subdf["Shift"].unique():
                subdf_shift = subdf[subdf["Shift"]==shift]
                ax[i,j].scatter(subdf_shift["feature"], subdf_shift["loss"], alpha=0.5, color=shift_colors[shift])
            # ax[i,j].scatter(subdf["feature"], subdf["loss"], alpha=0.5)
            ax[i,j].plot(subdf_preds["feature"], subdf_preds["monotonic_pred_loss"], color="red")
            ax[i,j].fill_between(subdf_preds["feature"], subdf_preds["monotonic_pred_loss_lower"], subdf_preds["monotonic_pred_loss_upper"], color="red", alpha=0.3)
            ax[i,j].set_title(f"{dataset}|{feature_name}: MAPE={min_mape_df[(min_mape_df['Dataset']==dataset) & (min_mape_df['feature_name']==feature_name)]['monotonic mape'].values[0]:.2f}")
            ax[i,j].scatter(subdf_train["feature"], subdf_train["loss"], alpha=0.5, label="train")
    plt.tight_layout()
    plt.show()


def find_best_train_shift():
    df = get_gam_data()
    print(df.groupby(["train_shift"])[["monotonic mape"]].mean())

if __name__ == '__main__':
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
    regplots(100, simulate=True)
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
    # load_old_dfs()
    # quantize_values_and_plot_kdes()