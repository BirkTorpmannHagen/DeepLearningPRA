from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import roc_auc_score


def ba(df, cols=("feature_name", "KS")):
    for dataset in df["Dataset"].unique():
        for shift in df["Shift"].unique():
            for feature_name in df["feature_name"].unique():
                ind = df[(df["Dataset"] == dataset) & (df["Shift"] == shift) & (df["feature_name"] == feature_name) & (df["fold"] == "ind")].copy()
                ood = df[(df["Dataset"] == dataset) & (df["Shift"] == shift) & (df["feature_name"] == feature_name) & (df["fold"] != "ind")].copy()
                if ind["feature"].mean() < ood["feature"].mean():
                    threshold = ind["feature"].max()
                    ind.loc[:, "pred"] = ind["feature"] < threshold  # Modify the entire 'pred' column
                    ood.loc[:, "pred"] = ood["feature"] > threshold
                else:
                    threshold = ind["feature"].min()
                    ind.loc[:, "pred"] = ind["feature"] > threshold
                    ood.loc[:, "pred"] = ood["feature"] < threshold

                balanced_accuracy = (ind["pred"].mean()+ood["pred"].mean())/2
                print(f"{dataset} {shift} {feature_name} Balanced Accuracy: {balanced_accuracy}")

def variance(df):
    for feature_name in df["feature_name"].unique():
        print(df[df["feature_name"]==feature_name].groupby(["Dataset", "Shift", "fold", "KS"])["feature"].apply(lambda x: x.std()))
def auroc(df, cols=("feature_name", "KS")):
    for dataset in df["Dataset"].unique():
        for shift in df["Shift"].unique():
            for feature_name in df["feature_name"].unique():
                for ks in df["KS"].unique():
                    ind = df[(df["Dataset"] == dataset) & (df["Shift"] == shift) & (df["feature_name"] == feature_name) & (df["fold"] == "ind") & (df["KS"]==ks)].copy()
                    ood = df[(df["Dataset"] == dataset) & (df["Shift"] == shift) & (df["feature_name"] == feature_name) & (df["fold"] != "ind")& (df["KS"]==ks)].copy()

                    aur = roc_auc_score([0]*len(ind)+[1]*len(ood), list(ind["feature"])+list(ood["feature"]))

                    print(f"{dataset} {shift} {feature_name} KS={ks} AUROC: {aur}")



def spearman(df, cols=("feature_name","KS")):
    return df[df["fold"]!="train"].groupby(list(cols)).apply(lambda x: spearmanr(x["feature"], x["loss"])[0])#.reset_index(name="Spearman R")


def pearson(df, cols=("feature_name","KS")):
    return df[df["fold"]!="train"].groupby(list(cols)).apply(lambda x: pearsonr(x["feature"], x["loss"])[0])#.reset_index(name="Pearson R")


def kendall(df, cols=("feature_name","KS")):
    return df[df["fold"]!="train"].groupby(list(cols)).apply(lambda x: kendalltau(x["feature"], x["loss"])[0])#.reset_index(name="Kendall Tau")
