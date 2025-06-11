import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pygam import LinearGAM
import seaborn as sns
from sklearn.neighbors import KernelDensity
from vae.utils.general import check_python


def get_optimal_threshold(ind, ood):
    merged = np.concatenate([ind, ood])
    max_acc = 0
    threshold = 0
    if ind.mean()<ood.mean():
        higher_is_ood = True
    else:
        higher_is_ood = False

    #if linearly seperable, set the threshold to the middle
    if ind.max()<ood.min() and higher_is_ood:
        return (ind.max()+ood.min())/2
    if ood.max()<ind.min() and not higher_is_ood:
        return (ind.max() + ood.min()) / 2

    for t in np.linspace(merged.min(), merged.max(), 100):
        if higher_is_ood:
            ind_acc = (ind<t).mean()
            ood_acc = (ood>t).mean()
        else:
            ind_acc = (ind>t).mean()
            ood_acc = (ood<t).mean()
        bal_acc = 0.5*(ind_acc+ood_acc)
        if not higher_is_ood:
            if bal_acc>=max_acc: #ensures that the threshold is near ind data for highly seperable datasets
                max_acc = bal_acc
                threshold = t
        else:
            if bal_acc>max_acc:
                max_acc = bal_acc
                threshold = t


    return threshold


class OODDetector:
    def __init__(self, df, ood_val_shift, threshold_method="val_optimal"):
        assert df["feature_name"].nunique() == 1
        assert df["Dataset"].nunique() == 1
        self.df = df
        self.threshold_method = threshold_method
        # self.ind_val = df[(df["shift"] == "ind_val")&(~df["ood"])]
        # self.ood_val = df[(df["shift"] == ood_val_shift)&(df["ood"])]
        self.ind_val = df[~df["ood"]]
        self.ood_val = df[df["ood"]]
        self.higher_is_ood = self.ood_val["feature"].mean() >self.ind_val["feature"].mean()
        if threshold_method == "val_optimal":
            self.threshold = get_optimal_threshold(self.ind_val["feature"], self.ood_val["feature"])
        if threshold_method == "ind_span":
            lower = self.ind_val["feature"].quantile(0.01)
            upper = self.ind_val["feature"].quantile(0.99)
            self.threshold = [lower,upper]
        if threshold_method == "density":
            self.density_model = KernelDensity(kernel='gaussian', bandwidth=0.1)
            self.density_model.fit(self.ind_val["feature"].values.reshape(-1, 1))

            self.threshold = 0





    def predict(self, batch):
        # if isinstance(batch["feature"], pd.Series): #if it is a batch
        #     feature = batch["feature"].mean()
        # else:
        #     feature = batch["feature"]
        feature = batch["feature"]
        if self.threshold_method == "ind_span":
            return not (self.threshold[0] <= feature <= self.threshold[1])
        elif self.threshold_method=="val_optimal":
            if self.higher_is_ood:
                return  feature > self.threshold or feature<self.ind_val["feature"].min()
            else:
                return feature < self.threshold or feature>self.ind_val["feature"].max()
        elif self.threshold_method == "density":
            prob = np.exp(self.density_model.score(np.array(feature).reshape(1, -1)))
            return prob<0.01



    def get_tpr(self, data):
        eval_copy = data.copy()
        eval_copy["pred_ood"] = eval_copy.apply(lambda row: self.predict(row))
        return eval_copy[eval_copy["ood"]]["pred_ood"].mean()

    def get_tnr(self, data):
        eval_copy = data.copy()
        eval_copy["pred_ood"] = eval_copy.apply(lambda row: self.predict(row), axis=1)
        return 1-eval_copy[eval_copy["ind"]]["pred_ood"].mean()

    def get_accuracy(self, data):
        return 0.5*(self.get_tpr(data)+self.get_tnr(data))

    def get_metrics(self, data):
        return self.get_tpr(data), self.get_tnr(data), self.get_accuracy(data)


    def plot_hist(self):
        sns.histplot(self.df, x="feature", hue="ood", alpha=0.5)

        if self.threshold_method == "ind_span":
            plt.axvline(self.threshold[0], color="red", linestyle="--")
            plt.axvline(self.threshold[1], color="red", linestyle="--")
        elif self.threshold_method == "val_optimal":
            plt.axvline(self.threshold, color="red", linestyle="--")

        # plt.title(self.get_accuracy(self.df))
        plt.show()

    def get_likelihood(self):
        return self.get_tpr(self.df), self.get_tpr(self.df)



class SyntheticOODDetector:
    def __init__(self, tpr, tnr):
        self.tpr = tpr
        self.tnr = tnr


    def predict(self, batch):

        # print(batch.columns)
        # input()
        if type(batch["ood"])==bool:
            if batch["ood"]:
                return 1 if np.random.rand() < self.tpr else 0
            else:
                return 0 if np.random.rand() < self.tnr else 1

        assert batch["ood"].nunique()==1
        if batch["ood"].all(): #if the sample is ood
            #return true with likelihood = tpr, else false (1-tpr)
            return 1 if np.random.rand() < self.tpr else 0
        else: #if the sample is ind
            #return true with likelihood = tnr, else false (1-tnr)
            return 0 if np.random.rand() < self.tnr else 1

    def get_likelihood(self):
        return self.tpr, self.tnr





class Trace:
    """
    Container for dsd verdict traces; used to estimate lambda
    """
    def __init__(self, trace_length=100):
        self.trace_length = trace_length
        self.trace = []

    def update(self, item):
        if len(self.trace) == self.trace_length:
            self.trace.pop(0)
        self.trace.append(item)
        return self.trace

