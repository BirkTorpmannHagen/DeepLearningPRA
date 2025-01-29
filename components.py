import numpy as np
from matplotlib import pyplot as plt
from pygam import LinearGAM

from utils import *

class OODDetector:
    def __init__(self, df):
        self.df = df
        threshold_df = df[(df["shift"] == "ind_val")]

        self.higher_is_ood = df[df["shift"]=="EtisLaribDB"]["feature"].mean() > df[df["shift"]=="ind_val"]["feature"].mean()
        if self.higher_is_ood:
            self.threshold = threshold_df["feature"].sample(len(threshold_df)//2).max() # sample half of the data to serve as thresholding
        else:
            self.threshold = threshold_df["feature"].sample(len(threshold_df)//2).min()

    def predict(self, batch):
        if self.higher_is_ood:
            return batch["feature"].mean() > self.threshold
        else:
            return batch["feature"].mean() < self.threshold

    def get_likelihood(self):
        self.ind = self.df[self.df["shift"] == "ind_val"]
        self.ood = self.df[self.df["shift"] == "EtisLaribDB"]
        tpr = (self.ood["feature"] > self.threshold).mean()
        tnr = (self.ind["feature"] < self.threshold).mean()
        return tpr, tnr


class SyntheticOODDetector:
    def __init__(self, tpr, tnr):
        self.tpr = tpr
        self.tnr = tnr


    def predict(self, batch):
        # print(batch.columns)
        # input()
        assert batch["ood"].nunique()==1
        if batch["ood"].all(): #if the sample is ood
            #return true with likelihood = tpr, else false (1-tpr)
            return 1 if np.random.rand() < self.tpr else 0
        else: #if the sample is ind
            #return true with likelihood = tnr, else false (1-tnr)
            return 0 if np.random.rand() < self.tnr else 1

    def get_likelihood(self):
        return self.tpr, self.tnr




class LossEstimator:
    def __init__(self, df):
        self.df = df
        self.gam = LinearGAM(constraints="monotonic_inc")
        self.train_set = df[(df["shift"]=="ind_val")|(df["shift"]==ETISLARIB) | (df["shift"]=="noise") |(df["shift"]=="adv")]
        self.train(self.train_set["feature"], self.train_set["loss"])
        # self.plot_gam()

    def plot_gam(self):
        plt.figure()
        plt.scatter(self.train_set["feature"], self.train_set["loss"], label="train")
        x = np.linspace(self.train_set["feature"].min(), self.train_set["feature"].max(), 100)
        plt.plot(x, self.gam.predict(x), label="gam")
        plt.legend()
        plt.savefig("gam.png")
        plt.title("Loss Estimator")
        plt.show()

    def train(self, X, Y):
        self.gam.fit(X, Y)
        return np.sqrt(((self.gam.predict(X)-Y)**2).mean()) #MSE

    def evaluate(self, batch):
        return np.sqrt(((self.gam.predict(batch["feature"])-batch["loss"])**2).mean())

    def predict(self, batch):
        return self.gam.predict(batch["feature"])


class SplitLossEstimator:
    def __init__(self, df):
        self.df = df
        self.gam = LinearGAM(constraints="monotonic_inc")
        self.ind_loss = df[df["shift"]=="ind_test"]["loss"].mean()
        self.train_set = df[(df["shift"]==ETISLARIB) | (df["shift"]=="noise") |(df["shift"]=="adv")]
        self.train(self.train_set["feature"], self.train_set["loss"])
        self.ood_detector = OODDetector(df)
        self.plot_gam()
    #
    def plot_gam(self):
        plt.figure()
        plt.scatter(self.train_set["feature"], self.train_set["loss"], label="train")
        x = np.linspace(self.train_set["feature"].min(), self.train_set["feature"].max(), 100)
        plt.plot(x, self.gam.predict(x), label="gam")
        plt.legend()
        plt.title("Split Loss Estimator")
        plt.show()

    def train(self, X, Y):
        self.gam.fit(X, Y)
        return np.sqrt(((self.gam.predict(X)-Y)**2).mean()) #MSE

    def evaluate(self, batch):
        if self.ood_detector.predict(batch):
            return batch["loss"].mean()-self.ind_loss
        return np.sqrt(((self.gam.predict(batch["feature"])-batch["loss"])**2).mean())

    def predict(self, batch):
        assert batch["ood"].nunique()==1
        if batch["ood"].all():
            return self.gam.predict(batch["feature"])
        else:
            return self.ind_loss



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

if __name__ == '__main__':
    data = load_pra_df("knn", batch_size=10, samples=1000)
    le_s = SplitLossEstimator(data)
    le = LossEstimator(data)
