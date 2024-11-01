from scipy.stats import gaussian_kde
from ooddetectors import RabanserSD, convert_to_pandas_df, process_dataframe, open_and_process
import seaborn as sns
import numpy as np
import pandas as pd
class LossPredictor:

    def __init__(self, num_samples, testbed, select_samples=True, num_bins=20):
        self.testbed = testbed
        self.sample_size = testbed.sample_size
        self.ood_detector = RabanserSD(testbed.rep_model, select_samples=select_samples)
        self.ood_detector.register_testbed(testbed)
        self.fname = f"{testbed.name}_ks_{num_samples}_noise.csv"
        self.num_bins = num_bins
        self.kdes = self.bootstrap_kdes()


    def bootstrap_kdes(self):
        try:
            data = open_and_process(self.fname)
        except FileNotFoundError:
            #TODO replace this with a method that computes w/ noise datasets.
            self.data = process_dataframe(convert_to_pandas_df(
                self.ood_detector.compute_pvals_and_loss(self.testbed.sample_size, test="ks")))

            self.data.to_csv(self.fname)

        # convert to log scale
        data["pvalue"] = data["pvalue"].apply(lambda x: np.log10(x))
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(axis=1, how='any', inplace=True)

        # group pvalues into bins

        labels = range(1, self.num_bins + 1)  # The bin labels; ensures unique labels for each bin
        data["bin"], bin_edges = pd.qcut(data["pvalue"], q=self.num_bins, retbins=True, labels=labels)
        data["bin"] = data["bin"].astype(int)
        data["bin"] = data["bin"].apply(lambda x: round(bin_edges[x - 1]))

        # compute KDEs for each bin
        kdes = {} # list of KDE objects for each bin
        for bin in pd.unique(data["bin"]):
            kdes[bin]=gaussian_kde(data[data["bin"] == bin]["loss"])
        return kdes



    def predict(self, loader):
        self.ood_detector.compute_features_and_loss_for_loaders(dataloaders=loader, sample_size=self.sample_size, test="ks")
        #todo need to do some refactoring here
