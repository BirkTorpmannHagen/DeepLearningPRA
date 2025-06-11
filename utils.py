import os
from os.path import join

import numpy as np
import pandas as pd

from components import OODDetector





#write a method that takes an object as argument and returns a class that extends that object


class ArgumentIterator:
    #add arguments to an iterator for use in parallell processing
    def __init__(self, iterable, variables):
        self.iterable = iterable
        self.index = 0
        self.variables = variables

    def __next__(self):
        if self.index >= len(self.iterable):
            # print("stopping")
            raise StopIteration
        else:
            self.index += 1
            return self.iterable[self.index-1], *self.variables

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

def load_all(batch_size=30, samples=1000, feature="all", compute_ood=False, prefix="single_data"):
    dfs = []
    for dataset in DATASETS:
        if feature!="all":
            dfs.append(load_pra_df(dataset, feature, model="", batch_size=batch_size, samples=samples, compute_ood=compute_ood, prefix=prefix))
        else:
            for dsd in DSDS:
               dfs.append(load_pra_df(dataset, dsd, model="", batch_size=batch_size, samples=samples, compute_ood=compute_ood, prefix=prefix))
    return pd.concat(dfs)


def load_pra_df(dataset_name, feature_name, model, batch_size=30, samples=1000, prefix="single_data", compute_ood=True):
    try:
        df = pd.concat(
        [pd.read_csv(join("single_data", fname)) for fname in os.listdir("single_data") if dataset_name in fname and feature_name in fname and model in fname])
    except:
        print("no data found for ", dataset_name, feature_name)
        return pd.DataFrame()

    df["Dataset"]=dataset_name
    df["batch_size"]=batch_size
    df["Model"]=model
    df.drop(columns=["Unnamed: 0"], inplace=True)

    if batch_size!=1:
        def sample_loss_feature(group, n_samples, n_size):
            samples = []
            for i in range(n_samples):
                sample = group.sample(n=n_size, replace=True)  # Sampling with replacement
                # input()
                if sample["Dataset"].all()=="Polyp":
                    mean_loss = sample['loss'].median()
                else:
                    mean_loss = sample['loss'].mean()

                mean_feature = sample['feature'].mean()
                samples.append({'loss': mean_loss, 'feature': mean_feature})
                # samples.append({'loss': mean_loss, 'feature': brunnermunzel(df[df["fold"]=="train"]["feature"], sample['feature'])[0], "reduction":"bm"})
            return pd.DataFrame(samples)
            # Return a DataFrame of means with the original group keys

        df = df.groupby(["fold", "feature_name", "Dataset"]).apply(sample_loss_feature, samples, batch_size).reset_index()


    if dataset_name=="Polyp":
        if batch_size==1:
            df["correct_prediction"] = df["loss"] < 0.5  # arbitrary threshold
        else:
            df["correct_prediction"] = df["loss"] < df[df["fold"]=="ind_val"]["loss"].max()  #maximum observed val mean jaccard
    else:
        # print(df[df["fold"]=="ind_val"]["loss"].quantile(0.05))
        if batch_size==1:
            df["correct_prediction"] = df["loss"]>0.5 #arbitrary threshold;
        else:
            df["correct_prediction"] = df["loss"] > df[df["fold"] == "ind_val"]["loss"].min()

    df["shift"] = df["fold"].apply(lambda x: x.split("_")[0] if "_0." in x else x)            #what kind of shift has occured?
    df["shift_intensity"] = df["fold"].apply(lambda x: x.split("_")[1] if "_" in x else x)  #what intensity?
    df["ood"] = ~df["fold"].isin(["train", "ind_val", "ind_test"])#&~df["correct_prediction"] #is the prediction correct?
    if compute_ood:
        dfs = []
        for ood_val in df[df["ood"]]["fold"].unique():
            if "noise" in ood_val:
                continue
            else:
                df_copy = df.copy()
                detector = OODDetector(df_copy, ood_val)

                df_copy["verdict"] = df_copy.apply(detector.predict, axis=1)
                df_copy["ood_val_shift"] = ood_val
                dfs.append(df_copy)
        df = pd.concat(dfs)

    return df


DSD_PRINT_LUT = {"grad_magnitude": "GradNorm", "cross_entropy" : "Entropy", "energy":"Energy", "knn":"kNN", "mahalanobis":"Mahalanobis", "softmax":"Softmax", "typicality":"Typicality"}
DATASETS = ["CCT", "OfficeHome", "Office31", "NICO", "Polyp"]
DSDS = ["knn", "grad_magnitude", "cross_entropy", "energy", "mahalanobis"]
# BATCH_SIZES = [32]
BATCH_SIZES = [1, 8, 16, 32, 64]
THRESHOLD_METHODS = [ "val_optimal", "ind_span", "density"]
# BATCH_SIZES = np.arange(1, 64)