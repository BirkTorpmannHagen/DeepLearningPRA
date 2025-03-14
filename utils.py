import os
from os.path import join

import numpy as np
import pandas as pd



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

def load_all(batch_size=30, samples=1000, feature="all"):
    dfs = []
    for dataset in DATASETS:
        if feature!="all":
            dfs.append(load_pra_df(dataset, feature, batch_size, samples))
        else:
            for dsd in DSDS:
               dfs.append(load_pra_df(dataset, dsd, batch_size, samples))
    return pd.concat(dfs)


def load_pra_df(dataset_name, feature_name, batch_size=30, samples=1000):

    try:
        df = pd.concat(
        [pd.read_csv(join("single_data", fname)) for fname in os.listdir("single_data") if dataset_name in fname and feature_name in fname])
    except:
        print("no data found for ", dataset_name, feature_name)
        return pd.DataFrame()

    df["Dataset"]=dataset_name
    df["batch_size"]=batch_size
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
    return df


DSD_PRINT_LUT = {"grad_magnitude": "GradNorm", "cross_entropy" : "Entropy", "energy":"Energy", "knn":"kNN"}
DATASETS = ["CCT", "OfficeHome", "Office31", "NICO", "Polyp"]
DSDS = ["knn", "grad_magnitude", "cross_entropy", "energy"]
# BATCH_SIZES = [32]
BATCH_SIZES = [1, 8, 16, 32, 64]
# BATCH_SIZES = np.arange(1, 64)