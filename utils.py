import os
from os.path import join

import numpy as np
import pandas as pd
from tqdm import tqdm

from components import OODDetector


def sample_loss_feature(group, n_samples, n_size, reduce=True):
    samples = []
    for i in range(n_samples):
        sample = group.sample(n=n_size, replace=True)  # Sampling with replacement
        sample["index"] = i
        if reduce:
            if sample["Dataset"].all() == "Polyp":
                mean_loss = sample['loss'].median()
            else:
                mean_loss = sample['loss'].mean()

            mean_feature = sample['feature'].mean()
            samples.append({'loss': mean_loss, 'feature': mean_feature, "acc": sample["acc"].mean(), "index": i})
        else:
            for _, row in sample.iterrows():
                samples.append({
                    'loss': row['loss'],
                    'feature': row['feature'],
                    'acc': row['acc'],
                    'index': i
                })
    return pd.DataFrame(samples)

def sample_biased_loss_feature(group, n_samples, n_size, bias="None", reduce=True):
    samples = []
    print("Sampling group with bias:", bias)
    for i in range(n_samples):
        if bias == "Unbiased":
            sample = group.sample(n=n_size, replace=True)  # Sampling with replacement
        elif bias == "Class":
            sample_class = np.random.choice(group["class"])
            sample = group[group["class"] == sample_class].sample(n=n_size, replace=True)
        elif bias == "Synthetic":
            group["feature_bins"] = pd.cut(group["feature"], bins=len(group) // n_size)
            random_bin = np.random.choice(group["feature_bins"].unique())
            sample = group[group["feature_bins"] == random_bin].sample(n=n_size, replace=True)
            group.drop(columns=["feature_bins"], inplace=True)  # Clean up after sampling
        elif bias == "Temporal":
            rand_indx = np.random.choice(group["idx"])
            group_weights = group["idx"].apply(lambda x: np.abs(x-rand_indx))
            group_weights = group_weights / group_weights.sum()
            sample = group.sample(n=n_size, replace=True, weights=group_weights)  # Sampling with replacement
        else:
            raise ValueError(f"Unknown bias type: {bias}")
        sample["index"] = i
        if reduce:
            if sample["Dataset"].all() == "Polyp":
                mean_loss = sample['loss'].median()
            else:
                mean_loss = sample['loss'].mean()

            mean_feature = sample['feature'].mean()
            samples.append({'loss': mean_loss, 'feature': mean_feature, "acc": sample["acc"].mean(), "index": i})
        else:
            for _, row in sample.iterrows():
                samples.append({
                    'loss': row['loss'],
                    'feature': row['feature'],
                    'acc': row['acc'],
                    'index': i
                })
    return pd.DataFrame(samples)

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

def load_random_rabanser(batch_size, prefix="debiased_data"):
    dfs = []
    for dataset in DATASETS:
        try:
            fname = f"{dataset}_normal_RandomSampler_{batch_size}_k=0_rabanser.csv"
            df = pd.read_csv(
                join(prefix, f"{dataset}_normal_RandomSampler_{batch_size}_k=0_rabanser.csv"))
        except FileNotFoundError:
            print(f"No data found for {prefix}/{dataset}_normal_RandomSampler_{batch_size}_k=0_rabanser.csv")
            continue
        df["feature_name"] = "rabanser"
        df["Dataset"] = dataset
        df["batch_size"] = batch_size
        if dataset == "Polyp":
            df["correct_prediction"] = df["loss"] < df[df["fold"] == "ind_val"][
                "loss"].max()  # maximum observed val mean jaccard
        else:
            df["correct_prediction"] = df["loss"] < df[df["fold"] == "ind_val"]["loss"].quantile(
                0.95)  # losswise definition
            # df["correct_prediction"] = df["acc"]>=ind_val_acc   #accuracywise definition
        df["shift"] = df["fold"].apply(
            lambda x: x.split("_")[0] if "_0." in x else x)  # what kind of shift has occured?
        df["shift_intensity"] = df["fold"].apply(
            lambda x: x.split("_")[1] if "_" in x else x)  # what intensity?
        df["ood"] = ~df["fold"].isin(["train", "ind_val", "ind_test"])
        dfs.append(df)
    try:
        return pd.concat(dfs)
    except ValueError:
        print(f"No data found for and {batch_size}.csv")
        return []


def load_all_biased(prefix="debiased_data", filter_batch=False):
    dfs = []
    for dataset in DATASETS:
        for sampler in ["RandomSampler", "SequentialSampler", "ClassOrderSampler", "ClusterSampler"]:
            for batch_size in BATCH_SIZES[1:]:
                if filter_batch:
                    if batch_size!=filter_batch:
                        continue
                for k in [-1, 0, 1, 5, 10]:
                    for feature in DSDS:
                        if feature=="softmax" and dataset=="Polyp":
                            continue
                        try:
                            df = pd.read_csv(join(prefix, f"{dataset}_normal_{sampler}_{batch_size}_k={k}_{feature}.csv"))
                        except FileNotFoundError:
                            # print(f"No data found for {prefix}/{dataset}_normal_{sampler}_{batch_size}_k={k}_{feature}.csv")
                            continue
                        try:

                            df["bias"] = sampler
                            df["feature_name"]=feature
                            df["k"]=k
                            df["Dataset"] = dataset
                            df["batch_size"] = batch_size
                            if dataset == "Polyp":
                               df["correct_prediction"] = df["loss"] < df[df["fold"] == "ind_val"][
                                "loss"].max()  # maximum observed val mean jaccard
                            else:
                                df["correct_prediction"] = df["loss"] < df[df["fold"] == "ind_val"]["loss"].quantile(
                                    0.95)  # losswise definition
                                # df["correct_prediction"] = df["acc"]>=ind_val_acc   #accuracywise definition

                                df["shift"] = df["fold"].apply(
                                lambda x: x.split("_")[0] if "_0." in x else x)  # what kind of shift has occured?


                            df["shift_intensity"] = df["fold"].apply(
                                lambda x: x.split("_")[1] if "_" in x else x)  # what intensity?
                            df["ood"] = ~df["fold"].isin(["train", "ind_val", "ind_test"])
                            dfs.append(df)
                        except TypeError:
                            print(f"{dataset}_normal_{sampler}_{batch_size}_k={k}_{feature}.csv")
    return pd.concat(dfs)


def load_all(batch_size=30, samples=1000, feature="all", shift="normal", prefix="final_data", reduce=True):
    dfs = []
    for dataset in DATASETS:
        if feature!="all":
            dfs.append(load_pra_df(dataset, feature, model="", batch_size=batch_size, samples=samples, prefix=prefix, reduce=reduce, shift=shift))
        else:
            for dsd in DSDS:
               dfs.append(load_pra_df(dataset, dsd, model="", batch_size=batch_size, samples=samples, prefix=prefix, reduce=reduce, shift=shift))
    return pd.concat(dfs)


def load_pra_df(dataset_name, feature_name, model="" , batch_size=1, samples=1000, prefix="final_data", shift="normal", reduce=True):
    try:
        df = pd.concat(
        [pd.read_csv(join(prefix, fname)) for fname in os.listdir(prefix) if dataset_name in fname and feature_name in fname and model in fname  and shift in fname])
    except:
        print("no data found for ", dataset_name, feature_name)
        return pd.DataFrame()

    df["Dataset"]=dataset_name
    df["batch_size"]=batch_size
    if model!="":
        df["Model"]=model
    try:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    except:
        pass

    if batch_size!=1:
        df = df.groupby(["fold", "feature_name", "Dataset"]).apply(sample_loss_feature, samples, batch_size, reduce=reduce).reset_index()
    # ind_acc = df[df["fold"]=="ind_val"]["acc"].mean()

    if dataset_name=="Polyp":
        if batch_size==1 or not reduce:
            df["correct_prediction"] = df["loss"] < 0.5  # arbitrary threshold
        else:
            df["correct_prediction"] = df["loss"] < df[df["fold"]=="ind_val"]["loss"].max()  #maximum observed val mean jaccard
    else:
        # print(df[df["fold"]=="ind_val"]["loss"].quantile(0.05))
        if batch_size==1 or not reduce:
            df["correct_prediction"] = df["acc"]==1 #arbitrary threshold;
        else:
            df["correct_prediction"] = df["loss"] < df[df["fold"] == "ind_val"]["loss"].quantile(0.95)#losswise definition
            # df["correct_prediction"] = df["acc"]>=ind_val_acc   #accuracywise definition
    df["shift"] = df["fold"].apply(lambda x: x.split("_")[0] if "_0." in x else x)            #what kind of shift has occured?
    df["shift_intensity"] = df["fold"].apply(lambda x: x.split("_")[1] if "0." in x else "InD" if "ind" in x else "OoD")  #what intensity?
    df["ood"] = ~df["fold"].isin(["train", "ind_val", "ind_test"])
    df["batch_size"]=batch_size

    return df


DSD_PRINT_LUT = {"grad_magnitude": "GradNorm", "cross_entropy" : "Entropy", "energy":"Energy", "knn":"kNN", "mahalanobis":"Mahalanobis", "softmax":"Softmax", "typicality":"Typicality"}
DATASETS = ["CCT", "OfficeHome", "Office31", "NICO", "Polyp"]
DSDS = ["knn", "grad_magnitude", "cross_entropy", "energy", "typicality", "softmax", "rabanser"]
# BATCH_SIZES = [32]
BATCH_SIZES = [1, 8, 16, 32, 64]
THRESHOLD_METHODS = [ "val_optimal", "ind_span"]
DATASETWISE_RANDOM_LOSS = {
    "CCT": -np.log(1/15),
    "OfficeHome": -np.log(1/65),
    "Office31": -np.log(1/31),
    "NICO": -np.log(1/60),
    "Polyp": -10 #segmentation task; never incidentally correct
}
BIAS_TYPES = ["Unbiased", "Class", "Synthetic", "Temporal"]
SAMPLERS = ["RandomSampler",  "ClassOrderSampler", "ClusterSampler", "SequentialSampler",]
SYNTHETIC_SHIFTS = ["noise", "multnoise", "brightness", "contrast", "hue", "saltpepper", "saturation",  "smear"]

SAMPLER_LUT = dict(zip(SAMPLERS, BIAS_TYPES))
# BATCH_SIZES = np.arange(1, 64)
def load_polyp_data():
    dfs = []
    for dsd_name in ["energy", "knn", "grad_magnitude", "cross_entropy", "mahalanobis"]:
        for model in ["deeplabv3plus", "unet", "segformer"]:
            for ood_val in ["CVC-ClinicDB", "EndoCV2020", "EtisLaribDB"]:
                df = load_pra_df(dataset_name="Polyp", feature_name=dsd_name, model=model, batch_size=1, samples=1000)
                print(df.head(10))
                dsd = OODDetector(df, ood_val)
                df["verdict"] = df.apply(lambda row: dsd.predict(row), axis=1)
                df["ood_val"] = ood_val
                df["IoU"] = 1 - df["loss"]
                df["model"] = model
                df["feature_name"] = dsd_name
                dfs.append(df)
    df = pd.concat(dfs)
    # df = df[df["shift"] != "train"]
    print(df.head(10))
    df.replace(DSD_PRINT_LUT, inplace=True)
    return df
