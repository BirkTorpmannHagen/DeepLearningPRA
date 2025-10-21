import os
from os.path import join

import numpy as np
import pandas as pd
import itertools

def mean_pairwise_abs_diff(values):
    vals = list(values)
    if len(vals) < 2:
        return np.nan
    diffs = [abs(a - b) for a, b in itertools.combinations(vals, 2)]
    return float(np.mean(diffs))

def sample_loss_feature(group, n_samples, n_size, stratisfication=False):
    samples = []
    for i in range(n_samples):
        if stratisfication:

            largest_ind_val_loss = np.max(
                [group[group["fold"] == "ind_val"].sample(n=n_size)["loss"] for _ in range(n_samples)])

            strat_size = round(n_size * stratisfication)
            if strat_size==0:
                sample = group[group["shift_intensity"]=="InD"].sample(n=n_size, replace=True)  # Sampling only OOD samples
            elif strat_size==n_size:
                sample= group[group["shift_intensity"]=="OoD"].sample(n=n_size, replace=True)  # Sampling InD samples
            else:
                sample_ind = group[group["shift_intensity"]=="InD"].sample(n=n_size-strat_size, replace=True)
                sample_ood = group[group["shift_intensity"]=="OoD"].sample(n=strat_size, replace=True)  # Sampling OoD Organic Samples
                sample = pd.concat([sample_ind, sample_ood])

        else:
            sample = group.sample(n=n_size, replace=True)  # Sampling with replacement

        sample["index"] = i
        if i==0 and stratisfication:
            print(stratisfication)
            print(sample)

        if sample["Dataset"].all() == "Polyp":
            mean_loss = sample['loss'].median()
        else:
            mean_loss = sample['loss'].mean()

        mean_feature = sample['feature'].mean()
        samples.append({'loss': mean_loss, 'feature': mean_feature, "acc": sample["acc"].mean(), "index": i})
    return pd.DataFrame(samples)



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
                        if feature=="knn" and k!=-1:
                            continue
                        if feature=="rabanser" and k==-1:
                            continue
                        try:
                            df = pd.read_csv(join(prefix, f"{dataset}_normal_{sampler}_{batch_size}_k={k}_{feature}.csv"), converters={
    "feature": lambda x: float(x) if x not in ("[]", "") else 0.0})
                        except FileNotFoundError:
                            print(f"No data found for {prefix}/{dataset}_normal_{sampler}_{batch_size}_k={k}_{feature}.csv")
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


def load_all(batch_size=30, samples=100, feature="all", shift="normal", prefix="fine_data", stratisfication=False, groupbyfolds=True):
    dfs = []
    for dataset in DATASETS:
        if feature!="all":
            dfs.append(load_pra_df(dataset, feature, model="", batch_size=batch_size, samples=samples, prefix=prefix, stratisfication=stratisfication, shift=shift, groupbyfolds=groupbyfolds))
        else:
            for dsd in DSDS:
                dfs.append(load_pra_df(dataset, dsd, model="", batch_size=batch_size, samples=samples, prefix=prefix, stratisfication=stratisfication, shift=shift, groupbyfolds=groupbyfolds))
    return pd.concat(dfs)


def load_pra_df(dataset_name, feature_name, model="" , batch_size=1, samples=1000, prefix="coarse_data", shift="normal", stratisfication=False, groupbyfolds=True):
    if dataset_name=="Polyp" and feature_name=="softmax":
        return pd.DataFrame() #softmax does not work for segmentation
    try:
            df = pd.concat([pd.read_csv(join(prefix, fname)) for fname in os.listdir(prefix) if dataset_name in fname and feature_name in fname and model in fname  and shift in fname])
    except:
        print("no data found for ", dataset_name, feature_name)
        return pd.DataFrame()

    df["Dataset"]=dataset_name
    df["batch_size"]=batch_size
    if groupbyfolds:
        df["shift"] = df["fold"].apply(lambda x: x.split("_")[0] if "_0." in x else x)  # what kind of shift has occured?
        df["shift_intensity"] = df["fold"].apply(
            lambda x: x.split("_")[1] if "0." in x else "InD" if "ind" in x else "Train" if "train" in x else "OoD")  # what intensity?
    if model!="":
        df["Model"]=model
    try:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    except:
        pass
    sampled_ind_val_loss = np.quantile(np.array([df[df["fold"] == "ind_val"]["loss"].sample(batch_size).mean() for _ in range(samples)]), 0.95)

    if batch_size!=1:
        if groupbyfolds:
            df = df.groupby(["fold", "feature_name", "Dataset"]).apply(sample_loss_feature, samples, batch_size, stratisfication=False).reset_index()
        else:
            df = df.groupby(["feature_name", "Dataset"]).apply(sample_loss_feature, samples, batch_size, stratisfication=stratisfication).reset_index()
    # ind_acc = df[df["fold"]=="ind_val"]["acc"].mean()

    if dataset_name=="Polyp":
        if batch_size==1:
            df["correct_prediction"] = df["loss"] < 0.5  # arbitrary threshold
        else:
            df["correct_prediction"] = df["loss"] < sampled_ind_val_loss  #maximum observed val mean jaccard
    else:
        # print(df[df["fold"]=="ind_val"]["loss"].quantile(0.05))
        if batch_size==1:
            df["correct_prediction"] = df["acc"]==1 #arbitrary threshold;
        else:
            df["correct_prediction"] = df["loss"] < sampled_ind_val_loss#losswise definition
            # df["correct_prediction"] = df["acc"]>=ind_val_acc   #accuracywise definition
    if groupbyfolds:
        df["ood"] = ~df["fold"].isin(["train", "ind_val", "ind_test"])

    df["shift"] = df["fold"].apply(
        lambda x: x.split("_")[0] if "_0." in x else x)  # what kind of shift has occured?
    df["shift_intensity"] = df["fold"].apply(
        lambda x: x.split("_")[
            1] if "0." in x else "InD" if "ind" in x else "Train" if "train" in x else "OoD")  # what intensity?
    df["batch_size"]=batch_size
    df["Organic"] = df["shift_intensity"].isin(["InD", "OoD"])
    return df


DSD_PRINT_LUT = {"grad_magnitude": "GradNorm", "cross_entropy" : "Entropy", "energy":"Energy", "knn":"kNN", "mahalanobis":"Mahalanobis", "softmax":"Softmax", "typicality":"Typicality"}
DSD_LUT = {value: key for key, value in DSD_PRINT_LUT.items()}
DATASETS = ["CCT", "OfficeHome", "Office31", "NICO", "Polyp"]
DSDS = ["knn", "grad_magnitude", "cross_entropy", "energy", "typicality", "softmax", "rabanser"]

BATCH_SIZES = [1, 8, 16, 32, 64]
THRESHOLD_METHODS = [ "val_optimal", "ind_span", "logistic"]

DATASETWISE_RANDOM_LOSS = {
    "CCT": -np.log(1/15),
    "OfficeHome": -np.log(1/65),
    "Office31": -np.log(1/31),
    "NICO": -np.log(1/60),
    "Polyp": 1 #segmentation task; never incidentally correct
}
DATASETWISE_RANDOM_CORRECTNESS = {
    "CCT": 1/15,
    "OfficeHome": 1/65,
    "Office31": 1/31,
    "NICO": 1/60,
    "Polyp": 0 #segmentation task; never incidentally correct
}
COLUMN_PRINT_LUT = {"feature_name":"Feature", "loss":"Loss", "rate":"p(E)", "shift_intensity":"Shift Intensity", "shift":"Shift", "feature": "Feature Value"}
BIAS_TYPES = ["Unbiased", "Class", "Synthetic", "Temporal"]
SAMPLERS = ["RandomSampler",  "ClassOrderSampler", "ClusterSampler", "SequentialSampler",]
SYNTHETIC_SHIFTS = ["noise", "multnoise", "hue", "saltpepper", "saturation", "brightness", "contrast", "smear", "adv", "fgsm"]
SHIFT_PRINT_LUT= {"normal": "Organic", "noise": "Additive Noise", "multnoise": "Multiplicative Noise",
             "hue": "Hue", "saltpepper": "Salt+Pepper Noise", "brightness":"Brightness", "contrast":"Contrast", "smear":"Smear", "adv": "FGSM"}

SAMPLER_LUT = dict(zip(SAMPLERS, BIAS_TYPES))
# BATCH_SIZES = np.arange(1, 64)
