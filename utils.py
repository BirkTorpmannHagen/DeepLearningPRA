import os
from os.path import join

import pandas as pd
import torch
import torch.nn as nn



ETISLARIB = "EtisLaribDB" #training set
CVCCLINIC = "CVC-ClinicDB" #validation set
ENDOCV = "EndoCV2020" #test set


class WrappedResnet(nn.Module):
    def __init__(self, model, input_size=32):
        super().__init__()
        self.model = model
        self.latent_dim = self.get_encoding_size(input_size)
        print(self.latent_dim)

    def get_encoding_size(self, input_size):
        dummy = torch.zeros((1, 3, input_size, input_size)).to("cuda")
        return self.get_encoding(dummy).shape[-1]

    def get_encoding(self, X):
        return torch.nn.Sequential(*list(self.model.children())[:-1])(X).flatten(1)

    def forward(self, x):
        return self.model(x)



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


def load_pra_df(feature_name, batch_size=30, samples=100):
    df = pd.concat(
        [pd.read_csv(join("single_data", fname)) for fname in os.listdir("single_data") if "Polyp" in fname])
    df.drop(columns=["Unnamed: 0"], inplace=True)
    df = df[df["feature_name"] == feature_name]
    df = df[df["fold"]!="ind"]
    # df["intensity"] = df["shift"].apply(lambda x: x.split("_")[1] if "_" in x else x)
    df = df[~df["fold"].isin(["smear", "saturation", "brightness"])]
    if batch_size!=1:
        def sample_loss_feature(group, n_samples, n_size):
            samples = []
            for i in range(n_samples):
                sample = group.sample(n=n_size, replace=True)  # Sampling with replacement
                # input()
                mean_loss = sample['loss'].mean()
                mean_feature = sample['feature'].mean()
                samples.append({'loss': mean_loss, 'feature': mean_feature})
                # samples.append({'loss': mean_loss, 'feature': brunnermunzel(df[df["fold"]=="train"]["feature"], sample['feature'])[0], "reduction":"bm"})
            return pd.DataFrame(samples)
            # Return a DataFrame of means with the original group keys

        cols = list(df.columns)
        cols.remove("loss")
        cols.remove("feature")

        df = df.groupby(cols).apply(sample_loss_feature, samples, batch_size).reset_index()
        df.drop(columns=["level_2"], inplace=True)
    df["correct_prediction"] = df["loss"] < 0.5  # arbitrary threshold
    df["shift"] = df["fold"].apply(lambda x: x.split("_")[0] if "_0." in x else x)            #what kind of shift has occured?
    df["shift_intensity"] = df["fold"].apply(lambda x: x.split("_")[1] if "_" in x else x)  #what intensity?
    df["ood"] = ~df["fold"].isin(["train", "ind_val", "ind_test"])#&~df["correct_prediction"] #is the prediction correct?
    return df
