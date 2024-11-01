import pandas as pd

from tqdm import tqdm
import pickle as pkl
import numpy as np

# import torch_two_sample as tts
def list_to_str(some_list):
    return "".join([i.__name__ for i in some_list])

def process_dataframe(data, filter_noise=False, combine_losses=True, filter_by_sampler=""):
    # data = data[data["sampler"] != "ClassOrderSampler"]
    # print(pd.unique(data["sampler"]))
    if filter_by_sampler!="":
        data = data[data["sampler"]==filter_by_sampler]
    if "noise" in str(pd.unique(data["fold"])) and filter_noise:
        data = data[(data["fold"] == "noise_0.2") | (data["fold"] == "ind")]
    if isinstance(data["loss"], str):
        data["loss"] = data["loss"].str.strip('[]').str.split().apply(lambda x: [float(i) for i in x])
        if combine_losses:
            data["loss"] = data["loss"].apply(lambda x: np.mean(x))
        else:
            data=data.explode("loss")
    data["oodness"] = data["loss"] / data[data["fold"] == "ind"]["loss"].quantile(0.95)
    return data

def convert_stats_to_pandas_df(train_norms, train_loss, ind_val_features, ood_features, ind_val_losses, ood_losses, feature_name):
    dataset = []
    for i in range(len(train_norms)):
        dataset.append({"fold": "train", "feature_name":feature_name, "feature": train_norms[i], "loss": train_loss[i]})
    for fold, ind_fs in ind_val_features.items():
        for i in range(len(ind_fs)):
            dataset.append({"fold": fold, "feature_name":feature_name, "feature": ind_fs[i], "loss": ind_val_losses[fold][i]})

    for fold, ood_fs in ood_features.items():
        for i in range(len(ood_fs)):
            dataset.append({"fold": fold, "feature_name":feature_name, "feature": ood_fs[i], "loss": ood_losses[fold][i]})
    df = pd.DataFrame(dataset)
    return df

def convert_to_pandas_df(train_norms, train_losses, ind_val_features, ood_features, ind_val_losses, ood_losses, feature_names):

    dataframes = []
    for fi, feature_name in enumerate(feature_names):

        dataset = []
        for fold, train_fs in train_norms.items():
            for i in range(train_fs.shape[0]):
                print(train_norms)
                print(i)
                dataset.append({"fold": "train", "feature_name": feature_name, "feature": train_fs[i][fi], "loss": train_losses["train"][i]})
        print(dataset)
        for fold, ind_fs in ind_val_features.items():
            for i in range(ind_fs.shape[0]):
                dataset.append({"fold": fold, "feature_name":feature_name, "feature": ind_fs[i][fi], "loss": ind_val_losses[fold][i]})

        for fold, ood_fs in ood_features.items():
            for i in range(ood_fs.shape[0]):
                dataset.append({"fold": fold, "feature_name":feature_name, "feature": ood_fs[i][fi], "loss": ood_losses[fold][i]})
        df = pd.DataFrame(dataset)
        print(df["fold"].unique())
        dataframes.append(df)
    return dataframes

class BaseSD:
    def __init__(self, rep_model):
        self.rep_model = rep_model

    def register_testbed(self, testbed):
        self.testbed = testbed


class FeatureSD(BaseSD):
    """
    General class for gradient-based detectors, including jacobian.
    Computes a gradient norm/jacobian norm/hessian norm/etc
    """
    def __init__(self, rep_model, feature_fns, k=0):
        super().__init__(rep_model)
        self.feature_fns = feature_fns
        self.num_features=len(feature_fns)


    def save(self, var, varname):
        pkl.dump(var, open(f"cache_{list_to_str(self.feature_fns)}_{self.testbed.__class__.__name__}_{varname}.pkl", "wb"))

    def load(self, varname):
        return pkl.load(open(f"cache_{list_to_str(self.feature_fns)}_{self.testbed.__class__.__name__}_{varname}.pkl", "rb"))

    def get_features(self, dataloader):
        features = np.zeros((len(dataloader), self.testbed.batch_size, self.num_features))
        if self.testbed.batch_size==1:
            features = np.zeros((len(dataloader), self.num_features))

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            x = data[0].cuda()
            feats = np.zeros((self.num_features, self.testbed.batch_size))
            for j, feature_fn in enumerate(self.feature_fns):
                if feature_fn.__name__=="typicality":
                    features[i,:, j]=feature_fn(self.testbed.glow, x, self.train_test_norms).detach().cpu().numpy()
                else:
                    features[i,:, j]=feature_fn(self.rep_model, x, self.train_test_norms).detach().cpu().numpy()

        features = features.reshape((len(dataloader)*self.testbed.batch_size, self.num_features))

        return features

    def get_encodings(self, dataloader):
        features = np.zeros((len(dataloader), self.testbed.batch_size, self.rep_model.latent_dim))
        if self.testbed.batch_size==1:
            features = np.zeros((len(dataloader), self.num_features))
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

            x = data[0].cuda()
            features[i] = self.rep_model.get_encoding(x).detach().cpu().numpy()
        return features



    def compute_features_and_loss_for_loaders(self, dataloaders):

        features = dict(
            zip(dataloaders.keys(),
                          [self.get_features(loader)
                           for fold_name, loader in dataloaders.items()]))

        losses = dict(
            zip(dataloaders.keys(),
                [self.testbed.compute_losses(loader) for fold_name, loader in dataloaders.items()]))
        return features, losses

    def compute_pvals_and_loss(self):
        """

        :param sample_size: sample size for the tests
        :return: ind_p_values: p-values for ind fold for each sampler
        :return ood_p_values: p-values for ood fold for each sampler
        :return ind_sample_losses: losses for each sampler on ind fold, in correct order
        :return ood_sample_losses: losses for each sampler on ood fold, in correct order
        """
        # sample_size=min(sample_size, len(self.testbed.ind_val_loaders()[0]))

        # resubstitution estimation of entropy
        self.train_test_norms = self.get_encodings(self.testbed.ind_test_loader()["train_test"])

        try:
            train_norms = self.load("train_norms")
            train_loss = self.load("train_loss")
            train_test_norms = self.load("train_test_norms")
            train_test_losses = self.load("train_test_norms")
            ind_val_features = self.load("val_norms")
            ind_val_losses = self.load("val_loss")
        except FileNotFoundError as e:
            print(e)
            train_norms, train_loss = self.compute_features_and_loss_for_loaders(self.testbed.ind_loader())
            ind_val_features, ind_val_losses = self.compute_features_and_loss_for_loaders(self.testbed.ind_val_loaders())
            self.save(train_norms, "train_norms")
            self.save(train_loss, "train_loss")
            self.save(self.train_test_norms, "train_test_norms")
            self.save(ind_val_features, "val_norms")
            self.save(ind_val_losses, "val_loss")

        ood_features, ood_losses = self.compute_features_and_loss_for_loaders(self.testbed.ood_loaders())
        return train_norms, train_loss, ind_val_features, ood_features, ind_val_losses, ood_losses



def open_and_process(fname, filter_noise=False, combine_losses=True, exclude_sampler=""):
    try:
        data = pd.read_csv(fname)
        # data = data[data["sampler"] != "ClassOrderSampler"]
        # print(pd.unique(data["sampler"]))
        if exclude_sampler!="":
            data = data[data["sampler"]!=exclude_sampler]
        if "_" in str(pd.unique(data["fold"])) and filter_noise:
            folds =  pd.unique(data["fold"])
            prefix = folds[folds != "ind"][0].split("_")[0]
            max_noise = sorted([float(i.split("_")[1]) for i in pd.unique(data["fold"]) if "_" in i])[-1]
            data = data[(data["fold"] == f"{prefix}_{max_noise}") | (data["fold"] == "ind")]

        try:
            data["loss"] = data["loss"].map(lambda x: float(x))
        except:

            data["loss"] = data["loss"].str.strip('[]').str.split().apply(lambda x: [float(i) for i in x])
            if combine_losses:
                data["loss"] = data["loss"].apply(lambda x: np.mean(x))
            else:
                data=data.expbrlode("loss")
        data.loc[data['fold'] == 'ind', 'oodness'] = 0
        data.loc[data['fold'] != 'ind', 'oodness'] = 2
        # data["oodness"] = data["loss"] / data[data["fold"] == "ind"]["loss"].max()
        if filter_noise and "_" in str(pd.unique(data["fold"])):
            assert len(pd.unique(data["fold"])) == 2, f"Expected 2 unique folds, got {pd.unique(data['fold'])}"
        return data
    except FileNotFoundError:
        # print(f"File {fname} not found")
        return None
