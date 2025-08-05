import torch
from torch.utils.data import Sampler, DataLoader
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
from torch.utils.data import RandomSampler
class ClassOrderSampler(Sampler):
    """
    Sampler that splits the data_source into classes, returns indexes in order of class
    Induces selection bias via label shift
    """
    def __init__(self, data_source, num_classes):
        super(ClassOrderSampler, self).__init__(data_source)
        self.data_source = data_source
        self.indices = [[] for i in range(num_classes)]

        #initial pass to sort the indices by class
        for i, data in enumerate(data_source):

            self.indices[min(data[1], num_classes-1)].append(i)


    def __iter__(self):
        return iter(sum(self.indices, []))

    def __len__(self):
        return len(self.data_source)


class SequentialSampler(Sampler):
    """
    Samples sequentially from the dataset (assuming the dataloader fetches subsequent frames e.g from a video)
    """
    def __init__(self, data_source):
        super(SequentialSampler, self).__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source))) #essentially: just prevents accidental shuffling

    def __len__(self):
        return len(self.data_source)

class ClusterSampler(Sampler):
    """
    Returns indices corresponding to a KMeans-clustering of the latent representations.
    (Artificial) selection bias
    """
    def __init__(self, data_source, rep_model, batch_size):
        super(ClusterSampler, self).__init__(data_source)
        self.data_source = data_source
        self.rep_model = rep_model
        self.rep_model.eval()
        self.batch_size = batch_size
        self.loader = DataLoader(data_source, batch_size=32, drop_last=True, shuffle=False)
        self.reps = np.zeros((len(self.loader), 32, rep_model.latent_dim))
        with torch.no_grad():
            for i, list in tqdm(enumerate(self.loader)):
                x=list[0].to("cuda").float()
                self.reps[i] = rep_model.get_encoding(x).cpu().numpy()
        self.reps = self.reps.reshape(-1, rep_model.latent_dim)
        self.num_clusters = int(len(self.data_source)//batch_size)
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit_predict(self.reps)

    def __iter__(self):
        return iter(np.concatenate([np.arange(len(self.loader)*32)[self.kmeans==i] for i in range(self.num_clusters)], axis=0))

    def __len__(self):
        return len(self.data_source)

