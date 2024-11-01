import albumentations as alb
import torchvision.transforms as transforms
import PIL.Image as img
import torch
from torch.utils import data
import torchvision.transforms as transforms
import torch.nn as nn



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
