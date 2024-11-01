import torch.nn
from torch.utils.data import ConcatDataset

from datasets.nico import build_nico_dataset
from datasets.njord_dataset import build_njord_datasets
from datasets.office31 import build_office31_dataset
from datasets.officehome import build_officehome_dataset
from datasets.polyps import build_polyp_dataset
from datasets.eccv import build_ECCV
from vae.vae_experiment import VAEXperiment
from segmentor.deeplab import SegmentationModel
import yaml
from glow.model import Glow
from classifier.resnetclassifier import ResNetClassifier
from ooddetectors import *


from datasets.synthetic_shifts import *
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
import torch.nn as nn
from vae.models.vanilla_vae import VanillaVAE
import torchvision.transforms as transforms
# import segmentation_models_pytorch as smp
DEFAULT_PARAMS = {
    "LR": 0.00005,
    "weight_decay": 0.0,
    "scheduler_gamma": 0.95,
    "kld_weight": 0.00025,
    "manual_seed": 1265

}
class BaseTestBed:
    def __init__(self, num_workers=5, mode="normal"):
        self.mode=mode
        self.num_workers=5
        self.noise_range = np.arange(0.0, 0.35, 0.05)[1:]
        self.batch_size = 16

    def compute_losses(self, loaders):
        pass

    def dl(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True)


    def ind_loader(self):
        return  {"train":self.dl(self.ind)}
    def ind_test_loader(self):
        return  {"train_test":self.dl(self.ind_test)}

    def ood_loaders(self):
        if self.mode=="noise":
            ood_sets = [self.dl(TransformedDataset(self.ind_val, additive_noise, "noise", noise)) for noise in self.noise_range]
            dicted = dict(zip(["noise_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
            return dicted
        elif self.mode=="dropout":
            ood_sets = [self.dl(TransformedDataset(self.ind_val, random_occlusion, "dropout", noise)) for
                        noise in self.noise_range]
            dicted = dict(zip(["dropout_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
            return dicted
        elif self.mode=="saturation":
            ood_sets = [self.dl(TransformedDataset(self.ind_val, desaturate, "saturation", noise)) for
                        noise in self.noise_range]
            dicted = dict(zip(["contrast_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
            return dicted
        elif self.mode=="brightness":
            ood_sets = [self.dl(TransformedDataset(self.ind_val, brightness_shift, "brightness", noise)) for
                        noise in self.noise_range]
            dicted = dict(zip(["brightness_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
            return dicted
        elif self.mode=="hue":
            ood_sets = [self.dl(TransformedDataset(self.ind_val, hue_shift, "hue", noise)) for
                        noise in self.noise_range]
            dicted = dict(zip(["hue_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
            return dicted
        elif self.mode=="fgsm":
            ood_sets = [self.dl(TransformedDataset(self.ind_val, targeted_fgsm, "fgsm", noise)) for
                        noise in self.noise_range]
            dicted = dict(zip(["adv_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
            return dicted
        elif self.mode=="multnoise":
            ood_sets = [self.dl(TransformedDataset(self.ind_val, multiplicative_noise, "multnoise", noise)) for
                        noise in self.noise_range]
            dicted = dict(zip(["multnoise_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
            return dicted
        elif self.mode=="saltpepper":
            ood_sets = [self.dl(TransformedDataset(self.ind_val, salt_and_pepper, "saltpepper", noise)) for
                        noise in self.noise_range]
            dicted = dict(zip(["saltpepper_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
            return dicted
        elif self.mode=="smear":
            ood_sets = [self.dl(TransformedDataset(self.ind_val, smear, "smear", noise)) for
                        noise in self.noise_range]
            dicted = dict(zip(["smear_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
            return dicted
        else:
            loaders =  {"ood": self.dl(self.ood)}
            return loaders


    def ind_val_loaders(self):
        loaders =  {"ind": self.dl(self.ind_val)}
        return loaders


    def compute_losses(self, loader):
        losses = np.zeros((len(loader), self.batch_size))
        criterion = nn.CrossEntropyLoss(reduction="none")  # still computing loss for each sample, just batched
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            with torch.no_grad():
                x = data[0].to("cuda")
                y = data[1].to("cuda")
                yhat = self.classifier(x)
                losses[i] = criterion(yhat, y).cpu().numpy()
        return losses.flatten()
class Office31TestBed(BaseTestBed):
    def __init__(self, sample_size, rep_model="vae", mode="severity"):
        super().__init__(sample_size)
        self.trans = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(), ])
        self.ind, self.ind_val, self.ood = build_office31_dataset("../../Datasets/office31", self.trans, self.trans)

        self.num_classes = num_classes = self.ind.num_classes
        self.contexts = len(self.ind.contexts)
        # self.ind, self.ind_test = random_split(self.ind, [0.5, 0.5])
        range1 = range(int(0.5 * len(self.ind)))
        range2 = range(int(0.5 * len(self.ind)) + 2, int(len(self.ind)))
        assert len(set(range1).intersection(range2)) == 0
        self.ind_test = torch.utils.data.Subset(self.ind, range1)
        self.ind = torch.utils.data.Subset(self.ind, range2)

        self.classifier = ResNetClassifier.load_from_checkpoint(
            "Office31Dataset_logs/checkpoints/epoch=95-step=13536.ckpt", num_classes=num_classes,
            resnet_version=101).to("cuda").eval()
        self.glow = Glow(3, 32, 4).cuda().eval()
        self.glow.load_state_dict(torch.load("glow_logs/Office31Dataset_checkpoint/model_040001.pt"))
        # self.rep_model = self.glow
        # self.vae = VanillaVAE(3, 512).to("cuda").eval()
        # self.rep_model = self.vae
        self.mode = mode

class OfficeHomeTestBed(BaseTestBed):
    def __init__(self, sample_size, rep_model="vae", mode="severity"):
        super().__init__(sample_size)
        self.trans = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(), ])
        self.ind, self.ind_val, self.ood = build_officehome_dataset("../../Datasets/OfficeHome", self.trans, self.trans)

        self.num_classes = num_classes = self.ind.num_classes
        self.contexts = len(self.ind.contexts)
        # self.ind, self.ind_test = random_split(self.ind, [0.5, 0.5])
        range1 = range(int(0.5 * len(self.ind)))
        range2 = range(int(0.5 * len(self.ind)) + 2, int(len(self.ind)))
        assert len(set(range1).intersection(range2)) == 0
        self.ind_test = torch.utils.data.Subset(self.ind, range1)
        self.ind = torch.utils.data.Subset(self.ind, range2)

        self.classifier = ResNetClassifier.load_from_checkpoint(
            "OfficeHome_logs/checkpoints/epoch=153-step=33572.ckpt", num_classes=num_classes,
            resnet_version=101).to("cuda").eval()
        self.glow = Glow(3, 32, 4).cuda().eval()
        self.glow.load_state_dict(torch.load("glow_logs/OfficeHome_checkpoint/model_040001.pt"))
        # self.rep_model = self.glow
        # self.vae = VanillaVAE(3, 512).to("cuda").eval()
        # self.rep_model = self.vae
        self.mode = mode

class ECCVTestBed(BaseTestBed):
    def __init__(self, sample_size, rep_model="vae", mode="severity"):
        super().__init__(sample_size)
        self.trans = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(), ])
        self.ind, self.ind_val, self.ind_test, self.ood = build_ECCV("../../Datasets/ECCV", self.trans, self.trans)
        self.num_classes = num_classes = self.ind.num_classes
        self.classifier = ResNetClassifier.load_from_checkpoint(
            "ECCV_logs/checkpoints/epoch=96-step=85360.ckpt", num_classes=num_classes,
            resnet_version=101).to("cuda").eval()
        self.glow = Glow(3, 32, 4).cuda().eval()
        self.glow.load_state_dict(torch.load("glow_logs/ECCV_checkpoint/model_040001.pt"))

        # self.rep_model = self.glow
        # self.vae = VanillaVAE(3, 512).to("cuda").eval()
        # self.rep_model = self.vae
        self.mode = mode



class PolypTestBed(BaseTestBed):
    def __init__(self,rep_model, mode="normal"):
        super().__init__()
        self.ind, self.ind_val, self.ood = build_polyp_dataset("../../Datasets/Polyps", ex=False)
        self.noise_range = np.arange(0.05, 0.3, 0.05)
        self.batch_size=1
        #vae
        if rep_model=="vae":
            self.vae = VanillaVAE(in_channels=3, latent_dim=512).to("cuda").eval()
            vae_exp = VAEXperiment(self.vae, DEFAULT_PARAMS)
            vae_exp.load_state_dict(
                torch.load("vae_logs/PolypDataset/version_0/checkpoints/epoch=180-step=7240.ckpt")[
                    "state_dict"])

        #segmodel
        self.classifier = SegmentationModel.load_from_checkpoint(
            "segmentation_logs/lightning_logs/version_14/checkpoints/epoch=199-step=64600.ckpt").to("cuda")
        self.classifier.eval()

        #assign rep model
        if rep_model == "vae":
            self.rep_model = self.vae
        elif rep_model=="glow":
            self.glow = Glow(3, 32, 4).cuda().eval()
            self.glow.load_state_dict(torch.load("glow_logs/Polyp_checkpoint/model_040001.pt"))
            self.rep_model = self.glow
        else:
            self.rep_model = self.classifier

        self.mode = mode


    def compute_losses(self, loader):
        losses = np.zeros(len(loader))
        print("computing losses")
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            x = data[0].to("cuda")
            y = data[1].to("cuda")
            losses[i]=self.classifier.compute_loss(x,y).mean()
        return losses

