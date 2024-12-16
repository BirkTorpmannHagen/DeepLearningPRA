import torch.nn
from torch.utils.data import ConcatDataset

from datasets.polyps import build_polyp_dataset
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
    """
    Abstract class for testbeds; feel free to override for your own datasets!
    """
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
        return  {"ind_train":self.dl(self.ind_train)}

    def ind_val_loader(self):
        return  {"ind_val":self.dl(self.ind_val)}

    def ind_test_loader(self):
        return  {"ind_test":self.dl(self.ind_test)}

    def ood_loaders(self):
        if self.mode=="noise":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, additive_noise, "noise", noise)) for noise in self.noise_range]
            loaders = dict(zip(["noise_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        elif self.mode=="dropout":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, random_occlusion, "dropout", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["dropout_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        elif self.mode=="saturation":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, desaturate, "saturation", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["contrast_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        elif self.mode=="brightness":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, brightness_shift, "brightness", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["brightness_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        elif self.mode=="hue":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, hue_shift, "hue", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["hue_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        elif self.mode=="fgsm":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, targeted_fgsm, "fgsm", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["adv_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
            return loaders
        elif self.mode=="multnoise":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, multiplicative_noise, "multnoise", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["multnoise_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
            return loaders
        elif self.mode=="saltpepper":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, salt_and_pepper, "saltpepper", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["saltpepper_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        elif self.mode=="smear":
            ood_sets = [self.dl(TransformedDataset(self.ind_test, smear, "smear", noise)) for
                        noise in self.noise_range]
            loaders = dict(zip(["smear_{}".format(noise_val) for noise_val in self.noise_range], ood_sets))
        else:
            loaders =  self.get_ood_dict()
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


class PolypTestBed(BaseTestBed):
    def __init__(self,rep_model, mode="normal"):
        super().__init__()
        self.ind_train, self.ind_val, self.ind_test, self.etis, self.cvc, self.endo = build_polyp_dataset("../../Datasets/Polyps")
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
            "segmentation_logs/checkpoints/best.ckpt").to("cuda")
        self.classifier.eval()

        #assign rep model
        self.glow = Glow(3, 32, 4).cuda().eval()
        self.glow.load_state_dict(torch.load("glow_logs/Polyp_checkpoint/model_040001.pt"))
        self.mode = mode

    def get_ood_dict(self):
        return {"EtisLaribDB":self.dl(self.etis),
                "CVC-ClinicDB":self.dl(self.cvc),
                "EndoCV2020":self.dl(self.endo)}

    def compute_losses(self, loader):
        losses = np.zeros(len(loader))
        print("computing losses")
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            x = data[0].to("cuda")
            y = data[1].to("cuda")
            losses[i]=self.classifier.compute_loss(x,y).mean()
        return losses

