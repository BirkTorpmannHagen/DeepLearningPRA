import torch.nn
from torch.utils.data import ConcatDataset

from datasets.polyps import build_polyp_dataset
from vae.vae_experiment import VAEXperiment
from segmentor.deeplab import SegmentationModel
import yaml
from glow.model import Glow
from classifier.resnetclassifier import ResNetClassifier
from ooddetectors import *

from testbeds.base import BaseTestBed
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



class PolypTestBed(BaseTestBed):
    def __init__(self,rep_model, mode="normal"):
        super().__init__()
        self.mode = mode

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

        # #assign rep model
        # self.glow = Glow(3, 32, 4).cuda().eval()
        # self.glow.load_state_dict(torch.load("../glow_logs/Polyp_checkpoint/model_040001.pt"))
        # self.mode = mode

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

