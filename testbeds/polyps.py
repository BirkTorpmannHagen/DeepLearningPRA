import numpy as np
import torch.nn
from torch.utils.data import ConcatDataset

from datasets.polyps import build_polyp_dataset
from vae.vae_experiment import VAEXperiment
from segmentor.deeplab import SegmentationModel
import yaml
from glow.plmodules import GlowPL

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
    def __init__(self,rep_model, mode="normal", model_name="deeplabv3plus", batch_size=16, sampler="RandomSampler"):
        super().__init__(num_workers=5, mode=mode, sampler=sampler, batch_size=batch_size)
        self.mode = mode

        self.ind_train, self.ind_val, self.ind_test, self.etis, self.cvc, self.endo = build_polyp_dataset("../../Datasets/Polyps")
        self.noise_range = np.arange(0.05, 0.3, 0.05)
        self.batch_size=batch_size
        #vae
        if rep_model=="vae":
            self.vae = VanillaVAE(in_channels=3, latent_dim=512).to("cuda").eval()
            vae_exp = VAEXperiment(self.vae, DEFAULT_PARAMS)
            vae_exp.load_state_dict(
                torch.load("vae_logs/PolypDataset/version_0/checkpoints/epoch=180-step=7240.ckpt")[
                    "state_dict"])

        #segmodel
        self.classifier = SegmentationModel.load_from_checkpoint(
            f"segmentation_logs/checkpoints/{model_name}/best.ckpt", model_name=model_name).to("cuda")
        self.classifier.eval()

        #assign rep model
        self.glow = GlowPL.load_from_checkpoint("glow_logs/KvasirSegmentationDataset/checkpoints/epoch=99-step=5000.ckpt", in_channel=3, n_flow=32,n_block=4, affine=True, conv_lu=True, optimizer="adam", batch_size=32, img_size=64, lr=1e-4)
        self.mode = mode

    def get_ood_dict(self):
        return {"EtisLaribDB":self.dl(self.etis),
                "CVC-ClinicDB":self.dl(self.cvc),
                "EndoCV2020":self.dl(self.endo)}

    def compute_losses(self, loader, reduce=False):
        if reduce:
            losses = np.zeros((len(loader), 3))
        else:
            losses = np.zeros((len(loader), self.batch_size, 3))
        with torch.no_grad():
            for i, data in tqdm(enumerate(loader), total=len(loader)):
                with torch.no_grad():
                    x = data[0].to("cuda")
                    y = data[1].to("cuda")
                    idx = data[2]
                    if reduce:
                        loss = self.classifier.compute_loss(x, y, reduce=reduce).cpu()
                        losses[i] = np.array([loss, 1-loss, idx.numpy().mean()])
                    else:
                        loss = self.classifier.compute_loss(x, y, reduce=reduce).cpu().unsqueeze(1)

                        losses[i] = torch.cat([loss, 1-loss, idx.unsqueeze(1)], dim=1).cpu()
        if reduce:
            return losses
        return losses.reshape(-1, 3)
