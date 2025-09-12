import albumentations as alb
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils import data
import albumentations
import random
import torch.nn.functional as F

def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
def additive_noise(x, intensity):
    seed_all(0)
    noise = torch.randn_like(x) * intensity
    return x + noise

def desaturate(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    desat = albumentations.ColorJitter(hue=0, brightness=0, saturation=intensity*3, contrast=0, always_apply=True)
    transforms = alb.Compose([desat])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed

def contrast(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    desat = albumentations.ColorJitter(hue=0, brightness=0, saturation=0, contrast=intensity*3, always_apply=True)
    transforms = alb.Compose([desat])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed

def hue_shift(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    desat = albumentations.ColorJitter(hue=intensity*3, brightness=0, saturation=0, contrast=0, always_apply=True)
    transforms = alb.Compose([desat])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed

def brightness_shift(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    desat = albumentations.ColorJitter(brightness=intensity*3, hue=0, contrast=0, saturation=0, always_apply=True)
    transforms = alb.Compose([desat])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed

def multiplicative_noise(x, intensity):
    seed_all(0)
    noise = 1+torch.randn_like(x) * intensity*2
    return x * noise

def salt_and_pepper(x, intensity, scale=0.25):
    seed_all(0)
    noise = torch.rand_like(x)
    x = x.clone()
    scaled=intensity*scale
    x[noise<scaled] = 0
    x[noise>1-scale] = 1
    return x

def smear(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    desat = albumentations.GridDistortion(always_apply=True, distort_limit=intensity*2)
    transforms = alb.Compose([desat])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed


@torch.no_grad()
def _pick_targets(model, x, target=None):
    # Called under no_grad to avoid building a graph just to pick targets
    logits = model(x)
    if target is None:
        target = logits.argmin(dim=1)  # least-likely targeted attack
    return target

def targeted_fgsm(model, x, eps=1.0, clamp=(0,1), target=None):
    """
    Safe to call from DataLoader __getitem__.
    Produces a *CPU* tensor in [0,1].
    """
    # Ensure float input
    if not torch.is_floating_point(x):
        x = x.float()
    lo, hi = clamp

    # Make sure we’re on the same device as the model
    dev = next(model.parameters()).device

    # — Important for DataLoader context —
    # Some training scripts have global autocast/no_grad in weird places.
    # We explicitly enable grads and disable autocast here.
    torch.set_grad_enabled(True)
    autocast_enabled = torch.is_autocast_enabled()
    try:
        if autocast_enabled:
            torch.set_autocast_enabled(False)

        model_was_training = model.training
        model.eval()  # deterministic BN/Dropout; still differentiable

        x_adv = x.to(dev).detach().requires_grad_(True)

        # Pick target labels without tracking grads
        with torch.no_grad():
            target = _pick_targets(model, x_adv, target)

        logits = model(x_adv)
        loss = F.cross_entropy(logits, target, reduction='sum')

        # Clean any stale grads in case this gets called repeatedly
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        # Prefer backward + .grad for clearer debugging
        loss.backward()
        grad = x_adv.grad
        # print(grad)
        # Diagnostics you can temporarily print if needed:
        # print("grad dtype:", grad.dtype, "max abs:", grad.abs().max().item())

        # If grad is None or zero, bail early with the original image (no change)
        if grad is None or grad.abs().max().item() == 0.0:
            # Return as CPU in [0,1]
            return x.clamp(lo, hi).cpu()

        # Targeted FGSM minimizes CE to the target → step in *negative* grad direction
        g = (-grad).sign()            # <- this is the sign vector you care about
        x_adv = x_adv + eps * g
        x_adv = x_adv.clamp_(lo, hi).detach().to("cpu").squeeze(0)
        return x_adv
    finally:
        # Restore autocast if we changed it
        if autocast_enabled:
            torch.set_autocast_enabled(True)
        # (Don’t flip model back to train here; DataLoader shouldn’t mutate model state)
        torch.set_grad_enabled(False)

def random_occlusion(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    occlusion = albumentations.Cutout(int(intensity*100), max_h_size=int(x.shape[0]*0.1), max_w_size=int(x.shape[1]*0.1), always_apply=True)
    transforms = alb.Compose([occlusion])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed


class TransformedDataset(data.Dataset):
    #generic wrapper for adding noise to datasets
    def __init__(self, dataset, transform, transform_name, transform_param, model=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.transform_param = transform_param
        self.transform_name = transform_name
        self.model = model

    def __getitem__(self, index):

        batch = self.dataset.__getitem__(index)
        x = batch[0]
        rest = batch[1:]
        if self.transform_name=="fgsm":
            x = torch.clip(self.transform(self.model, x.unsqueeze(0), self.transform_param).squeeze(0).cpu(), 0, 1)
        else:
            x = torch.clip(self.transform(x, self.transform_param), 0, 1)
        # if index==0:
        #     plt.imshow(x.permute(1,2,0))
        #     plt.savefig(f"test_plots/{self.transform_name}_{self.transform_param}.png")
        #     plt.show()
        #     plt.close()
        return (x, *rest)

    def __str__(self):
        return f"{type(self.dataset).__name__}_{self.transform_name}_{str(self.transform_param)}"

    def __len__(self):
        # return 1000 #debug
        return self.dataset.__len__()


