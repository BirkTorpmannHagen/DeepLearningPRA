from tqdm import tqdm
from math import log
from datasets.eccv import *
from datasets.nico import *
from datasets.officehome import *
from datasets.office31 import *
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from glow.model import Glow
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_data(dataset, batch_size):
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=3)

    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=3
             )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3
    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p
    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def train(data, model, optimizer, img_size=32):
    dataset = iter(sample_data(data, 8))
    n_bins = 2.0 ** 5

    z_sample = []
    z_shapes = calc_z_shapes(3, img_size, 4, 4)
    for z in z_shapes:
        z_new = torch.randn(20, *z) * 0.7
        z_sample.append(z_new.to(device))

    with tqdm(range(42500)) as pbar:
        for i in pbar:
            image = next(dataset)[0]
            image = image.to(device)

            image = image * 255


            image = torch.floor(image / 2 ** (8 - 5))

            image = image / n_bins - 0.5


            log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, img_size, n_bins)
            model.zero_grad()
            loss.backward()
            # warmup_lr = 1e-4 * min(1, i * batch_size / (50000 * 10))
            warmup_lr = 1e-4
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )

            if i % 100 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model.reverse(z_sample).cpu().data,
                        f"glow_logs/sample/{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                    )

            if i % 10000 == 0:
                try:
                    os.mkdir(f"glow_logs/{data.__class__.__name__}_checkpoint")
                except FileExistsError:
                    pass
                torch.save(
                    model.state_dict(), f"glow_logs/{data.__class__.__name__}_checkpoint/model_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    optimizer.state_dict(), f"glow_logs/{data.__class__.__name__}_checkpoint/optim_{str(i + 1).zfill(6)}.pt"
                )


def train_new(dataset, img_size=32):
    model_single = Glow(3, 32, 4)
    # model = nn.DataParallel(model_single)
    model = model_single
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train(dataset, model, optimizer,img_size)


if __name__ == "__main__":
    trans = transforms.Compose([transforms.Resize((32,32)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    # dataset = CIFAR100("../../Datasets/CIFAR100", train=True, transform=trans, download=True)
    # train_new(dataset)
    # dataset = CIFAR10("../../Datasets/CIFAR10", train=True, transform=trans, download=True)
    # train_new(dataset)
    # dataset = MNIST3("../../Datasets/MNIST", train=True, transform=trans, download=True)
    # train_new(dataset)
    # dataset = EMNIST3("../../Datasets/EMNIST", train=True, transform=trans, download=True)
    # train_new(dataset)
    # dataset,_,_= build_njord_datasets(32)
    # dataset, _, _ = build_polyp_dataset("../../Datasets/Polyps", img_size=32)

    # trans = transforms.Compose([transforms.Resize((32,32)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    # dataset, _ = build_imagenette_dataset("../../Datasets/imagenette2", trans, trans)
    # dataset, _,_ = build_officehome_dataset("../../Datasets/OfficeHome", train_transform=trans, val_transform=trans)
    # train_new(dataset, img_size=32)

    dataset, _,_ = build_office31_dataset("../../Datasets/office31", train_transform=trans, val_transform=trans)
    train_new(dataset, img_size=32)

    dataset, _,_, _ = build_ECCV("../../Datasets/ECCV", train_transform=trans, val_transform=trans)
    train_new(dataset, img_size=32)

    # trans = transforms.Compose([transforms.Resize((32,32)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    # dataset, _ = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="dim", seed=0)

    # train_new(dataset, img_size=32)
