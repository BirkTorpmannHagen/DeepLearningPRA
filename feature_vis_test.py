import matplotlib.pyplot as plt

from classifier.resnetclassifier import ResNetClassifier
from datasets.domain_datasets import MNIST3
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import seaborn as sns
if __name__ == '__main__':
    classifier = ResNetClassifier.load_from_checkpoint("MNIST3_logs/checkpoints/epoch=6-step=26250-v1.ckpt",
                                                        num_classes=10, resnet_version=101).cuda().eval()
    trans = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(), ])
    dl = torch.utils.data.DataLoader(MNIST3("../../Datasets/MNIST", train=False, transform=trans), batch_size=1, shuffle=True)
    encs =  [[] for i in range(10)]
    ys = []
    for i, (x, y) in tqdm(enumerate(dl), total=len(dl)):
        with torch.no_grad():
            if i>1000:
                break
            encs[y].append(classifier.get_encoding(x.cuda()).cpu().numpy())

    norms = np.zeros((10, 10))

    for y in range(10):
        for comp in range(10):
            norms[y, comp] = np.sum((np.array(encs[y])-np.array(encs[comp]))**2)

    sns.heatmap(norms.numpy(), annot=True, cmap="viridis")
    plt.show()
    fix, ax = plt.subplots(5,2)
    for i in range(10):
        encs[i] = np.array(encs[i]).mean(0)
        ax.flatten()[i].imshow(encs[i].reshape(32,64)[::2,::4], cmap="gray", interpolation="nearest")

    plt.tight_layout()
    plt.show()