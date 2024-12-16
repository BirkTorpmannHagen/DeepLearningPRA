import torch
import torch.nn.functional as F
import torchvision.transforms
from torch.autograd import Variable


def cross_entropy(model, image, num_features=1):
    out = model(image)
    if isinstance(out, list):
        out = out[1]  #for njord
    return model.criterion(out, torch.ones_like(out))


def grad_magnitude(model, x, num_features=1):
    image = x.detach().clone()
    image.requires_grad = True
    output = model(image)
    if isinstance(output, list):
        output = output[1]  #for njord
    loss = model.criterion(output, torch.ones_like(output)).mean()
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    data_grad.requires_grad=False
    image.requires_grad=False

    return torch.norm(torch.norm(data_grad, "fro", dim=(1,2)), "fro", dim=-1) #torch only likes 2-dims


def typicality(model, img, num_features=1):
    return -model.estimate_log_likelihood(img)


def knn(model, img, train_test_norms):

    train_test_norms = torch.Tensor(train_test_norms).cuda()
    train_test_norms = train_test_norms.view(-1, train_test_norms.shape[-1])
    encoded = model.get_encoding(img)
    min_dists = torch.zeros(encoded.shape[0])
    for bidx in range(encoded.shape[0]):
        dist = torch.norm(encoded[bidx]-train_test_norms, dim=1)
        min_dists[bidx] = torch.min(dist)
    return min_dists


def energy(model, img, num_features=1):
    energy = torch.logsumexp(model(img), dim=1)
    if len(energy.shape)>1:
        while energy.shape[-1]!=1:
            energy = torch.logsumexp(energy, dim=-1)
    return energy

def softmax(model, img, num_features=1):
    sm = F.softmax(model(img))
    feat = torch.max(sm, dim=1)[0]
    if len(feat.shape)>1:
        while feat.shape[-1]!=1:
            feat = torch.max(feat, dim=-1)[0]
    assert (feat>=0).all()
    return feat


if __name__ == '__main__':
    from classifier.resnetclassifier import ResNetClassifier

