import torch
import torch.nn.functional as F
import torchvision.transforms
from torch.autograd import Variable

from components import ks_distance


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

def rabanser_ks(model, img, train_test_norms):
    encoding = model.get_encoding(img)
    return ks_distance(encoding, train_test_norms)

def energy(model, img, num_features=1):
    energy = torch.logsumexp(model(img), dim=1)
    while len(energy.shape)>1:
        energy = torch.logsumexp(energy, dim=-1)
    return energy

def softmax(model, img, num_features=1):
    sm = F.softmax(model(img))
    feat = torch.max(sm, dim=1)[0]
    while len(feat.shape)!=1:
            feat = torch.max(feat, dim=-1)[0]
    return feat


def mahalanobis(model, image, featlist):
    model.eval()
    all_features = torch.Tensor(featlist).cuda()
    with torch.no_grad():
        # Get feature vector for input image: [1, C]
        feat = model.get_encoding(image.cuda()).squeeze(0)  # [C]

        # Compute mean and covariance from training features
        all_features = all_features

        feature_mean = torch.mean(all_features, dim=0)  # [C]
        centered = all_features - feature_mean  # [N, C]

        # Covariance and regularization
        cov = centered.T @ centered / (centered.shape[0] - 1)  # [C, C]
        cov += 1e-5 * torch.eye(cov.shape[0], device=feat.device)  # Regularization

        # Precision matrix
        precision = torch.inverse(cov)  # [C, C]

        # Mahalanobis distance
        delta = feat - feature_mean  # [C]
        m_dist = delta @ precision @ delta  # scalar

        return m_dist  # Return as float

# def odin(model, img, temperature=1000, epsilon=0.001):
#     img.requires_grad = True
#     outputs = model(img)
#     scaled_output = outputs / temperature
#     probs = F.softmax(scaled_output, dim=1)
#     max_score, pred = torch.max(probs, dim=1)
#     loss = F.cross_entropy(scaled_output, pred)
#     loss.backward()
#     gradient = torch.sign(img.grad.data)
#     perturbed = img - epsilon * gradient
#     outputs_perturbed = model(perturbed) / temperature
#     softmax_perturbed = F.softmax(outputs_perturbed, dim=1)
#     return torch.max(softmax_perturbed, dim=1)[0]


if __name__ == '__main__':
    from classifier.resnetclassifier import ResNetClassifier

