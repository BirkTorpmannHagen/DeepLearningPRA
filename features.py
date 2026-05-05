import torch
import torch.nn.functional as F
import torchvision.transforms
from torch.autograd import Variable

from components import ks_distance


def cross_entropy(model, image, num_features=1):
    out = model(image)
    return model.criterion(out, torch.ones_like(out))


def _final_linear_weight(model):
    last = None
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            last = m
    if last is None:
        raise ValueError("GradNorm: model has no nn.Linear or nn.Conv2d layer")
    return last.weight


def grad_magnitude(model, x, num_features=1):
    """GradNorm OOD score (Huang et al. 2021).

    L1 norm of the gradient of the cross-entropy between the softmax output
    and a uniform target, taken with respect to the weights of the final
    fully-connected layer. Computed per sample.
    """
    final_weight = _final_linear_weight(model)
    scores = torch.empty(x.shape[0], device=x.device)

    with torch.enable_grad():
        for i in range(x.shape[0]):
            xi = x[i:i+1].detach()
            output = model(xi)
            if isinstance(output, list):
                output = output[1]
            log_probs = F.log_softmax(output, dim=1)
            # CE(softmax(z), uniform) = - (1/C) * sum_c log p_c
            ce = -log_probs.mean()
            model.zero_grad(set_to_none=True)
            grads = torch.autograd.grad(ce, final_weight, retain_graph=False)[0]
            scores[i] = grads.abs().sum()
    return scores


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

