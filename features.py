import torch
import torch.nn.functional as F
import torchvision.transforms
from torch.autograd import Variable

# def odin(model, image, feature_transform):
#     #modified odin; no temperature scaling. in effect just xent of perturbed data
#     image.requires_grad = True
#     output = model(image)
#     if isinstance(output, list):
#         output = output[1]  # for njord
#     nnOutputs = output
#     nnOutputs = nnOutputs - torch.max(nnOutputs)
#     nnOutputs = torch.exp(nnOutputs) / torch.sum(torch.exp(nnOutputs))
#     # nnOutputs = nnOutputs.unsqueeze(0)
#     maxIndexTemp = torch.argmax(nnOutputs)
#     loss = model.criterion(nnOutputs, torch.ones_like(output) * maxIndexTemp).mean()
#     loss.backward()
#     data_grad = image.grad.data
#     data_grad = data_grad.squeeze(0)
#     # perturb image
#     perturbed_image = image + data_grad.sign() * 0.1
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     val =  cross_entropy(model, perturbed_image)
#     return val

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

def odin(model, img, encodings):
    noiseMagnitude1 = 0.1

    temper = 2
    inputs = Variable(img, requires_grad=True)
    outputs = model(inputs)

    nnOutputs = outputs.data[0]
    nnOutputs = nnOutputs - torch.max(nnOutputs)
    nnOutputs = torch.exp(nnOutputs) / torch.sum(torch.exp(nnOutputs))
    outputs = outputs / temper

    maxIndexTemp = torch.argmax(nnOutputs)
    # labels = Variable(torch.LongTensor([maxIndexTemp]).cuda())
    labels = torch.ones_like(outputs)*maxIndexTemp
    labels = Variable(labels.cuda())
    loss = model.criterion(outputs, labels).mean()
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    # Normalizing the gradient to the same space of image
    gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
    gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
    gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data
    nnOutputs = nnOutputs
    nnOutputs = nnOutputs - torch.max(nnOutputs)
    nnOutputs = torch.exp(nnOutputs) / torch.sum(torch.exp(nnOutputs))
    # print(nnOutputs.shape)
    return torch.max(nnOutputs, dim=1)[0]

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

