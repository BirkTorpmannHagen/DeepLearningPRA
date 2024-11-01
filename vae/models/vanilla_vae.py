import matplotlib.pyplot as plt

from vae.models.base import BaseVAE
from torch import nn
from .types_ import *
import torch
import torch.nn.functional as F
import torchvision.models as models
import classifier.cifarresnet as cifarresnet
from torch.autograd import Variable

from torch.distributions import Normal

def gaussian_likelihood(x, mu_x, sigma_x):
    constant_term = 1 / torch.sqrt(2 * torch.pi * sigma_x ** 2)
    exponent_term = -((x - mu_x) ** 2) / (2 * sigma_x ** 2)
    likelihood = constant_term * torch.exp(exponent_term)

    # Calculate the likelihood for each data point and take the average across all data points
    return torch.mean(likelihood)


# class CIFARVAE(BaseVAE):
#     def __init__(self, image_size, channel_num, kernel_num, z_size):
#         # configurations
#         super().__init__()
#         self.image_size = image_size
#         self.channel_num = channel_num
#         self.kernel_num = kernel_num
#         self.z_size = self.latent_dim =  z_size
#
#         # encoder
#         self.encoder = nn.Sequential(
#             self._conv(channel_num, kernel_num // 4),
#             self._conv(kernel_num // 4, kernel_num // 2),
#             self._conv(kernel_num // 2, kernel_num),
#         )
#
#         # encoded feature's size and volume
#         self.feature_size = image_size // 8
#         self.feature_volume = kernel_num * (self.feature_size ** 2)
#
#         # q
#         self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
#         self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)
#
#         # projection
#         self.project = self._linear(z_size, self.feature_volume, relu=False)
#
#         # decoder
#         self.decoder = nn.Sequential(
#             self._deconv(kernel_num, kernel_num // 2),
#             self._deconv(kernel_num // 2, kernel_num // 4),
#             self._deconv(kernel_num // 4, channel_num),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # encode x
#         encoded = self.encoder(x)
#
#         # sample latent code z from q given x.
#         mean, logvar = self.q(encoded)
#         z = self.z(mean, logvar)
#
#         z_projected = self.project(z).view(
#             -1, self.kernel_num,
#             self.feature_size,
#             self.feature_size,
#         )
#
#         # reconstruct x from z
#         x_reconstructed = self.decoder(z_projected)
#
#         # return the parameters of distribution of q given x and the
#         # reconstructed image.
#         # return (mean, logvar), x_reconstructed
#         return  x_reconstructed, x, mean, logvar
#
#         # ==============
#         # VAE components
#         # ==============
#
#
#     def decode(self, z):
#         # z_projected = self.project(z).view(
#         #     -1, self.kernel_num,
#         #     self.feature_size,
#         #     self.feature_size,
#         # )
#         z_projected = z
#         # reconstruct x from z
#         x_reconstructed = self.decoder(z_projected)
#         return x_reconstructed
#     def q(self, encoded):
#         unrolled = encoded.view(-1, self.feature_volume)
#         return self.q_mean(unrolled), self.q_logvar(unrolled)
#
#     def z(self, mean, logvar):
#         std = logvar.mul(0.5).exp_()
#         eps = (
#             Variable(torch.randn(std.size())).cuda() if self._is_on_cuda else
#             Variable(torch.randn(std.size()))
#         )
#         return eps.mul(std).add_(mean)
#
#     def loss_function(self,
#                       *args,
#                       **kwargs) -> dict:
#         """
#         Computes the VAE loss function.
#         KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
#         :param args:
#         :param kwargs:
#         :return:
#         """
#         recons = args[0]
#         input = args[1]
#         mu = args[2]
#         log_var = args[3]
#
#         kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
#         recons_loss = F.mse_loss(recons, input)
#
#         kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
#
#         loss = recons_loss + kld_weight * kld_loss
#         return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}
#
#         # =====
#         # Utils
#         # =====
#
#     def sample(self, size, current_device: int, **kwargs):
#         z = Variable(
#             torch.randn(size, self.z_size).cuda() if self._is_on_cuda() else
#             torch.randn(size, self.z_size)
#         )
#         z_projected = self.project(z).view(
#             -1, self.kernel_num,
#             self.feature_size,
#             self.feature_size,
#         )
#         return self.decoder(z_projected).data
#
#     def _is_on_cuda(self):
#         return next(self.parameters()).is_cuda
#
#     # ======
#     # Layers
#     # ======
#
#     def _conv(self, channel_size, kernel_num):
#         return nn.Sequential(
#             nn.Conv2d(
#                 channel_size, kernel_num,
#                 kernel_size=4, stride=2, padding=1,
#             ),
#             nn.BatchNorm2d(kernel_num),
#             nn.ReLU(),
#         )
#
#     def _deconv(self, channel_num, kernel_num):
#         return nn.Sequential(
#             nn.ConvTranspose2d(
#                 channel_num, kernel_num,
#                 kernel_size=4, stride=2, padding=1,
#             ),
#             nn.BatchNorm2d(kernel_num),
#             nn.ReLU(),
#         )
#
#     def _linear(self, in_size, out_size, relu=True):
#         return nn.Sequential(
#             nn.Linear(in_size, out_size),
#             nn.ReLU(),
#         ) if relu else nn.Linear(in_size, out_size)
#     def encode(self, input: Tensor) -> List[Tensor]:
#         """
#         Encodes the input by passing through the encoder network
#         and returns the latent codes.
#         :param input: (Tensor) Input tensor to encoder [N x C x H x W]
#         :return: (Tensor) List of latent codes
#         """
#         result = self.encoder(input)
#         result = torch.flatten(result, start_dim=1)
#
#         # Split the result into mu and var components
#         # of the latent Gaussian distribution
#         mu = self.q_mean(result)
#         log_var = self.q_logvar(result)
#
#         return [mu, log_var]
#     def estimate_log_likelihood(self, sample, num_samples=100, prior_mean=0, prior_std=1):
#         """
#         Estimate the log likelihood for a given sample using a trained VAE and importance sampling.
#
#         Args:
#             vae: The trained VAE model.
#             sample: A single CIFAR10 image (shape: [3, 32, 32]).
#             num_samples: Number of samples to use for the Monte Carlo estimate (default: 100).
#             prior_mean: Mean of the prior distribution (default: 0).
#             prior_std: Standard deviation of the prior distribution (default: 1).
#
#         Returns:
#             log_likelihood: The estimated log likelihood for the given sample.
#         """
#
#
#
#         # Encode the sample to obtain the (mu, sigma) encoding
#         mu, log_sigma = self.encode(sample)
#         sigma = torch.exp(log_sigma)
#         sigma = sigma + 1e-5
#         # Compute the prior and learned distributions
#         prior_dist = Normal(prior_mean, prior_std)
#         learned_dist = Normal(mu, sigma)
#
#         # Sample multiple times from the learned latent distribution using the reparameterization trick
#         epsilon = torch.randn(num_samples, sigma.shape[-1]).to("cuda")
#         z = mu + epsilon * sigma
#
#         # Decode the latent variables to reconstruct the samples
#         reconstructions = self.decode(z)
#
#         # Compute the log probabilities of the reconstructions under the learned and prior distributions
#         log_probs_prior = prior_dist.log_prob(z).sum(dim=1)
#         log_probs_learned = learned_dist.log_prob(z).sum(dim=1)
#         log_probs_recon = -F.binary_cross_entropy(reconstructions, sample.repeat(num_samples, 1, 1, 1),
#                                                   reduction='none').view(num_samples, -1).sum(dim=1)
#
#         # Compute the importance weights for each sample
#         importance_weights = log_probs_recon + log_probs_prior - log_probs_learned
#
#         # Compute the log likelihood using the importance weights
#         log_likelihood = torch.logsumexp(importance_weights, dim=0) - torch.log(torch.tensor(float(num_samples)))
#
#         return log_likelihood.item()
#     def elbo_likelihood(self, sample):
#         """
#         Estimate the ELBO for a given sample using a trained VAE.
#
#         Args:
#             vae: The trained VAE model.
#             sample: A single CIFAR10 image (shape: [3, 32, 32]).
#             prior_mean: Mean of the prior distribution (default: 0).
#             prior_std: Standard deviation of the prior distribution (default: 1).
#
#         Returns:
#             elbo: The estimated ELBO for the given sample.
#         """
#
#         # Ensure the input sample has the correct shape
#         # Encode the sample to obtain the (mu, sigma) encoding
#         mu, log_sigma = self.encode(sample)
#         sigma = torch.exp(log_sigma)
#
#         # Sample from the learned latent distribution using the reparameterization trick
#         epsilon = torch.randn_like(sigma)
#         z = mu + epsilon * sigma
#
#         # Decode the latent variable to reconstruct the sample
#         reconstruction = self.decode(z)
#
#         # Compute the reconstruction loss (negative log likelihood of the sample)
#         recon_loss = F.binary_cross_entropy(reconstruction, sample, reduction='sum') / sample.size(0)
#
#         # Compute the KL divergence between the learned distribution and the prior
#         prior_dist = Normal(0, 1)
#         learned_dist = Normal(mu, sigma)
#         kl_div = torch.distributions.kl_divergence(learned_dist, prior_dist).sum() / sample.size(0)
#
#         # Compute the ELBO by subtracting the KL divergence from the reconstruction loss
#         elbo = recon_loss - kl_div
#
#         return elbo.item()
#     def entropy(self, prob_distribution):
#         # Calculate the entropy for a given probability distribution
#         prob_distribution = torch.tensor(prob_distribution)
#         non_zero_indices = prob_distribution > 0
#         entropy = -torch.sum(prob_distribution[non_zero_indices] * torch.log(prob_distribution[non_zero_indices]))
#         return entropy
#
#     def get_encoding(self, x):
#         return self.encode(x)[0]
#     def generate(self, x: Tensor, **kwargs) -> Tensor:
#         """
#         Given an input image x, returns the reconstructed image
#         :param x: (Tensor) [B x C x H x W]
#         :return: (Tensor) [B x C x H x W]
#         """
#
#         return self.forward(x)[0]
#     # def sample(self,
#     #            num_samples:int,
#     #            current_device: int, **kwargs) -> Tensor:
#     #     """
#     #     Samples from the latent space and return the corresponding
#     #     image space map.
#     #     :param num_samples: (Int) Number of samples
#     #     :param current_device: (Int) Device to run the model
#     #     :return: (Tensor)
#     #     """
#     #     z = Variable(
#     #                 torch.randn(num_samples, self.z_size).cuda() if self._is_on_cuda() else
#     #                 torch.randn(num_samples, self.z_size)
#     #             )
#     #     z_projected = self.project(z).view(
#     #         -1, self.kernel_num,
#     #         self.feature_size,
#     #         self.feature_size,
#     #     )
#     #
#     #     samples = self.decode(z)
#     #     return samples

class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 patch_size=512,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.patch_size = patch_size
        modules = []
        if hidden_dims is None:
            hidden_dims = [4, 8, 16, 32, 64, 128, 256, 512]
        self.criterion = nn.CrossEntropyLoss()

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        # self.encoder = get_encoder(
        #     "resnet34",
        #     in_channels=in_channels,
        #     weights="imagenet",
        #     output_stride=8
        # )
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def estimate_log_likelihood(self, sample, num_samples=100, prior_mean=0, prior_std=1):
        """
        Estimate the log likelihood for a given sample using a trained VAE and importance sampling.

        Args:
            vae: The trained VAE model.
            sample: A single CIFAR10 image (shape: [3, 32, 32]).
            num_samples: Number of samples to use for the Monte Carlo estimate (default: 100).
            prior_mean: Mean of the prior distribution (default: 0).
            prior_std: Standard deviation of the prior distribution (default: 1).

        Returns:
            log_likelihood: The estimated log likelihood for the given sample.
        """



        # Encode the sample to obtain the (mu, sigma) encoding
        mu, log_sigma = self.encode(sample)
        sigma = torch.exp(log_sigma)
        sigma = sigma + 1e-5
        # Compute the prior and learned distributions
        prior_dist = Normal(prior_mean, prior_std)
        learned_dist = Normal(mu, sigma)

        # Sample multiple times from the learned latent distribution using the reparameterization trick
        epsilon = torch.randn(num_samples, sigma.shape[-1]).to("cuda")
        z = mu + epsilon * sigma

        # Decode the latent variables to reconstruct the samples
        reconstructions = self.decode(z)

        # Compute the log probabilities of the reconstructions under the learned and prior distributions
        log_probs_prior = prior_dist.log_prob(z).sum(dim=1)
        log_probs_learned = learned_dist.log_prob(z).sum(dim=1)

        reconstructions = torch.clip(reconstructions, 0,1) #debug
        sample = torch.clip(sample, 0, 1)


        log_probs_recon = -F.binary_cross_entropy(reconstructions, sample.repeat(num_samples, 1, 1, 1),
                                                  reduction='none').view(num_samples, -1).sum(dim=1)

        # Compute the importance weights for each sample
        importance_weights = log_probs_recon + log_probs_prior - log_probs_learned

        # Compute the log likelihood using the importance weightss<
        log_likelihood = torch.logsumexp(importance_weights, dim=0) - torch.log(torch.tensor(float(num_samples)))
        return log_likelihood.item()


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def get_encoding(self, x):
        return self.encode(x)[0]
    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class ResNetVAE(BaseVAE):
    def __init__(self, patch_size=32, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(ResNetVAE, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim
        self.latent_dim = CNN_embed_dim
        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (3, 3), (4, 4), (5, 5)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding
        self.patch_size = patch_size
        # encoding components
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()  # y = (y1, y2, y3) \in [0 ,1]^3
        )

    def estimate_log_likelihood(self, sample, num_samples=100, prior_mean=0, prior_std=1):
        """
        Estimate the log likelihood for a given sample using a trained VAE and importance sampling.

        Args:
            vae: The trained VAE model.
            sample: A single CIFAR10 image (shape: [3, 32, 32]).
            num_samples: Number of samples to use for the Monte Carlo estimate (default: 100).
            prior_mean: Mean of the prior distribution (default: 0).
            prior_std: Standard deviation of the prior distribution (default: 1).

        Returns:
            log_likelihood: The estimated log likelihood for the given sample.
        """



        # Encode the sample to obtain the (mu, sigma) encoding
        mu, log_sigma = self.encode(sample)
        sigma = torch.exp(log_sigma)
        sigma = sigma + 1e-5
        # Compute the prior and learned distributions
        prior_dist = Normal(prior_mean, prior_std)
        learned_dist = Normal(mu, sigma)

        # Sample multiple times from the learned latent distribution using the reparameterization trick
        epsilon = torch.randn(num_samples, sigma.shape[-1]).to("cuda")
        z = mu + epsilon * sigma

        # Decode the latent variables to reconstruct the samples
        reconstructions = self.decode(z)

        # Compute the log probabilities of the reconstructions under the learned and prior distributions
        log_probs_prior = prior_dist.log_prob(z).sum(dim=1)
        log_probs_learned = learned_dist.log_prob(z).sum(dim=1)

        reconstructions = torch.clip(reconstructions, 0,1) #debug
        sample = torch.clip(sample, 0, 1)


        log_probs_recon = -F.binary_cross_entropy(reconstructions, sample.repeat(num_samples, 1, 1, 1),
                                                  reduction='none').view(num_samples, -1).sum(dim=1)

        # Compute the importance weights for each sample
        importance_weights = log_probs_recon + log_probs_prior - log_probs_learned

        # Compute the log likelihood using the importance weightss<
        log_likelihood = torch.logsumexp(importance_weights, dim=0) - torch.log(torch.tensor(float(num_samples)))
        return log_likelihood.item()

    def elbo_likelihood(self, sample):
        """
        Estimate the ELBO for a given sample using a trained VAE.

        Args:
            vae: The trained VAE model.
            sample: A single CIFAR10 image (shape: [3, 32, 32]).
            prior_mean: Mean of the prior distribution (default: 0).
            prior_std: Standard deviation of the prior distribution (default: 1).

        Returns:
            elbo: The estimated ELBO for the given sample.
        """

        # Ensure the input sample has the correct shape
        # Encode the sample to obtain the (mu, sigma) encoding
        mu, log_sigma = self.encode(sample)
        sigma = torch.exp(log_sigma)

        # Sample from the learned latent distribution using the reparameterization trick
        epsilon = torch.randn_like(sigma)
        z = mu + epsilon * sigma

        # Decode the latent variable to reconstruct the sample
        reconstruction = self.decode(z)

        # Compute the reconstruction loss (negative log likelihood of the sample)
        recon_loss = F.binary_cross_entropy(reconstruction, sample, reduction='sum') / sample.size(0)

        # Compute the KL divergence between the learned distribution and the prior
        prior_dist = Normal(0, 1)
        learned_dist = Normal(mu, sigma)
        kl_div = torch.distributions.kl_divergence(learned_dist, prior_dist).sum() / sample.size(0)

        # Compute the ELBO by subtracting the KL divergence from the reconstruction loss
        elbo = recon_loss - kl_div

        return elbo.item()



    def entropy(self, prob_distribution):
        # Calculate the entropy for a given probability distribution
        prob_distribution = torch.tensor(prob_distribution)
        non_zero_indices = prob_distribution > 0
        entropy = -torch.sum(prob_distribution[non_zero_indices] * torch.log(prob_distribution[non_zero_indices]))
        return entropy

    def get_encoding(self, x):
        return self.encode(x)[0]

    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(self.patch_size, self.patch_size), mode='bilinear')
        return x

    # def forward(self, x):
    #     mu, logvar = self.encode(x)
    #     z = self.reparameterize(mu, logvar)
    #     x_reconst = self.decode(z)
    #
    #     return x_reconst, z, mu, logvar

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']
        recons_loss =F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class CIFARVAE(ResNetVAE):

    def __init__(self):
        super().__init__(patch_size=32, fc_hidden1=1024, fc_hidden2=768, drop_p=0.0, CNN_embed_dim=256)
        resnet = cifarresnet.get_cifar("resnet18", [3]*3, model_urls=[])
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.resnet = nn.Sequential(*modules)
        self.latent_dim = 256