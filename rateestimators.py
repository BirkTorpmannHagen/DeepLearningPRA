import numpy as np
from scipy.stats import beta

class RateEstimator:
    def __init__(self):
        self.rate = 0.5

    def get_rate(self):
        return self.rate


class BernoulliEstimator(RateEstimator):
    def __init__(self, prior_rate=0.5, tpr=0.9, tnr=0.1):
        super().__init__()
        self.rate = prior_rate  # current rate estimate
        self.alpha = 0
        self.beta = 0
        self.tpr = tpr  # Sensitivity of the DSD
        self.tnr = tnr  # Specificity of the DSD
        prior_dist = beta(self.alpha, self.beta)
        self.rate = prior_dist.mean()

    def update_tpr_tnr(self, dsd_tpr, dsd_tnr):
        self.tpr = dsd_tpr
        self.tnr = dsd_tnr

    def update(self, trace_list):
        """Update the posterior using DSD predictions."""
        # Compute weighted evidence for shift (alpha) and no-shift (beta)

        # print(self.tpr, self.tnr)
        trace = np.array(trace_list)
        positive_likelihood = trace * self.tpr + (1 - trace) * (1 - self.tnr)
        negative_likelihood = (1 - trace) * self.tnr + trace * (1 - self.tpr)

        # Effective counts based on likelihoods
        self.alpha = positive_likelihood.sum()
        self.beta = negative_likelihood.sum()

        # Update the rate estimate
        self.rate = self.get_posterior_mean()
        return self.rate

    def sample(self, n, rate_groundtruth):
        event = np.random.binomial(1, rate_groundtruth, n)
        return event

    def get_posterior_mean(self):
        """Return the mean of the posterior distribution."""
        return self.alpha / (self.alpha + self.beta)

    def get_posterior_variance(self):
        """Return the variance of the posterior distribution."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total**2 * (total + 1))
