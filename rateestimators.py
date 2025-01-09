import numpy as np
from scipy.stats import beta
from hmmlearn import hmm
import pymc as pm

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

class HierarchicalBayesianEstimator(RateEstimator):
    #todo: reduce runtime overhead
    def __init__(self, prior_alpha=1, prior_beta=1, hyper_alpha=2, hyper_beta=2):
        super().__init__()
        """
        Initialize the Hierarchical Bayesian Estimator.
        Args:
        - prior_alpha, prior_beta: Parameters for the Beta distribution (shift probability).
        - hyper_alpha, hyper_beta: Hyperprior parameters for Gamma distributions (for alpha and beta).
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.hyper_alpha = hyper_alpha
        self.hyper_beta = hyper_beta
        self.trace = None
        self.posterior_mean = None
        self.posterior_variance = None

    def update(self, data, samples=250, chains=4):
        """
        Fit the model using PyMC.
        Args:
        - data: Binary observations (1 = shift, 0 = no shift).
        - samples: Number of posterior samples to draw.
        - chains: Number of chains for MCMC.
        """
        with pm.Model() as model:
            # Hyperpriors for alpha and beta
            alpha_hyper = pm.Gamma("alpha_hyper", alpha=self.hyper_alpha, beta=self.hyper_beta)
            beta_hyper = pm.Gamma("beta_hyper", alpha=self.hyper_alpha, beta=self.hyper_beta)

            # Beta prior for theta (shift probability)
            theta = pm.Beta("theta", alpha=alpha_hyper, beta=beta_hyper)

            # Observed data
            shifts = pm.Bernoulli("shifts", p=theta, observed=data)

            # Sampling from the posterior
            self.trace = pm.sample(samples, chains=chains, return_inferencedata=True)

        # Update posterior mean and variance
        self.posterior_mean = self.trace.posterior["theta"].mean().item()
        self.posterior_variance = self.trace.posterior["theta"].var().item()
        self.rate = self.posterior_mean

    def get_posterior_mean(self):
        """Return the posterior mean of shift probability."""
        if self.posterior_mean is None:
            raise ValueError("Model has not been fitted yet.")
        return self.posterior_mean

    def get_posterior_variance(self):
        """Return the posterior variance of shift probability."""
        if self.posterior_variance is None:
            raise ValueError("Model has not been fitted yet.")
        return self.posterior_variance


class HMMEstimator(RateEstimator):
    def __init__(self):
        super().__init__()
        self.model = hmm.MultinomialHMM(n_components=2, n_iter=100, random_state=42)

    def update(self, trace):
        self.model.fit(trace)
        return self.model


def get_hmm_distributed_events(ptrans_shifted, ptrans_normal, p_emission, num_iters):
    """ returns a list of hmm-distributed events"""
    pass


if __name__ == '__main__':
    # Bayesian filter with a 10% true shift rate
    # bf = BernoulliEstimator(prior_rate=0.1, tpr=0.96, tnr=0.97)
    bf = HierarchicalBayesianEstimator(prior_alpha=1, prior_beta=1, hyper_alpha=1, hyper_beta=1)
    test = np.random.binomial(1, 0.1, 10000)  # Simulate test data
    #todo test with traceleength 1000 but with this kind of data. 
    # test = np.zeros(10000)
    # test[5000:] = 1
    traces = [test[i:i+100] for i in range(0, len(test)-100)]
    print(traces)
    for i,trace in enumerate(traces):
        bf.update(trace)
        rate = bf.get_rate()
        print(rate, "+-", 3*np.sqrt(bf.get_posterior_variance()))

