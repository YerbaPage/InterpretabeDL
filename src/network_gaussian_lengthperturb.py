import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Config_File import Config
from network import BigModel
import math

class Gaussian(object):
    def __init__(self, rho):
        super().__init__()
        self.mu = 0.0
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self,length):
        epsilon = self.normal.sample(self.rho.repeat(length,1).size()).to(Config.device)
        return self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()

class BayesianPerturb(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Bias parameters
        self.bias_rho = nn.Parameter(torch.Tensor(in_features,out_features).uniform_(-5,-4)).to(Config.device)
        self.bias = Gaussian(self.bias_rho)
        # Prior distributions
        self.bias_prior = ScaleMixtureGaussian(Config.PI, Config.SIGMA_1.to(Config.device), Config.SIGMA_2.to(Config.device))
        self.log_prior = 0
        self.log_variational_posterior = 0

        self.bias_test=nn.Parameter(torch.zeros(1,1))

    def forward(self, length, sample=False, calculate_log_probs=False):
        if self.training or sample:
            bias = self.bias.sample(length)
        else:
            bias = self.bias_test
        if self.training or calculate_log_probs:
            self.log_prior = self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return bias

class BigModel_Bayesian(BigModel):
    def __init__(self, w2id):
        super(BigModel_Bayesian, self).__init__(w2id)
        self.perturb=BayesianPerturb(1,Config.word_dim)
        self.loss=None
        self.losses_init=nn.Parameter(torch.zeros(Config.gaussian_sample))

    def sample_elbo(self,input,target,num_sample,num_batches):
        temp = self.forward_emb(input)
        losses=[]
        ret_list=[]
        negative_log_likelihood_list=[]
        for i in range(num_sample):
            delta = self.perturb(temp[0].size(1), sample=True, calculate_log_probs=True)
            this_ret = F.log_softmax(self.forward_long([temp[0] + delta.unsqueeze(0), temp[1], temp[2], temp[3], temp[4]], input))
            losses.append( self.perturb.log_variational_posterior - self.perturb.log_prior)
            ret_list.append(this_ret)
            negative_log_likelihood_list.append(F.nll_loss(this_ret, target, size_average=False))
        ret=torch.mean(torch.stack(ret_list),dim=0,keepdim=False)
        negative_log_likelihood = torch.mean(torch.stack(negative_log_likelihood_list))
        self.loss=torch.mean(torch.stack(losses))/num_batches+negative_log_likelihood
        return ret

    def forward(self,input,num_sample=Config.gaussian_sample):
        temp = self.forward_emb(input)
        losses=[]
        ret_list=[]
        negative_log_likelihood_list=[]
        for i in range(num_sample):
            delta = self.perturb(temp[0].size(1), sample=True, calculate_log_probs=True)
            this_ret = F.log_softmax(self.forward_long([temp[0] + delta.unsqueeze(0), temp[1], temp[2], temp[3], temp[4]], input))
            ret_list.append(this_ret)
        ret=torch.mean(torch.stack(ret_list),dim=0,keepdim=False)
        return ret