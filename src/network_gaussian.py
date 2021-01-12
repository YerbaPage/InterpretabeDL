import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Config_File import Config
from network import BigModel
import math


class Gaussian(nn.Module):
    def __init__(self, rho):
        super().__init__()
        self.mu = 0.0
        self.rho = nn.Parameter(rho,requires_grad=Config.bias_requires_grad)
        #self.normal = torch.distributions.Normal(0, 1)
        self.normal = torch.distributions.Normal(torch.tensor(0.0).to(device=torch.device("cuda")),torch.tensor(1.0).to(device=torch.device("cuda")))

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self, word_emb):
        epsilon = self.normal.sample(word_emb.size())
        return self.sigma * epsilon

    def log_prob(self, word_emb, bias):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((bias - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class Gaussian_word_based(nn.Module):
    def __init__(self, rho):
        super().__init__()
        self.mu = 0.0
        self.transform = nn.Linear(Config.word_dim, Config.word_dim)
        nn.init.xavier_normal_(self.transform.weight)
        self.activate = nn.Tanh()
        self.normal = torch.distributions.Normal(0, 1)

    def sigma(self, word_emb):
        perturb = self.transform(word_emb)
        # perturb=self.activate(perturb)
        return torch.log1p(torch.exp(perturb))

    def sample(self, word_emb):
        epsilon = self.normal.sample(word_emb.size()).to(Config.device)
        return self.sigma(word_emb) * epsilon

    def log_prob(self, word_emb, bias):
        ret = (-math.log(math.sqrt(2 * math.pi))
               - torch.log(self.sigma(word_emb))
               - ((bias - self.mu) ** 2) / (2 * self.sigma(word_emb) ** 2)).sum()
        # if torch.min(ret)<-1e8:
        #    print('cwy debug')
        return ret


class Uniform(object):
    def __init__(self, rho):
        super().__init__()
        self.mu = 0.0
        self.rho = rho
        self.normal = torch.distributions.uniform(-1, 1)

    @property
    def sigma(self):
        return self.rho

    def sample(self, size):
        epsilon = self.normal.sample(size).to(Config.device)
        return self.sigma * epsilon


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input)) + 1e-10
        prob2 = torch.exp(self.gaussian2.log_prob(input)) + 1e-10
        ret = (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()
        # if torch.isnan(ret).sum()>0:
        #    print('cwy debug')
        return ret

class NormLoss(object):
    def __init__(self):
        pass

    def log_prob(self, input):
        return torch.norm(input, p=2, dim=(1, 2)).sum()
    def log_prob_param(self,input):
        return torch.norm(input,p=2).sum()


class U_quadratic(object):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.alpha = 12.0 / (b - a) / (b - a) / (b - a)
        self.beta = (a + b) / 2

    def log_prob(self, input):
        #prob = torch.min(self.alpha * (input - self.beta) * (input - self.beta),
        #                 torch.ones_like(input) * self.alpha * (self.b - self.beta) * (self.b - self.beta)) + 1e-8
        prob=self.alpha * (input - self.beta) * (input - self.beta)
        return -(torch.log(prob)).sum()

def U_quadratic_delta_norm(input):
    input=torch.max(input,torch.cuda.FloatTensor([Config.U_quadratic_a]))
    input=torch.min(input,torch.cuda.FloatTensor([Config.U_quadratic_b]))
    return input

class BayesianPerturb(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Bias parameters
        self.bias_rho = nn.Parameter(torch.Tensor(in_features, out_features).uniform_(Config.bias_gaussian_low,Config.bias_gaussian_high)).to(Config.device)
        self.bias = globals()[Config.bias_method](self.bias_rho)  # Gaussian(self.bias_rho)
        # Prior distributions
        if Config.prior_method == 'U_quadratic':
            self.bias_prior = U_quadratic(Config.U_quadratic_a, Config.U_quadratic_b)
        elif Config.prior_method == 'ScaleMixtureGaussian':
            self.bias_prior = ScaleMixtureGaussian(Config.PI, Config.SIGMA_1.to(Config.device),
                                                   Config.SIGMA_2.to(Config.device))
        elif Config.prior_method=='NormLoss' or Config.prior_method=='NormLoss_parameter':
            self.bias_prior=NormLoss()
        self.log_prior = 0
        self.log_variational_posterior = 0

        self.bias_test = nn.Parameter(torch.zeros(1, 1))

    def forward(self, word_emb, sample=False, calculate_log_probs=False):
        if self.training or sample:
            bias = self.bias.sample(word_emb)
        else:
            bias = self.bias_test
        if self.training or calculate_log_probs:
            if Config.prior_method=='NormLoss_parameter':
                if Config.bias_method=='Gaussian_word_based':
                    self.log_prior = self.bias_prior.log_prob_param(bias)
                else:
                    self.log_prior = self.bias_prior.log_prob_param(self.bias.rho)
            else:
                self.log_prior = self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.bias.log_prob(word_emb, bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return bias


class BigModel_Bayesian(BigModel):
    def __init__(self, w2id):
        super(BigModel_Bayesian, self).__init__(w2id)
        self.perturb = BayesianPerturb(1, Config.word_dim)
        self.loss = None
        self.losses_init = nn.Parameter(torch.zeros(Config.gaussian_sample))
        self.prior_loss = None
        self.variational_loss = None

    def sample_elbo(self, input, target, num_sample, prior_alpha):
        temp = self.forward_emb(input)
        losses = []
        ret_list = []
        negative_log_likelihood_list = []
        log_prior_list = []
        log_variational_list = []
        for i in range(num_sample):
            delta = self.perturb(temp[0], sample=True, calculate_log_probs=True)
            if Config.prior_method=='U_quadratic':
                delta=U_quadratic_delta_norm(delta)
            this_ret = F.log_softmax(self.forward_long(
                [temp[0] + delta , temp[1], temp[2], #/ torch.norm(delta, dim=(1, 2), keepdim=True)
                 temp[3], temp[4]], input), dim=-1)
            log_prior_list.append(prior_alpha * self.perturb.log_prior * Config.prior_factor)
            log_variational_list.append(prior_alpha * self.perturb.log_variational_posterior * Config.variational_factor)
            losses.append(log_prior_list[-1] + log_variational_list[-1])
            ret_list.append(this_ret)
            negative_log_likelihood_list.append(F.nll_loss(this_ret, target, size_average=False))
        ret = torch.mean(torch.stack(ret_list), dim=0, keepdim=False)
        negative_log_likelihood = torch.mean(torch.stack(negative_log_likelihood_list))
        self.loss = torch.mean(torch.stack(losses)) + negative_log_likelihood
        self.prior_loss = torch.mean(torch.stack(log_prior_list))
        self.variational_loss = torch.mean(torch.stack(log_variational_list))
        return ret

    def forward(self, input, num_sample=Config.gaussian_sample):
        temp = self.forward_emb(input)
        losses = []
        ret_list = []
        negative_log_likelihood_list = []
        for i in range(num_sample):
            delta = self.perturb(temp[0], sample=True, calculate_log_probs=True)
            this_ret = F.log_softmax(self.forward_long(
                [temp[0] + delta, temp[1], temp[2], temp[3], temp[4]],
                input), dim=-1) # / torch.norm(delta, dim=(1, 2), keepdim=True)
            ret_list.append(this_ret)
        ret = torch.mean(torch.stack(ret_list), dim=0, keepdim=False)
        return ret
