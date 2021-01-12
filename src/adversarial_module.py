import torch
from torch.autograd import Variable
from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import rand_init_delta
from advertorch.utils import is_float_or_torch_tensor
import torch.nn as nn
import numpy as np
from advertorch.utils import clamp
from advertorch.utils import replicate_input
from advertorch.utils import batch_multiply, batch_clamp, normalize_by_pnorm, clamp_by_pnorm, batch_l1_proj
import copy
from numpy import linalg as LA
from Config_File import Config

class PGD_from_zhengguangyu(object):
    def __init__(self, model, epsilon=None, alpha=None):
        super(PGD_from_zhengguangyu, self).__init__()
        self.model = model
        if epsilon is None:
            self.epsilon=Config.epsilon
        else:
            self.epsilon = epsilon
        if alpha is None:
            self.alpha=Config.alpha
        else:
            self.alpha = alpha
        self.grad_backup = {}

    def attack_on_emb(self, adv_emb, ori_emb, grad, is_first_attack,wordlevel=False):
        #if not is_first_attack:
        #    assert torch.norm(adv_emb - ori_emb) != 0

        assert not torch.isnan(adv_emb).any()
        #norm = torch.norm(grad, dim=(1, 2), keepdim=True)
        norm = torch.sum(grad.view(grad.size(0),-1).pow(2),dim=1,keepdim=True).pow(0.5).unsqueeze(-1)

        assert not torch.isnan(norm).any()
        r_at = self.alpha * grad / norm

        r_at_nan_replace = torch.isnan(r_at)
        zeros = torch.zeros_like(r_at)
        r_at = torch.where(r_at_nan_replace, zeros, r_at)

        assert not torch.isnan(r_at).any()

        adv_emb = adv_emb + r_at
        if wordlevel:
            adv_emb = self.project_emb_wordlevel(adv_emb, ori_emb)
        else:
            adv_emb = self.project_emb(adv_emb, ori_emb)
        assert not torch.isnan(adv_emb).any()

        return adv_emb

    def project_emb(self, new_data, ori_data):
        r = new_data - ori_data
        #r_norm = torch.norm(r, dim=(1, 2), keepdim=True)
        r_norm = torch.sum(r.view(r.size(0), -1).pow(2), dim=1, keepdim=True).pow(0.5).unsqueeze(-1)
        r_norm_expand = r_norm.expand(-1, new_data.size(1), new_data.size(2))
        proj_r = torch.where(r_norm_expand > self.epsilon, self.epsilon * r / r_norm_expand, r)
        return ori_data + Variable(proj_r.contiguous())

    def project_emb_wordlevel(self, new_data, ori_data):
        r = new_data - ori_data
        #r_norm = torch.norm(r, dim=(1, 2), keepdim=True)
        r_norm = torch.sum(r.view(-1,r.size(2)).pow(2), dim=1, keepdim=True).pow(0.5) #bsz*seq_len
        r_norm_expand = r_norm.expand(-1, new_data.size(2)).view(r.size(0),r.size(1),r.size(2)) #bsz, seq_len, feat_dim
        proj_r = torch.where(r_norm_expand > self.epsilon, self.epsilon * r / r_norm_expand, r)
        return ori_data + Variable(proj_r.contiguous())

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    self.grad_backup[name] = param.grad.clone()
                else:
                    self.grad_backup[name] = None

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        #self.emb_K_backup_grad=None
        #self.emb_K_backup_grad2=None

    def attack(self, epsilon=Config.epsilon, alpha=Config.alpha, emb_name='embed_model.embed_layer.weight', is_first_attack=False, mask=None,
               mode='pgd'):
        # set emb_name to the embedding's name in your model
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    if mode=='pgd':
                        r_at = alpha * param.grad / norm
                    else:
                        r_at = torch.rand_like(param.data)-0.5
                    if mask is not None:
                        r_at*=mask
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def attack_on_emb_efficient(self, emb, emb2, input, input2, epsilon=1, alpha=0.3, emb_name='embed_model.embed_layer.weight',
                      is_first_attack=False, mask=None, mask2=None,  mode='pgd', apply_equal_mask=False):
        # set emb_name to the embedding's name in your model

        emb_t=emb.clone()
        emb_t2 = emb2.clone()
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                    self.emb_K_backup_grad=Variable(emb.grad,requires_grad=False)
                    self.emb_K_backup_grad2=Variable(emb2.grad,requires_grad=False)
                norm = torch.norm(self.emb_K_backup_grad)
                norm2 = torch.norm(self.emb_K_backup_grad2)
                if norm != 0 and not torch.isnan(norm):
                    if mode=='pgd':
                        r_at = alpha * self.emb_K_backup_grad / norm
                        r_at2 = alpha * self.emb_K_backup_grad2 / norm2
                    else:
                        r_at_vocab = torch.rand_like(param.data)-0.5
                        r_at_vocab=r_at_vocab.unsqueeze(0).repeat(input.size(0),1,1)

                        r_at=torch.gather(r_at_vocab,1,input.unsqueeze(-1).repeat(1,1,r_at_vocab.size(-1)))
                        r_at2=torch.gather(r_at_vocab, 1, input2.unsqueeze(-1).repeat(1,1,r_at_vocab.size(-1)))

                    if apply_equal_mask:
                        r_at_all=torch.cat((r_at,r_at2),-2) #batch, len*2, dim
                        input_all=torch.cat((input,input2),-1) #batch, len*2
                        mean_equal_mask=input_all.unsqueeze(-1).eq(input_all.unsqueeze(-2)).float()
                        mean_equal_mask=mean_equal_mask/torch.sum(mean_equal_mask,-1,keepdim=True)
                        r_at_mean=torch.bmm(mean_equal_mask,r_at_all)
                        r_at = r_at_mean[:,:emb.size(1),:]
                        r_at2 = r_at_mean[:, emb.size(1):, :]

                    if mask is not None:
                        r_at*=mask.unsqueeze(-1).float()
                        r_at2*=(mask2.unsqueeze(-1).float())

                    emb_t+=r_at
                    emb_t = self.project_emb(emb_t, emb, epsilon)
                    emb_t2+=r_at2
                    emb_t2 = self.project_emb(emb_t2, emb2, epsilon)

        return Variable(emb_t,requires_grad=True), Variable(emb_t2,requires_grad=True)

    def attack_on_emb(self, emb, emb2, input, input2, epsilon=1, alpha=0.3, emb_name='embed_model.embed_layer.weight',
                      is_first_attack=False, mask=None, mask2=None, mode='pgd', apply_equal_mask=False):
        # set emb_name to the embedding's name in your model

        emb_t = emb.clone()
        emb_t2 = emb2.clone()
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(emb.grad)
                norm2 = torch.norm(emb2.grad)
                if norm != 0 and not torch.isnan(norm):
                    if mode == 'pgd':
                        r_at = alpha * emb.grad / norm
                        r_at2 = alpha * emb2.grad / norm2
                    else:
                        r_at_vocab = torch.rand_like(param.data) - 0.5
                        r_at_vocab = r_at_vocab.unsqueeze(0).repeat(input.size(0), 1, 1)

                        r_at = torch.gather(r_at_vocab, 1, input.unsqueeze(-1).repeat(1, 1, r_at_vocab.size(-1)))
                        r_at2 = torch.gather(r_at_vocab, 1, input2.unsqueeze(-1).repeat(1, 1, r_at_vocab.size(-1)))

                    if apply_equal_mask:
                        r_at_all = torch.cat((r_at, r_at2), -2)  # batch, len*2, dim
                        input_all = torch.cat((input, input2), -1)  # batch, len*2
                        mean_equal_mask = input_all.unsqueeze(-1).eq(input_all.unsqueeze(-2)).float()
                        mean_equal_mask = mean_equal_mask / torch.sum(mean_equal_mask, -1, keepdim=True)
                        r_at_mean = torch.bmm(mean_equal_mask, r_at_all)
                        r_at = r_at_mean[:, :emb.size(1), :]
                        r_at2 = r_at_mean[:, emb.size(1):, :]

                    if mask is not None:
                        r_at *= mask.unsqueeze(-1).float()
                        r_at2 *= (mask2.unsqueeze(-1).float())

                    emb_t += r_at
                    emb_t = self.project_emb(emb_t, emb, epsilon)
                    emb_t2 += r_at2
                    emb_t2 = self.project_emb(emb_t2, emb2, epsilon)

        return Variable(emb_t, requires_grad=True), Variable(emb_t2, requires_grad=True)

    def project_emb(self, new_data, ori_data, epsilon):
        r = new_data - ori_data
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return ori_data + r


    def restore(self, emb_name='embed_model.embed_layer.weight'):
        # set emb_name to the embedding's name in your model
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if  param.grad is not None:
                    self.grad_backup[name] = param.grad.clone()
                else:
                    self.grad_backup[name] = None

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

def perturb_iterative(xvar, batch_data, yvar, mask, predict, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0,
                      l1_sparsity=None):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.

    :param xvar: input data.
    :param yvar: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param l1_sparsity: sparsity value for L1 projection.
                  - if None, then perform regular L1 projection.
                  - if float value, then perform sparse L1 descent from
                    Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
    :return: tensor containing the perturbed input.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    delta.requires_grad_()
    delta2.requires_grad_()
    for ii in range(nb_iter):
        outputs = predict(xvar + delta,batch_data)
        loss = loss_fn(outputs, yvar)
        if minimize:
            loss = -loss

        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data

        elif ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)

        elif ord == 1:
            grad = delta.grad.data
            abs_grad = torch.abs(grad)

            batch_size = grad.size(0)
            view = abs_grad.view(batch_size, -1)
            view_size = view.size(1)
            if l1_sparsity is None:
                vals, idx = view.topk(1)
            else:
                vals, idx = view.topk(
                    int(np.round((1 - l1_sparsity) * view_size)))

            out = torch.zeros_like(view).scatter_(1, idx, vals)
            out = out.view_as(grad)
            grad = grad.sign() * (out > 0).float()
            grad = normalize_by_pnorm(grad, p=1)
            delta.data = delta.data + batch_multiply(eps_iter, grad)

            delta.data = batch_l1_proj(delta.data.cpu(), eps)
            if xvar.is_cuda:
                delta.data = delta.data.cuda()
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        delta.grad.data.zero_()

    x_adv = clamp(xvar + delta, clip_min, clip_max)
    return x_adv

class PGDAttack_custom(Attack, LabelMixin):
    """
    The projected gradient descent attack (Madry et al, 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, l1_sparsity=None, targeted=False):
        """
        Create an instance of the PGDAttack.

        """
        super(PGDAttack_custom, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.l1_sparsity = l1_sparsity
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def _verify_and_process_inputs(self,x, y):
        if self.targeted:
            assert y is not None

        if not self.targeted:
            if y is None:
                y = self._get_predicted_label(x)

        x[0] = replicate_input(x[0])
        x[1] = replicate_input(x[1])
        y = replicate_input(y)
        return x, y

    def perturb(self, x, batch_data, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval = perturb_iterative(
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity,
        )

        return rval.data