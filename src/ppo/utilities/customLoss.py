import torch
import numpy as np

def gaussian_loss(input: torch.Tensor, target: torch.Tensor, log_sigma: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    n = torch.abs(input - target)
    loss = log_sigma + 0.5 * n ** 2 / torch.exp(log_sigma) ** 2
    # loss = 0.5 * torch.log(log_sigma_**2) + 0.5 * \
    #     n ** 2 / log_sigma_**2
    return loss.mean()


def ll_gaussian(y, mu, log_var):  # log-likelihood of gaussian
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2)) * (y-mu)**2


def elbo(y_pred, y, mu, log_var, weight_decay=1e-4):
    # likelihood of observing y given Variational mu and sigma
    likelihood = ll_gaussian(y, mu, log_var)

    # prior probability of y_pred
    log_prior = ll_gaussian(y_pred, 0, torch.log(torch.tensor(1./weight_decay)))

    # variational probability of y_pred
    log_p_q = ll_gaussian(y_pred, mu, log_var)

    # by taking the mean we approximate the expectation
    return (likelihood + log_prior - log_p_q).mean()


def det_loss(y_pred, y, mu, log_var, weight_decay=1e-4):
    return -elbo(y_pred, y, mu, log_var, weight_decay=weight_decay)


def flow_loss(log_prob_z0, log_prob_zk, log_det, x_hat, X_batch, loss_fn, scale=1):
    return torch.mean(log_prob_z0) + scale * loss_fn(x_hat, X_batch) - torch.mean(log_prob_zk) - torch.mean(log_det)


def flow_loss_split(log_prob_z0, log_prob_zk, log_det, x_hat, X_batch, loss_fn):
    return torch.mean(log_prob_z0) - torch.mean(log_prob_zk) - torch.mean(log_det), loss_fn(x_hat, X_batch)
