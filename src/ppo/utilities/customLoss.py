import torch

def ll_gaussian(y, mu, log_var):  # log-likelihood of gaussian
    #sigma = torch.exp(0.5 * log_var)
    #return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2)) * (y-mu)**2
    var = torch.exp(log_var)
    return -0.5 * log_var - (1 / (2 * var)) * (y-mu)**2


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



def det_loss2(y_pred, y, mu, log_var, weight_decay=1e-4):
    var = torch.exp(log_var)
    criterion = torch.nn.GaussianNLLLoss(reduction='none')

    neg_log_likelihood = criterion(y, mu, var)
    neg_log_prior = criterion(y_pred, 0, torch.tensor(1./weight_decay))
    neg_log_p = criterion(y_pred, mu, var)

    return (neg_log_likelihood + neg_log_prior - neg_log_p).mean()