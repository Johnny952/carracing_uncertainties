import sys
sys.path.append('..')

from models import Net, Sensitivity, DropoutModel, BootstrapModel, BayesianModel, MixtureApprox
from utilities import smooth_l1_loss, Mixture

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

IMPLEMENTED_MODELS = (Sensitivity, DropoutModel, list, BayesianModel, MixtureApprox) # Bootstrap is a list

class Model:
    def __init__(self, nb_nets, lr, img_stack, gamma, batch_size, buffer_capacity, model='base', device='cpu'):
        self.model_name = model
        self.nb_nets = nb_nets
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity

        self.save_model = self.save_single_model
        self.load_model = self.load_single_model
        self.train = self.train_single_model
        self._criterion = F.smooth_l1_loss
        self._use_sigma = False

        if model == "base":
            self._model = Net(img_stack).double().to(self.device)

        elif model == "sensitivity":
            self._model = Sensitivity(img_stack).double().to(self.device)
            self._forward = self.sensitivity_uncert

            self.input_range = [0, 255]
            self.factor = 1/255
            self.delta = self.factor * (self.input_range[1] - self.input_range[0])
            self._criterion = smooth_l1_loss
            self._use_sigma = True

        elif model == "dropout":
            self._model = DropoutModel(img_stack).double().to(self.device)
            self._model.use_dropout(val=True)                          # Dropout turned off during training
            self._forward = self.dropout_uncert
            self._criterion = smooth_l1_loss
            self._use_sigma = True

        elif model == "bootstrap":
            self._model = [BootstrapModel(img_stack).double().to(self.device) for _ in range(nb_nets)]
            self._forward = self.boot_uncert
            self.save_model = self.save_boot_model
            self.load_model = self.load_boot_model
            self.train = self.train_boot_model
            self._criterion = smooth_l1_loss
            self._use_sigma = True

        elif model == "bnn":
            self._model = BayesianModel(img_stack).double().to(self.device)
            self._forward = self.bayes_uncert
            self.train = self.train_bayes_model

            self.sample_nbr = 10
            self.complexity_cost_weight = 1e-6
        
        elif model == "mixture":
            self._model = MixtureApprox(img_stack).double().to(self.device)
            self._forward = lambda state, use_uncert: self._model(state)
            self.train = self.train_mix_model
            self._criterion = smooth_l1_loss

            self._epist_criterion = nn.MSELoss()
            self._epist_loss_weight = 1e-10
            self._add_means = False
            self._dev = 1.0

        else:
            raise ValueError("Model not implemented")
        

        if model == "bootstrap":
            self._optimizer = [optim.Adam(net.parameters(), lr=lr) for net in self._model]
        else:
            self._optimizer = optim.Adam(self._model.parameters(), lr=lr)



    def forward_nograd(self, state, use_uncert=True):
        if isinstance(self._model, IMPLEMENTED_MODELS):
            with torch.no_grad():
                (alpha, beta), v, (epistemic, aleatoric) = self._forward(state, use_uncert=use_uncert)
        else:
            # Base model
            epistemic = torch.Tensor([0])
            aleatoric = torch.Tensor([0])
            with torch.no_grad():
                (alpha, beta), v = self._model(state)[:2]
        return (alpha, beta), v, (epistemic, aleatoric)




    def train_single_model(self, epochs, clip_param, database):
        (s, a, r, s_, old_a_logp) = database

        target_v = r + self.gamma * self.forward_nograd(s_, use_uncert=False)[1]
        adv = target_v - self.forward_nograd(s, use_uncert=False)[1]

        for _ in range(epochs):
            sampler = SubsetRandomSampler(range(self.buffer_capacity))
            self.train_once(self._model, self._optimizer, target_v, adv, old_a_logp, s, a, clip_param, sampler)
            

    def train_boot_model(self, epochs, clip_param, database):
        (s, a, r, s_, old_a_logp) = database

        target_v = r + self.gamma * self.forward_nograd(s_)[1]
        adv = target_v - self.forward_nograd(s)[1]

        indices = [torch.utils.data.RandomSampler(range(self.buffer_capacity), num_samples=self.buffer_capacity, replacement=True) for _ in range(self.nb_nets)]

        for _ in range(epochs):
            for net, optimizer, index in zip(self._model, self._optimizer, indices):
                self.train_once(net, optimizer, target_v, adv, old_a_logp, s, a, clip_param, index)



    def train_once(self, net, optimizer, target_v, adv, old_a_logp, s, a, clip_param, rand_sampler):
        sampler = BatchSampler(rand_sampler, self.batch_size, False)

        for index in sampler:

            if self._use_sigma:
                (alpha, beta), _, sigma = net(s[index])
            else:
                alpha, beta = net(s[index])[0]
            dist = Beta(alpha, beta)
            a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
            ratio = torch.exp(a_logp - old_a_logp[index])

            surr1 = ratio * adv[index]
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv[index]
            action_loss = -torch.min(surr1, surr2).mean()
            if self._use_sigma:
                value_loss = self._criterion(net(s[index])[1], target_v[index], sigma, 1.0, reduction="mean")
            else:
                value_loss = self._criterion(net(s[index])[1], target_v[index])
            loss = action_loss + 2. * value_loss

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
            optimizer.step()
    
    def train_mix_model(self, epochs, clip_param, database):
        (s, a, r, s_, old_a_logp) = database

        target_v = r + self.gamma * self.forward_nograd(s_)[1]
        adv = target_v - self.forward_nograd(s)[1]

        for _ in range(epochs):
            rand_sampler = SubsetRandomSampler(range(self.buffer_capacity))
            sampler = BatchSampler(rand_sampler, self.batch_size, False)

            for index in sampler:

                (alpha, beta), _, (_, sigma) = self._model(s[index])
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = self._criterion(self._model(s[index])[1], target_v[index], sigma, 1.0, reduction="mean")

                # Mixture approximation loss
                mix = Mixture(s[index], dev=self._dev, device=self.device)
                samples = mix.sample(self.nb_nets).double()
                if self._add_means:
                    samples = torch.cat((samples, s[index]), dim=0)
                logp = mix.logp(samples)
                net_logp = self._model(samples)[-1][0]
                mix_loss = self._epist_criterion(logp.float(), net_logp.float().squeeze())
                
                loss = action_loss + 2. * value_loss + self._epist_loss_weight * mix_loss
                self._optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self._model.parameters(), self.max_grad_norm)
                self._optimizer.step()

                
    
    def train_bayes_model(self, epochs, clip_param, database):
        (s, a, r, s_, old_a_logp) = database

        target_v = r + self.gamma * self.forward_nograd(s_)[1]
        adv = target_v - self.forward_nograd(s)[1]

        for _ in range(epochs):
            rand_sampler = SubsetRandomSampler(range(self.buffer_capacity))
            sampler = BatchSampler(rand_sampler, self.batch_size, False)

            for index in sampler:
                loss = 0
                for _ in range(self.sample_nbr):
                    (alpha, beta) = self._model(s[index])[0]
                    dist = Beta(alpha, beta)
                    a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                    ratio = torch.exp(a_logp - old_a_logp[index])

                    surr1 = ratio * adv[index]
                    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv[index]
                    action_loss = -torch.min(surr1, surr2).mean()
                    value_loss = self._criterion(self._model(s[index])[1], target_v[index])
                    kl_loss = self._model.nn_kl_divergence() * self.complexity_cost_weight
                    
                    loss += action_loss + 2. * value_loss + kl_loss
                self._optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self._model.parameters(), self.max_grad_norm)
                self._optimizer.step()



    def save_single_model(self, epoch, path='param/ppo_net_params.pkl'):
        tosave = {
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            }
        torch.save(tosave, path)

    def load_single_model(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if eval_mode:
            self._model.eval()
        else:
            self._model.train()
        return checkpoint['epoch']

    
    def save_boot_model(self, epoch, path='param/ppo_net_params.pkl'):
        tosave = {'epoch': epoch}
        for idx, (net, optimizer) in enumerate(zip(self._model, self._optimizer)):
            tosave['model_state_dict{}'.format(idx)] = net.state_dict()
            tosave['optimizer_state_dict{}'.format(idx)] = optimizer.state_dict()
        torch.save(tosave, path)
    
    def load_boot_model(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        for idx in range(len(self._model)):
            self._model[idx].load_state_dict(checkpoint['model_state_dict{}'.format(idx)])
            self._optimizer[idx].load_state_dict(checkpoint['optimizer_state_dict{}'.format(idx)])
            if eval_mode:
                self._model[idx].eval()
            else:
                self._model[idx].train()
        return checkpoint['epoch']










    def boot_uncert(self, state, use_uncert=False):
        alpha_list = []
        beta_list = []
        sigma_list = []
        v_list = []
        for net in self._model:
            (alpha, beta), v, sigma = net(state)
            sigma_list.append(sigma)
            alpha_list.append(alpha)
            beta_list.append(beta)
            v_list.append(v)
        sigma_list = torch.stack(sigma_list)
        alpha_list = torch.stack(alpha_list)
        beta_list = torch.stack(beta_list)
        v_list = torch.stack(v_list)

        #var = alpha*beta / ((alpha+beta+1)*(alpha+beta)**2)
        #epistemic = torch.mean(torch.var(alpha_list, dim=0)) + torch.mean(torch.var(beta_list, dim=0))
        epistemic = torch.mean(torch.var(alpha_list / (alpha_list + beta_list), dim=0))
        aleatoric = torch.mean(sigma_list, dim=0)
        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), torch.mean(v_list, dim=0), (epistemic, aleatoric)


    def dropout_uncert(self, state, use_uncert=False):
        if use_uncert:
            self._model.use_dropout(val=False)                 # Activate dropout layers

            # Estimate uncertainties
            alpha_list = []
            beta_list = []
            sigma_list = []
            v_list = []
            for _ in range(self.nb_nets):
                with torch.no_grad():
                    (alpha, beta), v, sigma = self._model(state)
                sigma_list.append(sigma)
                alpha_list.append(alpha)
                beta_list.append(beta)
                v_list.append(v)
            sigma_list = torch.stack(sigma_list)
            alpha_list = torch.stack(alpha_list)
            beta_list = torch.stack(beta_list)
            v_list = torch.stack(v_list)
            self._model.use_dropout(val=True)              # Deactivate dropout layers

            #var = alpha*beta / ((alpha+beta+1)*(alpha+beta)**2)
            epistemic = torch.mean(torch.var(alpha_list / (alpha_list + beta_list), dim=0))
        else:
            epistemic = torch.tensor([0])
        
        # Predict
        # alpha, beta, v = torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0), torch.mean(v_list, dim=0)
        # aleatoric = torch.mean(sigma_list, dim=0)
        (alpha, beta), v, sigma = self._model(state)
        aleatoric = sigma
                
        return (alpha, beta), v, (epistemic, aleatoric)

    def sensitivity_uncert(self, state, use_uncert=False):
        if use_uncert:
            # Random matrix -1/0/1
            rand_dir = self.delta*(torch.empty(self.nb_nets, state.shape[1], state.shape[2], state.shape[3]).random_(3).double().to(self.device) - 1)
            rand_dir += state
            rand_dir[rand_dir > self.input_range[1]] = self.input_range[1]
            rand_dir[rand_dir < self.input_range[0]] = self.input_range[0]

            # Estimate uncertainties
            with torch.no_grad():
                (alpha, beta), v, sigma = self._model(rand_dir)

            #var = alpha*beta / ((alpha+beta+1)*(alpha+beta)**2)
            epistemic = torch.mean(torch.var(alpha / (alpha + beta), dim=0))
            #aleatoric = torch.mean(sigma, dim=0)
        else:
            epistemic = torch.tensor([0])

        # Predict
        (alpha, beta), v, sigma = self._model(state)
        aleatoric = sigma

        #return (torch.mean(alpha, dim=0).view(1, -1), torch.mean(beta, dim=0).view(1, -1)), torch.mean(v, dim=0), (epistemic, aleatoric)
        return (alpha, beta), v, (epistemic, aleatoric)

    def bayes_uncert(self, state, use_uncert=False):
        alpha_list = []
        beta_list = []
        v_list = []
        for _ in range(self.nb_nets):
            (alpha, beta), v = self._model(state)
            alpha_list.append(alpha)
            beta_list.append(beta)
            v_list.append(v)
        alpha_list = torch.stack(alpha_list)
        beta_list = torch.stack(beta_list)
        v_list = torch.stack(v_list)

        epistemic = torch.mean(torch.var(alpha_list / (alpha_list + beta_list), dim=0))
        aleatoric = torch.Tensor([0])
        
        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), torch.mean(v_list, dim=0), (epistemic, aleatoric)