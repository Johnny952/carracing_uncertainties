from models.model import Net
from models.sensitivity import Sensitivity
from models.dropout import DropoutModel
from models.bootstrap import BootstrapModel
from models.bnn import BayesianModel

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import numpy as np

from customLoss import smooth_l1_loss

from blitz.losses import kl_divergence_from_nn

import nvidia_smi


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self, args, model="base"):
        self.training_step = 0
        self.q = args.uncert_q
        if model == "base":
            self.net = Net(args).double().to(device)
        elif model == "sensitivity":
            self.net = Sensitivity(args).double().to(device)
        elif model == "dropout":
            self.net = DropoutModel(args).double().to(device)
            self.net.use_dropout(val=True)                          # Dropout turned off during training
        elif model == "bootstrap":
            self.net = [BootstrapModel(args).double().to(device) for _ in range(self.q)]
        elif model == "bnn":
            self.net = BayesianModel(args).double().to(device)
        else:
            raise ValueError("Model not implemented")
        self.transition = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                                    ('r', np.float64), ('s_', np.float64, (args.img_stack, 96, 96))])
        self.buffer = np.empty(self.buffer_capacity, dtype=self.transition)
        self.counter = 0

        if model == "bootstrap":
            self.optimizer = [optim.Adam(net.parameters(), lr=1e-3) for net in self.net]
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.gamma = args.gamma

        self.input_range = [0, 255]
        self.factor = 1/255
        self.delta = self.factor * (self.input_range[1] - self.input_range[0])

        self.checkpoint = args.from_checkpoint

        self.sample_nbr = 15
        self.complexity_cost_weight = 1


    def select_action(self, state, eval=False):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)

        if isinstance(self.net, DropoutModel) and eval:
            (alpha, beta), _, (epistemic, aleatoric) = self.dropout_uncer(state)
        elif isinstance(self.net, Sensitivity) and eval:
            (alpha, beta), _, (epistemic, aleatoric) = self.sensitivity_uncert(state)
        elif isinstance(self.net, list) and eval:
            (alpha, beta), _, (epistemic, aleatoric) = self.boot_uncert(state)
        elif isinstance(self.net, BayesianModel) and eval:
            (alpha, beta), _, (epistemic, aleatoric) = self.bayes_uncert(state)
        else:
            epistemic = torch.Tensor([0])
            aleatoric = torch.Tensor([0])
            with torch.no_grad():
                if isinstance(self.net, list):
                    (alpha, beta) = self.boot_select_action(state)[0]
                elif isinstance(self.net, BayesianModel):
                    (alpha, beta) = self.bayes_uncert(state)[0]
                else:
                    if isinstance(self.net, DropoutModel):
                        self.net.use_dropout(val=True)
                    (alpha, beta) = self.net(state)[0]
        
        if eval:
            action = alpha / (alpha + beta)
            a_logp = 0

            action = action.squeeze().cpu().numpy()
        else:
            dist = Beta(alpha, beta)
            action = dist.sample()
            a_logp = dist.log_prob(action).sum(dim=1)

            action = action.squeeze().cpu().numpy()
            a_logp = a_logp.item()
        return action, a_logp, (epistemic, aleatoric)


    def save_param(self, epoch):
        if isinstance(self.net, list):
            tosave = {'epoch': epoch}
            for idx, (net, optimizer) in enumerate(zip(self.net, self.optimizer)):
                tosave['model_state_dict{}'.format(idx)] = net.state_dict()
                tosave['optimizer_state_dict{}'.format(idx)] = optimizer.state_dict()
        else:
            tosave = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }
        torch.save(tosave, 'param/ppo_net_params.pkl')
        #torch.save(self.net.state_dict(), 'param/ppo_net_params.pkl')

    def load_param(self, eval_mode=False):
        checkpoint = torch.load(self.checkpoint)
        if isinstance(self.net, list):
            checkpoint = torch.load(self.checkpoint)
            for idx, (net, optimizer) in enumerate(zip(self.net, self.optimizer)):
                self.net[idx].load_state_dict(checkpoint['model_state_dict{}'.format(idx)])
                self.optimizer[idx].load_state_dict(checkpoint['optimizer_state_dict{}'.format(idx)])
                if eval_mode:
                    self.net[idx].eval()
                else:
                    self.net[idx].train()
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if eval_mode:
                self.net.eval()
            else:
                self.net.train()
        return checkpoint['epoch']
    
    def eval_mode(self):
        self.net.eval()
    def train_mode(self):
        self.net.train()

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            if isinstance(self.net, list):
                target_v = r + self.gamma * self.boot_select_action(s_)[1]
                adv = target_v - self.boot_select_action(s)[1]
            elif isinstance(self.net, BayesianModel):
                target_v = r + self.gamma * self.bayes_uncert(s_)[1]
                adv = target_v - self.bayes_uncert(s)[1]
            else:
                target_v = r + self.gamma * self.net(s_)[1]
                adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        if isinstance(self.net, list):
            indices = [torch.utils.data.RandomSampler(range(self.buffer_capacity), num_samples=self.buffer_capacity, replacement=True) for _ in range(self.q)]

        for _ in range(self.ppo_epoch):
            if isinstance(self.net, list):
                for net, optimizer, index in zip(self.net, self.optimizer, indices):
                    self.train_once(net, optimizer, target_v, adv, old_a_logp, s, a, index)
            else:
                self.train_once(self.net, self.optimizer, target_v, adv, old_a_logp, s, a)
    

    def train_once(self, net, optimizer, target_v, adv, old_a_logp, s, a, index=None):
        if index is None:
            rand_sampler = SubsetRandomSampler(range(self.buffer_capacity))
        else:
            rand_sampler = index
        sampler = BatchSampler(rand_sampler, self.batch_size, False)

        for index in sampler:
                if isinstance(net, (DropoutModel, Sensitivity, list)):
                    (alpha, beta), _, sigma = net(s[index])
                    dist = Beta(alpha, beta)
                    a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                    ratio = torch.exp(a_logp - old_a_logp[index])

                    surr1 = ratio * adv[index]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                    action_loss = -torch.min(surr1, surr2).mean()
                    #value_loss = F.smooth_l1_loss(net(s[index])[1], target_v[index])
                    value_loss = smooth_l1_loss(net(s[index])[1], target_v[index], sigma, 1.0, reduction="mean")
                    
                    loss = action_loss + 2. * value_loss
                elif isinstance(net, BayesianModel):
                    loss = 0
                    for _ in range(self.sample_nbr):
                        #res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                        #print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
                        (alpha, beta) = net(s[index])[0]
                        dist = Beta(alpha, beta)
                        a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                        ratio = torch.exp(a_logp - old_a_logp[index])

                        surr1 = ratio * adv[index]
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                        action_loss = -torch.min(surr1, surr2).mean()
                        value_loss = F.smooth_l1_loss(net(s[index])[1], target_v[index])
                        
                        loss += action_loss + 2. * value_loss
                        loss += net.nn_kl_divergence()
                else:
                    alpha, beta = net(s[index])[0]
                    dist = Beta(alpha, beta)
                    a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                    ratio = torch.exp(a_logp - old_a_logp[index])

                    surr1 = ratio * adv[index]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                    action_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.smooth_l1_loss(net(s[index])[1], target_v[index])
                    
                    loss = action_loss + 2. * value_loss

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
                optimizer.step()

    def boot_select_action(self, state):
        alpha_list = []
        beta_list = []
        v_list = []
        for net in self.net:
            (alpha, beta), v, _ = net(state)
            alpha_list.append(alpha)
            beta_list.append(beta)
            v_list.append(v)
        alpha_list = torch.stack(alpha_list)
        beta_list = torch.stack(beta_list)
        v_list = torch.stack(v_list)
        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), torch.mean(v_list, dim=0)

    def boot_uncert(self, state):
        with torch.no_grad():
            alpha_list = []
            beta_list = []
            sigma_list = []
            v_list = []
            for net in self.net:
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


    def dropout_uncer(self, state, predict=False):
        self.net.use_dropout(val=False)                 # Activate dropout layers
        with torch.no_grad():
            alpha_list = []
            beta_list = []
            sigma_list = []
            v_list = []
            for _ in range(self.q):
                (alpha, beta), v, sigma = self.net(state)
                sigma_list.append(sigma)
                alpha_list.append(alpha)
                beta_list.append(beta)
                v_list.append(v)
            sigma_list = torch.stack(sigma_list)
            alpha_list = torch.stack(alpha_list)
            beta_list = torch.stack(beta_list)
            v_list = torch.stack(v_list)
            self.net.use_dropout(val=True)              # Deactivate dropout layers

            #var = alpha*beta / ((alpha+beta+1)*(alpha+beta)**2)
            epistemic = torch.mean(torch.var(alpha_list / (alpha_list + beta_list), dim=0))
            
            if predict:
                alpha, beta, v = torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0), torch.mean(v_list, dim=0)
                aleatoric = torch.mean(sigma_list, dim=0)
            else:
                (alpha, beta), v, sigma = self.net(state)
                aleatoric = sigma
                
        return (alpha, beta), v, (epistemic, aleatoric)

    def sensitivity_uncert(self, state):
        with torch.no_grad():

            # Random matrix -1/0/1
            rand_dir = self.delta*(torch.empty(self.q, state.shape[1], state.shape[2], state.shape[3]).random_(3).double().to(device) - 1)
            rand_dir += state
            rand_dir[rand_dir > self.input_range[1]] = self.input_range[1]
            rand_dir[rand_dir < self.input_range[0]] = self.input_range[0]

            (alpha, beta), v, sigma = self.net(rand_dir)

            #var = alpha*beta / ((alpha+beta+1)*(alpha+beta)**2)
            epistemic = torch.mean(torch.var(alpha / (alpha + beta), dim=0))
            #aleatoric = torch.mean(sigma, dim=0)

            (alpha, beta), v, sigma = self.net(state)
            aleatoric = sigma

        #return (torch.mean(alpha, dim=0).view(1, -1), torch.mean(beta, dim=0).view(1, -1)), torch.mean(v, dim=0), (epistemic, aleatoric)
        return (alpha, beta), v, (epistemic, aleatoric)

    def bayes_uncert(self, state):
        with torch.no_grad():
            alpha_list = []
            beta_list = []
            v_list = []
            for _ in range(self.q):
                (alpha, beta), v = self.net(state)
                alpha_list.append(alpha)
                beta_list.append(beta)
                v_list.append(v)
            alpha_list = torch.stack(alpha_list)
            beta_list = torch.stack(beta_list)
            v_list = torch.stack(v_list)

            epistemic = torch.mean(torch.var(alpha_list / (alpha_list + beta_list), dim=0))
            aleatoric = torch.Tensor([0])
        
        return (torch.mean(alpha_list, dim=0), torch.mean(beta_list, dim=0)), torch.mean(v_list, dim=0), (epistemic, aleatoric)