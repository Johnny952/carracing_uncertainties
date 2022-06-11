from abc import abstractclassmethod
import torch
import numpy as np
import wandb


class AbstactAgent:
    def __init__(
        self,
        nb_nets,
        img_stack,
        actions,
        learning_rate,
        gamma,
        buffer,
        epsilon,
        batch_size,
        device="cpu",
        clip_grad=False,
    ):

        self._device = device
        self._clip_grad = clip_grad
        self._batch_size = batch_size
        self._lr = learning_rate
        self._gamma = gamma
        self._img_stack = img_stack
        self._actions = actions
        self._epsilon = epsilon

        self._buffer = buffer
        
        self._nb_update = 0

    def empty_buffer(self):
        """Empty replay buffer"""
        self._buffer.empty()

    def number_experiences(self):
        """Get number of saved experiences

        Returns:
            int: Number of experiences
        """
        return len(self._buffer)

    def store_transition(self, state, action, next_state, reward, done):
        """Store a transition in replay buffer

        Args:
            state (np.ndarray): State in time t
            action (torch.Tensot): Action index taken in time t
            next_state (np.ndarray): State in time t+1
            reward (float): Reward of moving from state t to t+1
            done (bool): Whether state in time t+1 is terminal or not
        """
        self._buffer.push(
            torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(dim=0),
            action.unsqueeze(dim=0),
            torch.from_numpy(np.array(next_state, dtype=np.float32)).unsqueeze(dim=0),
            torch.Tensor([reward]),
            torch.Tensor([done]),
        )
        return self._buffer.able_sample()

    def epsilon_step(self):
        """Epsilon decay e = e*factor"""
        self._epsilon.step()

    def epsilon(self):
        return self._epsilon.epsilon()

    def select_action(self, observation, eval=False):
        """Selects a epislon greedy action

        Args:
            observation (np.ndarray): Observation of the environment
            eval (bool, optional): Whether to use only greedy actions or not. Defaults to False.

        Returns:
            int: The action taken
            int: The corresponding action index
        """
        aleatoric = torch.Tensor([0])
        epistemic = torch.Tensor([0])
        if eval or np.random.rand() > self._epsilon.epsilon():
            # Select action greedily
            with torch.no_grad():
                index, epistemic, aleatoric = self.get_values(
                    (torch.from_numpy(observation).unsqueeze(dim=0).float()).to(
                        self._device
                    )
                )
        else:
            # Select random action
            index = torch.randint(0, len(self._actions), size=(1,))
        return self._actions[index], index.cpu(), (epistemic, aleatoric)

    @abstractclassmethod
    def get_values(self, observation):
        raise NotImplementedError()

    def compute_loss(self, states, actions, next_states, rewards, dones):
        curr_Q1 = self._model1(states).gather(1, actions).squeeze(dim=-1)
        curr_Q2 = self._model2(states).gather(1, actions).squeeze(dim=-1)

        next_Q = torch.min(
            torch.max(self._model1(next_states), 1)[0],
            torch.max(self._model2(next_states), 1)[0],
        ).squeeze(dim=-1)
        expected_Q = rewards + (1 - dones) * self._gamma * next_Q

        loss1 = self._criterion(curr_Q1, expected_Q.detach())
        loss2 = self._criterion(curr_Q2, expected_Q.detach())

        return loss1, loss2

    def update(self):
        states, actions, next_states, rewards, dones = self.unpack(
            self._buffer.sample()
        )
        loss1, loss2 = self.compute_loss(states, actions, next_states, rewards, dones)

        self._optimizer1.zero_grad()
        loss1.backward()
        if self._clip_grad:
            for param in self._model1.parameters():
                param.grad.data.clamp_(-1, 1)
        self._optimizer1.step()

        self._optimizer2.zero_grad()
        loss2.backward()
        if self._clip_grad:
            for param in self._model2.parameters():
                param.grad.data.clamp_(-1, 1)
        self._optimizer2.step()

        self.log_loss(loss1.item(), loss2.item())
        self._nb_update += 1

        return states, actions, next_states, rewards, dones

    def log_loss(self, loss1, loss2):
        wandb.log(
            {
                "Update Step": self._nb_update,
                "Loss 1": float(loss1),
                "Loss 2": float(loss2),
                "Epsilon": float(self.epsilon()),
            }
        )

    def save_param(self, epoch, path="param/ppo_net_param.pkl"):
        tosave = {
            "epoch": epoch,
            "model1_state_disct": self._model1.state_dict(),
            "model2_state_disct": self._model2.state_dict(),
            "optimizer1_state_dict": self._optimizer1.state_dict(),
            "optimizer2_state_dict": self._optimizer2.state_dict(),
        }
        torch.save(tosave, path)

    def load_param(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model1.load_state_dict(checkpoint["model1_state_disct"])
        self._model2.load_state_dict(checkpoint["model2_state_disct"])
        self._optimizer1.load_state_dict(checkpoint["optimizer1_state_dict"])
        self._optimizer2.load_state_dict(checkpoint["optimizer2_state_dict"])

        if eval_mode:
            self._model1.eval()
            self._model2.eval()
        else:
            self._model1.train()
            self._model2.train()
        return checkpoint["epoch"]

    def unpack(self, batch):
        states = torch.cat(batch.state).float().to(self._device)
        actions = torch.cat(batch.action).long().to(self._device)
        next_states = torch.cat(batch.next_state).float().to(self._device)
        rewards = torch.cat(batch.reward).to(self._device)
        dones = torch.cat(batch.done).to(self._device)

        return states, actions, next_states, rewards, dones
