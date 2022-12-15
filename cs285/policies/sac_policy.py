from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        entropy = torch.exp(self.log_alpha)
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        
        observation = ptu.from_numpy(observation)
        action_distribution = self.forward(observation)
        if sample:
            action = action_distribution.rsample()
        else:
            action = action_distribution.mean
        action = ptu.to_numpy(action)
        return action

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = torch.distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            self.logstd = nn.Parameter(self.logstd.clip(min=self.log_std_bounds[0], max=self.log_std_bounds[1]))
            std = torch.exp(self.logstd)
            batch_dim = batch_mean.size(dim=0)
            batch_std = std.repeat(batch_dim, 1)
            action_distribution = sac_utils.SquashedNormal(loc=batch_mean, scale=batch_std)
            return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        # get actions predicted by the policy given the observation
        obs = ptu.from_numpy(obs)
        action_distribution = self.forward(obs)
        actions = action_distribution.rsample()
        log_prob = action_distribution.log_prob(actions).sum(dim=1)
 
        q1, q2 = critic.forward(obs, actions)
        q_min = torch.minimum(q1, q2).squeeze(1)

        actor_loss = (self.alpha * log_prob - q_min).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # update entropy regularizer, alpha
        action_distribution = self.forward(obs)
        actions = action_distribution.rsample()
        log_prob = action_distribution.log_prob(actions)

        alpha_loss = ( - self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, self.alpha