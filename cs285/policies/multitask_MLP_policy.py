import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure import utils


class MultitaskMLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 num_tasks,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.num_tasks = num_tasks

        if self.discrete:
            self.logits_na_shared, self.logits_na_tasks = ptu.build_multitask_network(
                input_size=self.ob_dim, 
                output_size = self.ac_dim,
                n_layers=self.n_layers,
                size=self.size, 
                num_tasks=self.num_tasks,
            )

            self.logits_na_shared.to(ptu.device)
            self.mean_net = None
            self.logstd = None

            self.optimizer_tasks = [None for _ in range(self.num_tasks)]
            for T in range(self.num_tasks):
                self.logits_na_tasks[T].to(ptu.device)
                self.optimizer_tasks[T] = optim.Adam(
                                                itertools.chain(self.logits_na_shared.parameters(), self.logits_na_tasks[T].parameters()),
                                                self.learning_rate
                                            )
        else:
            self.logits_na = None
            self.mean_net_shared, self.mean_net_tasks = ptu.build_multitask_network(
                input_size=self.ob_dim, 
                output_size = self.ac_dim,
                n_layers=self.n_layers,
                size=self.size, 
                num_tasks=self.num_tasks,
            )
            
            # shared logstd and optimizer
            self.mean_net_shared.to(ptu.device)

            # task specific logstd and optimizer
            self.logstd_tasks = []
            self.optimizer_tasks = []
            for T in range(self.num_tasks):
                self.logstd_tasks.append(nn.Parameter(
                    torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
                ))
                
                self.mean_net_tasks[T].to(ptu.device)
                self.optimizer_tasks.append(optim.Adam(
                    itertools.chain([self.logstd_tasks[T]], self.mean_net_shared.parameters(), self.mean_net_tasks[T].parameters()),
                    self.learning_rate
                ))

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, T, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        action_distribution = self.forward_task(T, observation)
        action = action_distribution.sample()
        return ptu.to_numpy(action)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network for a single task
    def forward_task(self, T, observation: torch.FloatTensor):
        if self.discrete:
            shared_output = self.logits_na_shared(observation)
            logits = self.logits_na_tasks[T](shared_output)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            shared_output = self.mean_net_shared(observation)
            batch_mean = self.mean_net_tasks[T](shared_output)
            scale_tril = torch.diag(torch.exp(self.logstd_tasks[T]))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation_tasks: torch.FloatTensor):
        if self.discrete:
            action_distribution_tasks = []
            for T in range(self.num_tasks):
                shared_output = self.logits_na_shared(observation_tasks[T])
                logits = self.logits_na_tasks[T](shared_output)
                action_distribution = distributions.Categorical(logits=logits)
                action_distribution_tasks.append(action_distribution)
            return action_distribution_tasks
        else:
            action_distribution_tasks = []
            for T in range(self.num_tasks):
                shared_output = self.mean_net_shared(observation_tasks[T])
                batch_mean = self.mean_net_tasks[T](shared_output)
                
                scale_tril = torch.diag(torch.exp(self.logstd_tasks[T]))
                batch_dim = batch_mean.shape[0]
                batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
                action_distribution = distributions.MultivariateNormal(
                    batch_mean,
                    scale_tril=batch_scale_tril,
                )

                action_distribution_tasks.append(action_distribution)
            
        return action_distribution_tasks


#####################################################
#####################################################


class MultitaskMLPPolicyAC(MultitaskMLPPolicy):
    def update(self, observations_tasks, actions_tasks, adv_n_tasks=None):
        # TODO: update the policy and return the loss
        observations_tasks = [ptu.from_numpy(observations_tasks[T]) for T in range(self.num_tasks)]
        actions_tasks = [ptu.from_numpy(actions_tasks[T]) for T in range(self.num_tasks)]
        adv_n_tasks = [ptu.from_numpy(adv_n_tasks[T])for T in range(self.num_tasks)]

        loss_tasks = [0 for _ in range(self.num_tasks)]
        for T in range(self.num_tasks):
            action_distribution = self.forward_task(T, observation=observations_tasks[T])
            loss = - action_distribution.log_prob(actions_tasks[T]) * adv_n_tasks[T]
            loss = loss.mean()

            self.optimizer_tasks[T].zero_grad()
            loss.backward()
            self.optimizer_tasks[T].step()

            loss_tasks[T] = loss.item()

        return loss_tasks
