from .base_critic import BaseCritic
from torch import nn
from torch import optim
import itertools

from cs285.infrastructure import pytorch_util as ptu


class ContinualBootstrappedContinuousCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """

    def __init__(self, hparams):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # number of tasks
        self.num_tasks = hparams['num_tasks']
        # number of new tasks
        self.num_new_tasks = hparams['num_new_tasks']
        # number of total tasks
        self.total_tasks = self.num_tasks

        # critic parameters
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']

        # critic network 
        self.critic_shared_mlp, self.critic_task_head = ptu.build_multitask_network(
            self.ob_dim, 
            1, 
            n_layers=self.n_layers, 
            size=self.size,
            num_tasks=self.num_tasks
        )
        
        self.loss = nn.MSELoss()

        # shared critic network 
        self.critic_shared_mlp.to(ptu.device)

        # task specific network
        self.task_specific_optimizer = [None for _ in range(self.num_tasks)]
        for T in range(self.num_tasks):
            self.critic_task_head[T].to(ptu.device)
            self.task_specific_optimizer[T] = optim.Adam(
                                                itertools.chain(self.critic_shared_mlp.parameters(), self.critic_task_head[T].parameters()),
                                                self.learning_rate,
                                            )

    def forward(self, obs):
        task_specific_network_output = []
        for T in range(self.total_tasks):
            shared_network_output = self.critic_shared_mlp(obs[T])
            task_specific_network_output.append(self.critic_task_head[T](shared_network_output).squeeze(1))
        return task_specific_network_output

    def forward_np(self, obs):
        obs = [ptu.from_numpy(obs[T]) for T in range(self.total_tasks)]
        task_specific_predictions = self(obs)
        return [ptu.to_numpy(task_specific_predictions[T]) for T in range(self.num_tasks)]

    def forward_task(self, T, obs):
        shared_network_output = self.critic_shared_mlp(obs)
        return self.critic_task_head[T](shared_network_output).squeeze(1)

    def forward_task_np(self, T, obs):
        obs = ptu.from_numpy(obs)
        predictions = self.forward_task(T, obs)
        return ptu.to_numpy(predictions)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        # TODO: Implement the pseudocode below: do the following (
        # self.num_grad_steps_per_target_update * self.num_target_updates)
        # times:
        # every self.num_grad_steps_per_target_update steps (which includes the
        # first step), recompute the target values by
        #     a) calculating V(s') by querying the critic with next_ob_no
        #     b) and computing the target values as r(s, a) + gamma * V(s')
        # every time, update this critic using the observations and targets
        #
        # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
        #       to 0) when a terminal state is reached
        # HINT: make sure to squeeze the output of the critic_network to ensure
        #       that its dimensions match the reward
        ob_no = [ptu.from_numpy(ob_no[T]) for T in range(self.total_tasks)]
        ac_na = [ptu.from_numpy(ac_na[T]) for T in range(self.total_tasks)]
        next_ob_no = [ptu.from_numpy(next_ob_no[T]) for T in range(self.total_tasks)]
        reward_n = [ptu.from_numpy(reward_n[T]) for T in range(self.total_tasks)]
        terminal_n = [ptu.from_numpy(terminal_n[T]) for T in range(self.total_tasks)]

        # compute loss for each task
        loss_tasks = [0 for _ in range(self.total_tasks)]
        # compute target value for each task   
        V_target_tasks = [0 for _ in range(self.total_tasks)]

        for t in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            if t % self.num_grad_steps_per_target_update == 0:
                for T in range(self.total_tasks):
                    V_sp1 = self.forward_task(T, next_ob_no[T])
                    V_target_tasks[T] = reward_n[T] + self.gamma * V_sp1 * (1 - terminal_n[T])

            for T in range(self.total_tasks):
                V_pred = self.forward_task(T, ob_no[T])
                loss = self.loss(V_pred, V_target_tasks[T].detach())

                self.task_specific_optimizer[T].zero_grad()
                loss.backward()
                self.task_specific_optimizer[T].step()

                loss_tasks[T] = loss.item()

        return loss_tasks

    def add_new_task(self):
        self.total_tasks = self.num_tasks + self.num_new_tasks

        _, critic_task_head_list = ptu.build_multitask_network(
            self.ob_dim, 
            1, 
            n_layers=self.n_layers, 
            size=self.size,
            num_tasks=self.num_new_tasks
        )
        for task_head in critic_task_head_list:
            self.critic_task_head.append(task_head)
        
        # task specific network
        for T in range(self.num_tasks, self.total_tasks):
            self.critic_task_head[T].to(ptu.device)
            self.task_specific_optimizer.append(optim.Adam(
                                                itertools.chain(self.critic_shared_mlp.parameters(), self.critic_task_head[T].parameters()),
                                                self.learning_rate,
                                            ))