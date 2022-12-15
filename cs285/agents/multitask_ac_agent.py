from collections import OrderedDict

from cs285.critics.multitask_bootstrapped_continuous_critic_task_only import \
    MultitaskBootstrappedContinuousCriticTaskOnly
from cs285.critics.multitask_bootstrapped_continuous_critic import \
    MultitaskBootstrappedContinuousCritic
from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.pytorch_util import from_numpy
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.multitask_MLP_policy import MultitaskMLPPolicyAC
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent


class MultitaskACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MultitaskACAgent, self).__init__()

        # self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.num_tasks = self.agent_params['num_tasks']

        self.beta = self.agent_params['beta']

        # task specific policy
        if agent_params['share_policy'] == False:
            self.actor = MultitaskMLPPolicyAC(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                self.agent_params['num_tasks'],
                self.agent_params['discrete'],
                self.agent_params['learning_rate'],
            )
        # sharing policy 
        if agent_params['share_policy']:
            self.actor = MLPPolicyAC(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                self.agent_params['discrete'],
                self.agent_params['learning_rate'],
            )

        if self.agent_params['separate_shared_and_specific_critic'] == False:
            # shared + task specific critic
            self.critic_tasks = MultitaskBootstrappedContinuousCritic(self.agent_params)
        else:
            # task specific critic
            self.critic_tasks = MultitaskBootstrappedContinuousCriticTaskOnly(self.agent_params)

        # sharing critic
        self.critic_shared = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffers = [ReplayBuffer() for _ in range(self.agent_params['num_tasks'])]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # advantage = estimate_advantage(...)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        loss_tasks = [OrderedDict() for _ in range(self.agent_params['num_tasks'])]
        for T in range(self.agent_params['num_tasks']):
            loss_tasks[T]['Critic_Loss'] = 0
            loss_tasks[T]['Actor_Loss'] = 0

        tasks_critic_losses = []
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            shared_critic_losses = self.critic_shared.multitask_update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            tasks_critic_losses = self.critic_tasks.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        advantage_shared = self.estimate_advantage_shared(ob_no, next_ob_no, re_n, terminal_n)
        advantage_tasks = self.estimate_advantage_tasks(ob_no, next_ob_no, re_n, terminal_n)

        advantage_average = np.multiply(self.beta, advantage_shared) + np.multiply(1 - self.beta, advantage_tasks)

        actor_losses = []
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            if isinstance(self.actor, MLPPolicyAC):
                actor_losses = self.actor.multitask_update(ob_no, ac_na, advantage_average)
            if isinstance(self.actor, MultitaskMLPPolicyAC):
                actor_losses = self.actor.update(ob_no, ac_na, advantage_average)

        for T in range(self.agent_params['num_tasks']):
            loss_tasks[T]['Critic_Loss'] = tasks_critic_losses[T]
            loss_tasks[T]['Actor_Loss'] = actor_losses[T]
            loss_tasks[T]['Shared Critic Loss'] = shared_critic_losses[T]

        return loss_tasks

    def estimate_advantage_tasks(self, ob_no, next_ob_no, re_n, terminal_n):
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        adv_n_tasks = [0 for _ in range(self.num_tasks)]
 
        for T in range(self.num_tasks):
            V_s = self.critic_tasks.forward_task_np(T, ob_no[T])
            V_sp1 = self.critic_tasks.forward_task_np(T, next_ob_no[T])

            Q = re_n[T] + self.gamma * V_sp1 * (1 - terminal_n[T])
            adv_n_tasks[T] = Q - V_s

        if self.standardize_advantages:
            for T in range(self.num_tasks):
                adv_n_tasks[T] = (adv_n_tasks[T] - np.mean(adv_n_tasks[T])) / (np.std(adv_n_tasks[T]) + 1e-8)
        return adv_n_tasks

    def estimate_advantage_shared(self, ob_no, next_ob_no, re_n, terminal_n):
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        adv_n_tasks = [0 for _ in range(self.num_tasks)]
 
        for T in range(self.num_tasks):
            V_s = self.critic_shared.forward_np(ob_no[T])
            V_sp1 = self.critic_shared.forward_np(next_ob_no[T])

            Q = re_n[T] + self.gamma * V_sp1 * (1 - terminal_n[T])
            adv_n_tasks[T] = Q - V_s

        if self.standardize_advantages:
            for T in range(self.num_tasks):
                adv_n_tasks[T] = (adv_n_tasks[T] - np.mean(adv_n_tasks[T])) / (np.std(adv_n_tasks[T]) + 1e-8)
        return adv_n_tasks

    def add_to_replay_buffer(self, T, paths):
        self.replay_buffers[T].add_rollouts(paths)

    def sample(self, T, batch_size):
        return self.replay_buffers[T].sample_recent_data(batch_size)
