import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Optional

from tianshou.policy import BasePolicy
from tianshou.policy.dist import DiagGaussian
from tianshou.data import Batch, to_torch_as, ReplayBuffer


class SACDiscretePolicy(BasePolicy):
    """Implementation of Discrete Soft Actor-Critic. arXiv:1910.07207

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s -> Q(s,
        a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s -> Q(s,
        a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param float alpha: entropy regularization coefficient, default to 0.2.
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to ``False``.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self,
                 actor: torch.nn.Module,
                 actor_optim: torch.optim.Optimizer,
                 critic1: torch.nn.Module,
                 critic1_optim: torch.optim.Optimizer,
                 critic2: torch.nn.Module,
                 critic2_optim: torch.optim.Optimizer,
                 dist_fn: torch.distributions.Distribution
                 = torch.distributions.Categorical,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 alpha: float = 0.2,
                 reward_normalization: bool = False,
                 ignore_done: bool = False,
                 estimation_step: int = 1,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        assert 0 <= tau <= 1, 'tau should in [0, 1]'
        self._tau = tau
        assert 0 <= gamma <= 1, 'gamma should in [0, 1]'
        self._gamma = gamma
        self._rew_norm = reward_normalization
        assert estimation_step > 0, 'estimation_step should greater than 0'
        self._n_step = estimation_step
        self._rm_done = ignore_done
        self.dist_fn = dist_fn

        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self._alpha = alpha
        self.__eps = np.finfo(np.float32).eps.item()

    def train(self) -> None:
        self.training = True
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.training = False
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def sync_weight(self) -> None:
        for o, n in zip(
                self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                input: str = 'obs', **kwargs) -> Batch:
        obs = getattr(batch, input)
        logits, h = self.actor(obs, state=state, info=batch.info)
        dist = self.dist_fn(logits)
        act = dist.sample()
        log_prob = torch.log(logits)
        return Batch(
            logits=logits, act=act, state=h, dist=logits, log_prob=log_prob)

    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indice: np.ndarray) -> Batch:
        if self._rm_done:
            batch.done = batch.done * 0.
        batch = self.compute_nstep_return(
            batch, buffer, indice, self._target_q,
            self._gamma, self._n_step, self._rew_norm)
        return batch

    def _target_q(self, buffer: ReplayBuffer,
                  indice: np.ndarray) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs: s_{t+n}
        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next')
            a_ = obs_next_result.act
            batch.act = to_torch_as(batch.act, a_)
            target_q = torch.min(
                self.critic1_old(batch.obs_next),
                self.critic2_old(batch.obs_next),
            ) - self._alpha * obs_next_result.log_prob
            target_q = (obs_next_result.dist * target_q).sum(
                    axis=1, keepdim=True)
        return target_q

    def learn(self, batch: Batch, **kwargs) -> Dict[str, float]:

        # critic 1
        current_q1 = self.critic1(batch.obs)
        a_ = to_torch_as(batch.act, current_q1).long().unsqueeze(1)
        current_q1 = current_q1.gather(1, a_)
        target_q = to_torch_as(batch.returns, current_q1)[:, None]
        critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        # critic 2
        current_q2 = self.critic2(batch.obs)
        current_q2 = current_q2.gather(1, a_)
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # actor
        obs_result = self(batch)
        current_q1a = self.critic1(batch.obs)
        current_q2a = self.critic2(batch.obs)
        actor_loss = self._alpha * obs_result.log_prob - torch.min(
            current_q1a, current_q2a)
        actor_loss = (obs_result.dist * actor_loss).sum(axis=1).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()

        return {
            'loss/actor': actor_loss.item(),
            'loss/critic1': critic1_loss.item(),
            'loss/critic2': critic2_loss.item(),
        }
