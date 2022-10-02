from copy import deepcopy
from typing import Dict, List, Union, Optional, Callable

import numpy as np
import os
import torch
import torch.nn.functional as F

from tianshou.policy import BasePolicy
from tianshou.data import Batch, to_torch, to_torch_as, ReplayBuffer


class SACDMultiCCWMPolicy(BasePolicy):
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
    :param int estimation_step: the number of estimation step, should be an int
            greater than 0, defaults to 1.
    :param int num_adv: the number of adversaries
    :param int num_landmark: the number of landmarks
    :param list obs_radii: the observable radii of the agents
    :param str intr_rew_options: a string of the intrinsic reward type being used
    :param bool grads_logging: whether to log gradients or not

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self,
                 actors: List[torch.nn.Module],
                 actor_optims: List[torch.optim.Optimizer],
                 critic1s: List[torch.nn.Module],
                 critic1_optims: List[torch.optim.Optimizer],
                 critic2s: List[torch.nn.Module],
                 critic2_optims: List[torch.optim.Optimizer],
                 world_models: List[torch.nn.Module],
                 wm_optims: List[torch.optim.Optimizer],
                 dist_fn: torch.distributions.Distribution
                 = torch.distributions.Categorical,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 alpha: float = 0.2,
                 reward_normalization: bool = False,
                 ignore_done: bool = False,
                 estimation_step: int = 1,
                 num_adv: int = 0,
                 num_landmark: int = 0,
                 obs_radii: List[float]=None,
                 intr_rew_options: str='000',
                 grads_logging: bool = False,
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
        self.intr_rew_options = intr_rew_options
        self.num_adv = num_adv
        self.total_num_agt = len(actors)
        self.total_num_lm = num_landmark
        self.obs_radii = obs_radii 
        self.partial_obs = max(self.obs_radii) < float('inf')

        assert len(actors) == len(critic1s) == len(critic2s)
        self.actors, self.actor_optims = actors, actor_optims
        self.critic1s, self.critic1_olds = critic1s, deepcopy(critic1s)
        for critic1_old in self.critic1_olds:
            critic1_old.eval()
        self.critic1_optims = critic1_optims
        self.critic2s, self.critic2_olds = critic2s, deepcopy(critic2s)
        for critic2_old in self.critic2_olds:
            critic2_old.eval()
        self.critic2_optims = critic2_optims
        self.world_models = world_models
        self.wm_optims = wm_optims
        self._alpha = alpha
        self.__eps = np.finfo(np.float32).eps.item()
        self.model_list = [('actor_', self.actors),
                           ('critic1_', self.critic1s),
                           ('critic2_', self.critic2s),
                           ('world_model_', self.world_models)]
        self.grads_logging = grads_logging
        self.max_error = 0

    def save(self, logdir, type="best"):
        for prefix, models in self.model_list:
            if type == 'final':
                prefix = 'final_' + prefix
            for i, model in enumerate(models):
                path = os.path.join(logdir, prefix + str(i) + '.pkl')
                torch.save(model.state_dict(), path)

    def load(self, logdir, type="best"):
        for prefix, models in self.model_list:
            if type == 'final':
                prefix = 'final_' + prefix
            for i, model in enumerate(models):
                path = os.path.join(logdir, prefix + str(i) + '.pkl')
                model.load_state_dict(torch.load(
                    path, map_location=lambda storage, _: storage))
        self.critic1_olds = deepcopy(self.critic1s)
        for critic1_old in self.critic1_olds:
            critic1_old.eval()
        self.critic2_olds = deepcopy(self.critic2s)
        for critic2_old in self.critic2_olds:
            critic2_old.eval()

    def train(self) -> None:
        self.training = True
        for actor in self.actors:
            actor.train()
        for critic1 in self.critic1s:
            critic1.train()
        for critic2 in self.critic2s:
            critic2.train()
        for world_model in self.world_models:
            world_model.train()

    def eval(self) -> None:
        self.training = False
        for actor in self.actors:
            actor.eval()
        for critic1 in self.critic1s:
            critic1.eval()
        for critic2 in self.critic2s:
            critic2.eval()
        for world_model in self.world_models:
            world_model.eval()

    def sync_weight(self) -> None:
        for i in range(len(self.actors)):
            critic1 = self.critic1s[i]
            critic2 = self.critic2s[i]
            critic1_old = self.critic1_olds[i]
            critic2_old = self.critic2_olds[i]
            for o, n in zip(critic1_old.parameters(), critic1.parameters()):
                o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
            for o, n in zip(critic2_old.parameters(), critic2.parameters()):
                o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                input: str = 'obs', **kwargs) -> Batch:
        obs = getattr(batch, input)
        logits = []
        acts = []
        log_probs = []
        for i, actor in enumerate(self.actors):
            logit, h = actor(obs[:, i])
            dist = self.dist_fn(logit)
            act = dist.sample()
            log_prob = torch.log(logit + self.__eps)

            acts.append(act.unsqueeze(1))
            logits.append(logit.unsqueeze(1))
            log_probs.append(log_prob.unsqueeze(1))
        acts = torch.cat(acts, dim=1)
        logits = torch.cat(logits, dim=1)
        log_probs = torch.cat(log_probs, dim=1)
        return Batch(
            logits=logits, act=acts, state=h, dist=logits, log_prob=log_probs)

    def imagine_agent_j_obs(self, obs_i, i, j_list):
        num_agts, num_lms = self.total_num_agt, self.total_num_lm
        batch_size = obs_i.shape[0]

        # observation has to have [agent_vis_mask] + [lm_vis_mask] + agent_pos + agent_vel + entity_pos as the initial elements
        agt_mask = obs_i[:, :num_agts] #agents visible to i, (bs,n_agts)
        lm_mask = obs_i[:, num_agts:num_agts+num_lms] #landmarks visible to i, (bs,n_lms)

        agt_pos = np.reshape(obs_i[:, num_agts+num_lms:3*num_agts+num_lms], (batch_size, num_agts, 2)) #visible agent pos,(bs,n_agts,2)
        agt_vel =  np.reshape(obs_i[:, 3*num_agts+num_lms:5*num_agts+num_lms], (batch_size, num_agts, 2)) #visible agent vel, (bs,n_agts*2)
        lm_pos =  np.reshape(obs_i[:, 5*num_agts+num_lms:5*num_agts+3*num_lms],  (batch_size, num_lms, 2)) #visible landmark pos, (bs,n_lms*2)
        
        obs_j_list, sum_obs_percents = [], 0.0
        # print('init obs:', obs_i[:5, :])
        for j in j_list:
            # print("agt j:", j)
            if j == i:
                obs_j_list.append(obs_i)
            else:
                if j < self.num_adv: # agent i shouldn't see j's special properties (e.g. color) if j is adversary
                    obs_j = np.zeros(obs_i.shape)
                else:
                    obs_j = obs_i.copy()
                j_mask = np.expand_dims(agt_mask[:, j], axis=-1) #if j visible to i, (bs,)
                
                j_pos = np.expand_dims(agt_pos[:, j, :], axis=1) #j's pos (bs,2)
                agt_pos_delta = np.tile(j_pos, (1, num_agts, 1)) - agt_pos #(bs,n_agts,2)
                lm_pos_delta = np.tile(j_pos, (1, num_lms, 1)) - lm_pos
                j_agt_mask = np.sqrt(np.sum(np.square(agt_pos_delta), axis=-1)) <= self.obs_radii[j] #agts visible to j, (bs,n_agts)
                j_lm_mask =  np.sqrt(np.sum(np.square(lm_pos_delta), axis=-1)) <= self.obs_radii[j] #landmarks visible to j, (bs,n_lms)
                # print("j mask:", j_mask[:5, :])
                # print("j pos:", j_pos[:5, :])
                # print("agt pos:", agt_pos[:5, :])
                # print("j agt mask:", j_agt_mask[:5, :])
                # print("j lm mask:", j_lm_mask[:5, :])

                j_agt_mask = np.expand_dims(np.logical_and(j_agt_mask,agt_mask), axis=-1) # the agents visible by both agts i and j
                j_lm_mask = np.expand_dims(np.logical_and(j_lm_mask,lm_mask), axis=-1) # the landmarks visible by both agts i and j
                agt_pos_j = j_agt_mask * agt_pos
                agt_vel_j = j_agt_mask * agt_vel
                lm_pos_j = j_lm_mask * lm_pos
                # print("agt pos j:", agt_pos_j[:5, :])
                # print("agent vel j:", agt_vel_j[:5, :])
                # print("lm pos j:", lm_pos_j[:5, :])

                obs_j[:, :num_agts] = np.squeeze(j_agt_mask)
                obs_j[:, num_agts:num_agts+num_lms] = np.squeeze(j_lm_mask)
                obs_j[:, num_agts+num_lms:3*num_agts+num_lms] = np.reshape(agt_pos_j, (batch_size, num_agts*2))
                obs_j[:, 3*num_agts+num_lms:5*num_agts+num_lms] = np.reshape(agt_vel_j, (batch_size, num_agts*2))
                obs_j[:, 5*num_agts+num_lms:5*num_agts+3*num_lms] = np.reshape(lm_pos_j, (batch_size, num_lms*2))
                final_obs_j = np.tile(j_mask, (1, obs_j.shape[1])) * obs_j
                # print("if j is visible?", j_mask[:5, :])
                # print("final obs j:", final_obs_j[:5, :])
                obs_j_list.append(final_obs_j)
                obs_percent = np.count_nonzero(final_obs_j, axis=1) / final_obs_j.shape[-1]
                # print(f"final obs {j}", final_obs_j[:10, :])
                # print(f"percent for {j}:", obs_percent[:10])
                sum_obs_percents += obs_percent

        return obs_j_list, sum_obs_percents

    def calculate_intrinsic_reward(self, batch: Batch) -> Batch:
        num_agents = self.total_num_agt
        batch_size = batch.obs.shape[0]
        intr_rew = np.zeros(batch.rew.shape)
        if not self.partial_obs or self.intr_rew_options in ['curio_self', 'elign_self']: # for full obs, self-pred baseline (partial obs) and curiosity baseline
            # compute my own prediction loss and assume it's also agent j's prediction loss on my obs + act
            l2_losses = np.zeros((batch_size, num_agents)) # (batch_size, n_agents)
            for i in range(self.num_adv, num_agents):
                world_model = self.world_models[i]
                input_i = np.concatenate((batch.obs[:,i], np.expand_dims(batch.act[:,i], axis=-1)), axis=1)
                pred_next_obs_i = world_model(input_i)
                l2_losses[:, i] = np.linalg.norm(batch.obs_next[:,i] - pred_next_obs_i.detach().cpu().numpy(), axis=1)
                if self.intr_rew_options == 'self_adv':
                    adv_mask = batch.obs[:,i][:, :self.num_adv] # adversaries' visibility masks are positioned the first in dim=1
                    # print(f"adv mask for {i}", adv_mask[:10, :])
                    loss_mask = ~np.all(adv_mask == 0, axis=1)
                    # print(f"loss mask for {i}", loss_mask[:10])
                    # print(f"BEFORE loss for {i}", l2_losses[:10, i])
                    l2_losses[:, i] *= loss_mask
                    # print(f"AFTER loss for {i}", l2_losses[:10, i])
            if np.max(l2_losses) > self.max_error:
                self.max_error = np.max(l2_losses)

            if self.intr_rew_options == 'self_adv':
                intr_rew = l2_losses
            elif self.intr_rew_options == 'curio_self':
                intr_rew = l2_losses / self.max_error
            else: # only self pred
                intr_rew = -l2_losses # (batch_size, n_agents)
            
        else:
            if self.intr_rew_options == 'elign_both': # intr rew from both both
                j_list = list(range(num_agents))
            elif self.intr_rew_options == 'elign_team' or self.intr_rew_options == 'curio_team': # intr rew from good agts only
                j_list = list(range(self.num_adv, num_agents))
            elif self.intr_rew_options == 'elign_adv': # intr rew from adversaries only
                j_list = list(range(0, self.num_adv))
            else:  # no intr rew
                print("Invalid intr rew options.")
                raise NotImplementedError
            inter_obs_percents = np.zeros((batch_size, num_agents))
            inter_freqs = np.zeros((batch_size, num_agents))
            for i in range(self.num_adv, num_agents): # apply intr rew only on good agents
                obs_j_list, sum_obs_percents = self.imagine_agent_j_obs(batch.obs[:,i], i, j_list)
                
                obs_n = np.concatenate(obs_j_list, axis=0) #(bs*j, obs_dim), where j = len(j_list)
                act_n = np.concatenate([batch.act[:,i]] * len(j_list), axis=0) #(bs*j, )
                inputs = np.concatenate((obs_n, np.expand_dims(act_n, axis=-1)), axis=1) #(bs*j, obs_dim+1)
                inputs = to_torch(inputs, device=self.world_models[i].device, dtype=torch.float)

                # duplicate obs_next_i j times and concatenate them as the ground truth
                true_next_obs_n = np.concatenate([batch.obs_next[:,i]] * len(j_list), axis=0) #(bs*j, obs_dim)
                true_next_obs_n = to_torch(true_next_obs_n, device=self.world_models[i].device, dtype=torch.float)

                # get the prediction losses and 
                world_model = self.world_models[i]
                pred_next_obs_n = world_model(inputs)
                
                pred_losses = torch.norm(true_next_obs_n - pred_next_obs_n, p=2, dim=1) #(bs*j, )

                # zero out the losses for invisible agents
                obs_mask = to_torch(~np.all(obs_n == 0, axis=1), device=self.world_models[i].device, dtype=torch.float)
                pred_losses *= obs_mask
                nonzero_obs_count = np.count_nonzero(batch.obs[:,i][:, j_list], axis=1)
                inter_num = nonzero_obs_count - 1 # minus self 

                inter_obs_percents[:, i] = sum_obs_percents / np.maximum(inter_num, np.ones(inter_num.shape)) # avoid dividing by 0
                inter_freqs[:, i] = np.minimum(inter_num, np.ones(inter_num.shape)) # 1 if there's any intersection
                # assert np.count_nonzero(inter_obs_percents[:, i] == 0) >= np.count_nonzero(inter_num == 0)
                # print(f'inter freq for {i}:', inter_freqs[:10, i])
                # print(f'inter obs size for {i}:', inter_obs_percents[:10, i] )

                pred_losses = pred_losses.view(batch_size, -1) #(bs, j)

                if self.intr_rew_options == 'elign_both': # intr rew from both both
                    # incentivize good agts to be 1) unpredictable by advs and 2) predictable by good agts
                    intr_rew[:, i] += 1 / nonzero_obs_count * (pred_losses[:, :self.num_adv].sum(dim=1) - pred_losses[:, self.num_adv:].sum(dim=1)).detach().cpu().numpy()
                elif self.intr_rew_options == 'elign_team': # intr rew from good agts only
                    intr_rew[:, i] += 1 / nonzero_obs_count * -pred_losses.sum(dim=1).detach().cpu().numpy()
                elif self.intr_rew_options == 'elign_adv' or self.intr_rew_options == 'curio_team': # maximizing losses: 1) intr rew from adversaries only; 2) ma curiosity
                    intr_rew[:, i] += 1 / nonzero_obs_count * pred_losses.sum(dim=1).detach().cpu().numpy()
                else:  # no intr rew
                    print("Invalid intr rew options. Should use SACDMultiPolicy instead.")
                    raise NotImplementedError
        
    
        intr_rew = 1 / (batch.obs.shape[-1]) * intr_rew

        output = {}
        #for i in range(num_agents):
        for i in range(self.num_adv, num_agents):
            output[f'intr_rew/actor_{i}'] = intr_rew[:, i].mean()
            if self.partial_obs and self.intr_rew_options in ['elign_team', 'elign_adv', 'elign_both', 'curio_team']:
                output[f'inter_freq/agent_{i}'] = inter_freqs[:, i].mean()
                denom = output[f'inter_freq/agent_{i}'] if output[f'inter_freq/agent_{i}'] != 0.0 else 1.0
                output[f'inter_obs_size/agent_{i}'] = (inter_obs_percents[:, i] / denom).mean()
        batch.rew += intr_rew
        return batch, output
    
    def compute_return(self,
                       batch: Batch,
                       buffer: ReplayBuffer,
                       indice: np.ndarray,
                       target_q_fn: Callable[[ReplayBuffer, np.ndarray],
                       torch.Tensor],
                       gamma: float = 0.99,
                       rew_norm: bool = False):
        if rew_norm:
            rand_indices = np.random.choice(
                len(buffer), min(len(buffer), 10000), replace=False)
            bfr = buffer.rew[rand_indices]  # avoid large buffer
            if bfr.ndim == 1:
                mean, std = bfr.mean(), bfr.std()
                if np.isclose(std, 0):
                    mean, std = 0, 1
            else:
                mean, std = bfr.mean(axis=0), bfr.std(axis=0)
                close = np.isclose(std, 0)
                std[close] = 1
                mean[close] = 0
        else:
            mean, std = 0, 1

        rew = (batch.rew - mean) / std
        gammas = np.zeros((indice.shape[0], batch.done.shape[1])) + gamma
        target_qs = target_q_fn(buffer, indice)
        returns = to_torch_as(rew, target_qs)
        gammas = to_torch_as(gammas, target_qs)
        batch.returns = target_qs * gammas + returns
        return batch

    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indice: np.ndarray) -> Batch:
        if self._rm_done:
            batch.done = np.zeros_like(batch.done)
        batch, output = self.calculate_intrinsic_reward(batch)
        batch = self.compute_return(
                batch, buffer, indice, self._target_q, self._gamma, self._rew_norm)
        return batch, output

    def _target_q(self, buffer: ReplayBuffer,
                  indice: np.ndarray) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs: s_{t+n}
        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next')
            a_ = obs_next_result.act
            batch.act = to_torch_as(batch.act, a_)
            target_qs = []
            for i in range(len(self.actors)):             
                critic1_old = self.critic1_olds[i]
                critic2_old = self.critic2_olds[i]
                target_q = torch.min(
                        critic1_old(to_torch(batch.obs_next, device=a_.device), a_),
                        critic2_old(to_torch(batch.obs_next, device=a_.device), a_),
                )  - self._alpha * obs_next_result.log_prob[:, i]
                target_q = (obs_next_result.dist[:, i] * target_q).sum(
                            dim=1, keepdim=True)
                target_qs.append(target_q)
            target_qs = torch.cat(target_qs, dim=1)
        return target_qs

    def learn(self, batch: Batch, **kwargs) -> Dict[str, float]:
        total_actor_loss = np.zeros(len(self.actors))
        total_critic1_loss = np.zeros(len(self.actors))
        total_critic2_loss = np.zeros(len(self.actors))
        total_wm_loss = np.zeros(len(self.actors))
        total_returns = np.zeros(len(self.actors))

        for i in range(len(self.actors)):
            actor = self.actors[i]
            actor_optim = self.actor_optims[i]

            critic1 = self.critic1s[i]
            critic1_optim = self.critic1_optims[i]
            critic2 = self.critic2s[i]
            critic2_optim = self.critic2_optims[i]

            # critic 1
            current_q1 = critic1(batch.obs, batch.act)
            a_ = to_torch_as(batch.act[:, i], current_q1).long().unsqueeze(1)
            current_q1 = current_q1.gather(1, a_)
            target_q = to_torch_as(batch.returns[:, i], current_q1)[:, None]
            critic1_loss = F.mse_loss(current_q1, target_q)
            critic1_optim.zero_grad()
            critic1_loss.backward()
            critic1_optim.step()

            # critic 2
            current_q2 = critic2(batch.obs, batch.act)
            current_q2 = current_q2.gather(1, a_)
            critic2_loss = F.mse_loss(current_q2, target_q)
            critic2_optim.zero_grad()
            critic2_loss.backward()
            critic2_optim.step()

            current_q1a = critic1(batch.obs, batch.act)
            current_q2a = critic2(batch.obs, batch.act)
            
            # actor
            logit, _ = actor(batch.obs[:, i])
            log_prob = torch.log(logit + self.__eps)
            actor_loss = self._alpha * log_prob - torch.min(
                    current_q1a, current_q2a)
            actor_loss = (logit * actor_loss).sum(dim=1).mean()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            world_model = self.world_models[i]
            inputs = np.concatenate((batch.obs[:, i], np.expand_dims(batch.act[:, i], axis=-1)), axis=1)
            pred_next_obs = world_model(to_torch(inputs, device=world_model.device, dtype=torch.float))
            true_next_obs = to_torch(batch.obs_next[:, i], device=world_model.device, dtype=torch.float)
            wm_loss = F.mse_loss(pred_next_obs, true_next_obs)
            self.wm_optims[i].zero_grad()
            wm_loss.backward()
            self.wm_optims[i].step()

            total_actor_loss[i] = actor_loss.item()
            total_critic1_loss[i] = critic1_loss.item()
            total_critic2_loss[i] = critic2_loss.item()
            total_wm_loss[i] = wm_loss.item()
            total_returns[i] = batch.returns[:, i].mean()

        self.sync_weight()

        output = {}
        for i in range(len(self.actors)):
            output[f'models/actor_{i}'] = total_actor_loss[i]
            output[f'models/critic1_{i}'] = total_critic1_loss[i]
            output[f'models/critic2_{i}'] = total_critic2_loss[i]
            output[f'models/returns_{i}'] = total_returns[i]
            output[f'models/world_model_{i}'] = total_wm_loss[i]
            if self.grads_logging:
                for tag, value in self.actors[i].named_parameters():
                    if value.grad is not None:
                        output[f'grads/actor_{i}_{tag}'] = value.grad
            
                for (tag1, value1), (tag2, value2) in zip(self.critic1s[i].named_parameters(), self.critic2s[i].named_parameters()):
                    if value1.grad is not None:
                        output[f'grads/critic1_{i}_{tag1}'] = value1.grad
                    if value2.grad is not None:
                        output[f'grads/critic2_{i}_{tag2}'] = value2.grad

        output[f'loss/actor'] = total_actor_loss.sum()
        output[f'loss/critic1'] = total_critic1_loss.sum()
        output[f'loss/critic2'] = total_critic2_loss.sum()
        output[f'loss/world_model'] = total_wm_loss.sum()

        return output
