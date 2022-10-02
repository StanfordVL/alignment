"""
PyTorch policy class used for SAC.
"""

# from turtle import back
from xml.dom import NotSupportedErr
import gym
from gym.spaces import Box, Discrete
import logging
import tree  # pip install dm_tree
from typing import Dict, List, Optional, Tuple, Type, Union
from policy_template import build_policy_class
from models import FullyConnectedNetwork as TorchFC
from torch_policy import TorchPolicy
import numpy as np
import ray
import ray.experimental.tf_utils
from ray.rllib.agents.sac.sac_tf_policy import (
    build_sac_model,
    postprocess_trajectory,
    validate_spaces,
)
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
    TorchDirichlet,
    TorchSquashedGaussian,
    TorchDiagGaussian,
    TorchBeta,
)

from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    concat_multi_gpu_td_errors,
    huber_loss,
)
from ray.rllib.utils.typing import (
    LocalOptimizer,
    ModelInputDict,
    TensorType,
    TrainerConfigDict,
)

torch, nn = try_import_torch()
F = nn.functional

logger = logging.getLogger(__name__)


def to_torch(x: Union[torch.Tensor, dict, np.ndarray],
             dtype: Optional[torch.dtype] = None,
             device: Union[str, int] = 'cpu'
             ) -> Union[dict, torch.Tensor]:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
        if dtype is not None:
            x = x.type(dtype)
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        x = x.to(device)
    elif isinstance(x, dict):
        for k, v in x.items():
            if k == 'obs':
                x[k] = to_torch(v, dtype, device)
            # try:
            #     x[k] = to_torch(v, dtype, device)
            # except TypeError:
            #     print(f"The object with key {k} could not be converted.")

    # elif isinstance(x, Batch):
    #     x.to_torch(dtype, device)
    return x

def _get_dist_class(
    policy: Policy, config: TrainerConfigDict, action_space: gym.spaces.Space
) -> Type[TorchDistributionWrapper]:
    """Helper function to return a dist class based on config and action space.
    Args:
        policy (Policy): The policy for which to return the action
            dist class.
        config (TrainerConfigDict): The Trainer's config dict.
        action_space (gym.spaces.Space): The action space used.
    Returns:
        Type[TFActionDistribution]: A TF distribution class.
    """
    if hasattr(policy, "dist_class") and policy.dist_class is not None:
        return policy.dist_class
    elif config["model"].get("custom_action_dist"):
        action_dist_class, _ = ModelCatalog.get_action_dist(
            action_space, config["model"], framework="torch"
        )
        return action_dist_class
    elif isinstance(action_space, Discrete):
        return TorchCategorical
    elif isinstance(action_space, Simplex):
        return TorchDirichlet
    else:
        assert isinstance(action_space, Box)
        if config["normalize_actions"]:
            return (
                TorchSquashedGaussian
                if not config["_use_beta_distribution"]
                else TorchBeta
            )
        else:
            return TorchDiagGaussian


def build_sac_model_and_action_dist(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: TrainerConfigDict,
) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
    """Constructs the necessary ModelV2 and action dist class for the Policy.
    Args:
        policy (Policy): The TFPolicy that will use the models.
        obs_space (gym.spaces.Space): The observation space.
        action_space (gym.spaces.Space): The action space.
        config (TrainerConfigDict): The SAC trainer's config dict.
    Returns:
        ModelV2: The ModelV2 to be used by the Policy. Note: An additional
            target model will be created in this function and assigned to
            `policy.target_model`.
    """
    model = build_sac_model(policy, obs_space, action_space, config)
    action_dist_class = _get_dist_class(policy, config, action_space)
    return model, action_dist_class


def action_distribution_fn(
    policy: Policy,
    model: ModelV2,
    input_dict: ModelInputDict,
    state_batches: Optional[List[TensorType]] = None,
    seq_lens: Optional[TensorType] = None,
    prev_action_batch: Optional[TensorType] = None,
    prev_reward_batch=None,
    explore: Optional[bool] = None,
    timestep: Optional[int] = None,
    is_training: Optional[bool] = None
) -> Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
    """The action distribution function to be used the algorithm.
    An action distribution function is used to customize the choice of action
    distribution class and the resulting action distribution inputs (to
    parameterize the distribution object).
    After parameterizing the distribution, a `sample()` call
    will be made on it to generate actions.
    Args:
        policy (Policy): The Policy being queried for actions and calling this
            function.
        model (TorchModelV2): The SAC specific Model to use to generate the
            distribution inputs (see sac_tf|torch_model.py). Must support the
            `get_policy_output` method.
        input_dict (ModelInputDict): The input-dict to be used for the model
            call.
        state_batches (Optional[List[TensorType]]): The list of internal state
            tensor batches.
        seq_lens (Optional[TensorType]): The tensor of sequence lengths used
            in RNNs.
        prev_action_batch (Optional[TensorType]): Optional batch of prev
            actions used by the model.
        prev_reward_batch (Optional[TensorType]): Optional batch of prev
            rewards used by the model.
        explore (Optional[bool]): Whether to activate exploration or not. If
            None, use value of `config.explore`.
        timestep (Optional[int]): An optional timestep.
        is_training (Optional[bool]): An optional is-training flag.
    Returns:
        Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
            The dist inputs, dist class, and a list of internal state outputs
            (in the RNN case).
    """
    # Get base-model output (w/o the SAC specific parts of the network).
    model_out, _ = model(input_dict, [], None)
    # Use the base output to get the policy outputs from the SAC model's
    # policy components.
    distribution_inputs = model.get_policy_output(model_out)
    # Get a distribution class to be used with the just calculated dist-inputs.
    action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)

    return distribution_inputs, action_dist_class, []

def build_world_model(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: TrainerConfigDict,
) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
    """Constructs the necessary ModelV2 and action dist class for the Policy.
    Args:
        policy (Policy): The TFPolicy that will use the models.
        obs_space (gym.spaces.Space): The observation space.
        action_space (gym.spaces.Space): The action space.
        config (TrainerConfigDict): The SAC trainer's config dict.
    Returns:
        ModelV2: The ModelV2 to be used by the Policy. Note: An additional
            target model will be created in this function and assigned to
            `policy.target_model`.
    """
    num_outputs = config['world_model_output_dim'] if 'world_model_output_dim' in config else int(np.product(obs_space.shape))
    model = TorchFC(
        obs_space, action_space, num_outputs, config, name="fcnet", 
        input_dim=int(np.product(obs_space.shape))+1
    )
    return model


def postprocess_trajectory_torch(policy, batch, other_agent_batches, episode):
    batch = postprocess_trajectory(policy, batch, other_agent_batches, episode)
    if other_agent_batches:
        batch = add_intrinsic_reward(policy, batch,other_agent_batches)
    return batch

def imagine_agent_j_obs(batch, j_list, radius):
    # Obs holds:
    #      - [0-22) 11 (x,y) positions for each player of the left team.
    #      - [22-44) 11 (x,y) motion vectors for each player of the left team.
    #      - [44-66) 11 (x,y) positions for each player of the right team.
    #      - [66-88) 11 (x,y) motion vectors for each player of the right team.
    #      - [88-94) the ball_position and the ball_direction as (x,y,z)
    #      - [94-97) one hot encoding of who controls the ball.
    #        [1, 0, 0]: nobody, [0, 1, 0]: left team, [0, 0, 1]: right team.
    #      - [97-108) one hot encoding of size 11 to indicate who is the active player
    #        in the left team.
    #      - [108-115) one hot encoding of the game mode. Vector of size 7 with the
    #        following meaning:
    #        {NormalMode, KickOffMode, GoalKickMode, FreeKickMode,
    #         CornerMode, ThrowInMode, PenaltyMode}.
    #      Can only be used when the scenario is a flavor of normal game
    #      (i.e. 11 versus 11 players).
    batch_size = batch[SampleBatch.OBS].shape[0]
    obs_dim = batch[SampleBatch.OBS].shape[1]
    num_player = (obs_dim - 16) // 5
    num_agent = 4
    num_adv = num_player - num_agent

    player_pos_idx = np.array(range(num_agent*2))
    player_dir_idx = np.array(range(num_agent*2, num_agent*4))
    adv_pos_idx = np.array(range(num_agent*4, num_agent*4+num_adv*2))
    adv_dir_idx = np.array(range(num_agent*4+num_adv*2, num_agent*4+num_adv*4))
    ball_pos_idx = np.array(range(num_player*4, num_player*4+3))
    ball_dir_idx = np.array(range(num_player*4+3, num_player*4+6))
    ball_ctrl_idx = np.array(range(num_player*4+6, num_player*4+9))
    active_player_onehot_idx = np.array(range(num_player*4+9, num_player*4+9+num_player))
    game_mode_onehot_idx = np.array(range(num_player*5+9, num_player*5+9+7))
    assert num_player*5+9+7 == batch[SampleBatch.OBS].shape[1]
    obs_i = batch[SampleBatch.OBS]
    player_id = np.argmax(obs_i[:, active_player_onehot_idx])
    raw_all_player_pos = obs_i[:, player_pos_idx] #(bs,11*2)
    all_player_pos = np.reshape(raw_all_player_pos, (batch_size,num_agent,2)) #(bs,11,2)
    raw_all_player_dir = obs_i[:, player_dir_idx]
    raw_all_adv_pos = obs_i[:, adv_pos_idx]
    all_adv_pos = np.reshape(raw_all_adv_pos, (batch_size,num_adv,2))
    raw_all_adv_dir = obs_i[:, adv_dir_idx]

    i_pos = all_player_pos[:, player_id, :]
    ball_pos = np.reshape(obs_i[:, ball_pos_idx], (batch_size,1,3))
    
    obs_j_list = []
    for j_id in j_list:
        if j_id == player_id:
            obs_j_list.append(obs_i)
        else:
            obs_j = -np.ones(obs_i.shape)
            if j_id < num_agent:
                j_pos = all_player_pos[:, j_id, :]
            else:
                j_pos = all_adv_pos[:, j_id-num_agent, :]
            
            j_i_diff = j_pos - i_pos 
            j_mask = np.sqrt(np.sum(np.square(j_i_diff), axis=-1)) <= radius
            if j_mask == 1: # j is out of i's receptive field
                j_others_diff =  np.tile(j_pos, (1, num_agent, 1)) - all_player_pos
                player_mask = np.sqrt(np.sum(np.square(j_others_diff), axis=-1)) <= radius
                vis_player_idx = np.nonzero(np.repeat(player_mask, 2))[0]
                
                j_adv_diff =  np.tile(j_pos, (1, num_adv, 1)) - all_adv_pos
                adv_mask = np.sqrt(np.sum(np.square(j_adv_diff), axis=-1)) <= radius
                vis_adv_idx = np.nonzero(np.repeat(adv_mask, 2))[0]
                
                j_ball_diff = j_pos - ball_pos[:,:,:-1]
                ball_mask = np.sqrt(np.sum(np.square(j_ball_diff), axis=-1)) <= radius

                obs_j[:, vis_player_idx] = raw_all_player_pos[:,vis_player_idx]
                obs_j[:, num_agent*2+vis_player_idx] = raw_all_player_dir[:,vis_player_idx]

                obs_j[:, num_agent*4+vis_adv_idx] = raw_all_adv_pos[:,vis_adv_idx]
                obs_j[:, num_agent*4+num_adv*2+vis_adv_idx] = raw_all_adv_dir[:,vis_adv_idx]

                if ball_mask:
                    obs_j[:, ball_pos_idx] = obs_i[:, ball_pos_idx] 
                    obs_j[:, ball_dir_idx] = obs_i[:, ball_dir_idx]
                    obs_j[:, ball_ctrl_idx] = obs_i[:, ball_ctrl_idx]

                obs_j[:, active_player_onehot_idx] = obs_i[:, active_player_onehot_idx]
                obs_j[:, game_mode_onehot_idx] = obs_i[:, game_mode_onehot_idx]

            obs_j_list.append(obs_j)

    return obs_j_list

def add_intrinsic_reward(policy, batch: SampleBatch, other_agent_batches):
    batch_size = batch[SampleBatch.OBS].shape[0]
    obs_i = batch[SampleBatch.OBS] #(bs, obs_dim)
    obs_size = obs_i.shape[-1]
    intr_rew = np.zeros(batch[SampleBatch.REWARDS].shape)
    world_model = policy.world_model
    inputs = obs_i
    if policy.align_mode == 'elign_self':
        true_next_obs_i = to_torch(batch[SampleBatch.NEXT_OBS])

        inputs = np.concatenate((obs_i, np.expand_dims(batch[SampleBatch.ACTIONS], axis=-1)), axis=1) #(bs, obs_dim+1)
        inputs = to_torch(inputs, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float)
        input_dict = {}
        input_dict[SampleBatch.OBS] = inputs
        # get self prediction loss  
        pred_next_obs_i, _ = world_model(input_dict)
        
        pred_loss = torch.norm(true_next_obs_i - pred_next_obs_i, p=2, dim=1) #(bs, )
        intr_rew += -pred_loss.detach().cpu().numpy()
    else:
        # get prediction losses from other agents
        num_player = (obs_size - 16) // 5
        num_agent = len(other_agent_batches) + 1 + 1 #goal keeper #0 is uncontrollable
        if policy.align_mode == 'elign_team':
            j_list = list(range(1, num_agent)) # first one is goal keeper (not controlled)
        elif policy.align_mode == 'elign_adv':
            j_list = list(range(num_agent, num_player))
        elif policy.align_mode == 'elign_both':
            j_list = list(range(1, num_player))
        else:
            raise NotSupportedErr
        
        obs_j_list = imagine_agent_j_obs(batch, j_list, policy.env_radius)
        
        obs_n = np.concatenate(obs_j_list, axis=0) #(bs*j, obs_dim), where j = len(j_list)
        act_n = np.concatenate([batch[SampleBatch.ACTIONS]] * len(j_list), axis=0) #(bs*j, )
        inputs = np.concatenate((obs_n, np.expand_dims(act_n, axis=-1)), axis=1) #(bs*j, obs_dim+1)
        inputs = to_torch(inputs, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float)

        # duplicate obs_next_i j times and concatenate them as the ground truth
        true_next_obs_n = np.concatenate([batch[SampleBatch.NEXT_OBS]] * len(j_list), axis=0) #(bs*j, obs_dim)
        true_next_obs_n = to_torch(true_next_obs_n, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float)

        input_dict = {}
        input_dict[SampleBatch.OBS] = inputs # inputs is a concatenation of obs AND action

        # get the prediction losses 
        pred_next_obs_n, _ = world_model(input_dict)
        
        pred_losses = torch.norm(true_next_obs_n - pred_next_obs_n, p=2, dim=1) #(bs*j, )
        #zero out the losses for invisible agents
        obs_mask = ~np.all(obs_n == -1, axis=1)

        pred_losses *= to_torch(obs_mask, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float)

        pred_losses = pred_losses.view(batch_size, -1) #(bs, j)

        visible_agents = np.sum(obs_mask)
        multiplier = 1 / visible_agents if visible_agents != 0 else 0
        if policy.align_mode == 'elign_team':
            intr_rew += multiplier * -pred_losses.sum(dim=1).detach().cpu().numpy()
        elif policy.align_mode == 'elign_adv':
            intr_rew += multiplier * pred_losses.sum(dim=1).detach().cpu().numpy()
        elif policy.align_mode == 'elign_both':
            intr_rew += multiplier * (pred_losses[:, num_agent:num_player].sum(dim=1) - pred_losses[:, 1:num_agent].sum(dim=1)).detach().cpu().numpy()
        else:
            raise NotSupportedErr

    intr_rew = 1 / obs_size * intr_rew

    batch[SampleBatch.REWARDS] += intr_rew
    return batch

def actor_critic_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for the Soft Actor Critic.
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[TorchDistributionWrapper]: The action distr. class.
        train_batch (SampleBatch): The training data.
    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # Look up the target model (tower) using the model tower.
    target_model = policy.target_models[model]

    # Should be True only for debugging purposes (e.g. test cases)!
    deterministic = policy.config["_deterministic_loss"]

    model_out_t, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], _is_training=True), [], None
    )

    model_out_tp1, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True), [], None
    )

    target_model_out_tp1, _ = target_model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True), [], None
    )

    alpha = torch.exp(model.log_alpha)

    # Discrete case.
    if model.discrete:
        # Get all action probs directly from pi and form their logp.
        log_pis_t = F.log_softmax(model.get_policy_output(model_out_t), dim=-1)
        policy_t = torch.exp(log_pis_t)
        log_pis_tp1 = F.log_softmax(model.get_policy_output(model_out_tp1), -1)
        policy_tp1 = torch.exp(log_pis_tp1)
        # Q-values.
        q_t = model.get_q_values(model_out_t)
        # Target Q-values.
        q_tp1 = target_model.get_q_values(target_model_out_tp1)
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(model_out_t)
            twin_q_tp1 = target_model.get_twin_q_values(target_model_out_tp1)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)
        q_tp1 -= alpha * log_pis_tp1

        # Actually selected Q-values (from the actions batch).
        one_hot = F.one_hot(
            train_batch[SampleBatch.ACTIONS].long(), num_classes=q_t.size()[-1]
        )
        q_t_selected = torch.sum(q_t * one_hot, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
        # Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best
    # Continuous actions case.
    else:
        # Sample single actions from distribution.
        action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
        action_dist_t = action_dist_class(model.get_policy_output(model_out_t), model)
        policy_t = (
            action_dist_t.sample()
            if not deterministic
            else action_dist_t.deterministic_sample()
        )
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)
        action_dist_tp1 = action_dist_class(
            model.get_policy_output(model_out_tp1), model
        )
        policy_tp1 = (
            action_dist_tp1.sample()
            if not deterministic
            else action_dist_tp1.deterministic_sample()
        )
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)

        # Q-values for the actually selected actions.
        q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS]
            )

        # Q-values for current policy in given current state.
        q_t_det_policy = model.get_q_values(model_out_t, policy_t)
        if policy.config["twin_q"]:
            twin_q_t_det_policy = model.get_twin_q_values(model_out_t, policy_t)
            q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)

        # Target q network evaluation.
        q_tp1 = target_model.get_q_values(target_model_out_tp1, policy_tp1)
        if policy.config["twin_q"]:
            twin_q_tp1 = target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1
            )
            # Take min over both twin-NNs.
            q_tp1 = torch.min(q_tp1, twin_q_tp1)

        q_t_selected = torch.squeeze(q_t, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)
        q_tp1 -= alpha * log_pis_tp1

        q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best

    # compute RHS of bellman equation
    q_t_selected_target = (
        train_batch[SampleBatch.REWARDS]
        + (policy.config["gamma"] ** policy.config["n_step"]) * q_tp1_best_masked
    ).detach()

    # Compute the TD-error (potentially clipped).
    base_td_error = torch.abs(q_t_selected - q_t_selected_target)
    if policy.config["twin_q"]:
        twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    critic_loss = [torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(base_td_error))]
    if policy.config["twin_q"]:
        critic_loss.append(
            torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error))
        )

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    if model.discrete:
        weighted_log_alpha_loss = policy_t.detach() * (
            -model.log_alpha * (log_pis_t + model.target_entropy).detach()
        )
        # Sum up weighted terms and mean over all batch items.
        alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))
        # Actor loss.
        actor_loss = torch.mean(
            torch.sum(
                torch.mul(
                    # NOTE: No stop_grad around policy output here
                    # (compare with q_t_det_policy for continuous case).
                    policy_t,
                    alpha.detach() * log_pis_t - q_t.detach(),
                ),
                dim=-1,
            )
        )
    else:
        alpha_loss = -torch.mean(
            model.log_alpha * (log_pis_t + model.target_entropy).detach()
        )
        # Note: Do not detach q_t_det_policy here b/c is depends partly
        # on the policy vars (policy sample pushed through Q-net).
        # However, we must make sure `actor_loss` is not used to update
        # the Q-net(s)' variables.
        actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy)

    wm_batch = train_batch.copy()
    inputs = np.concatenate((train_batch[SampleBatch.OBS], np.expand_dims(train_batch[SampleBatch.ACTIONS], axis=-1)), axis=1) #(bs*j, obs_dim+1)
    inputs = to_torch(inputs, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float)
    wm_batch[SampleBatch.OBS] = inputs
    new_obs_pred, _ = policy.world_model(wm_batch)
    wm_loss = F.mse_loss(
        new_obs_pred, train_batch[SampleBatch.NEXT_OBS])
    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["q_t"] = q_t
    model.tower_stats["policy_t"] = policy_t
    model.tower_stats["log_pis_t"] = log_pis_t
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    model.tower_stats["alpha_loss"] = alpha_loss

    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["td_error"] = td_error
    model.tower_stats["wm_loss"] = wm_loss
    # Return all loss terms corresponding to our optimizers.
    return tuple([actor_loss] + critic_loss + [alpha_loss] + [wm_loss])


def stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Stats function for SAC. Returns a dict with important loss stats.
    Args:
        policy (Policy): The Policy to generate stats for.
        train_batch (SampleBatch): The SampleBatch (already) used for training.
    Returns:
        Dict[str, TensorType]: The stats dict.
    """
    q_t = torch.stack(policy.get_tower_stats("q_t"))

    return {
        "actor_loss": torch.mean(torch.stack(policy.get_tower_stats("actor_loss"))),
        "critic_loss": torch.mean(
            torch.stack(tree.flatten(policy.get_tower_stats("critic_loss")))
        ),
        "alpha_loss": torch.mean(torch.stack(policy.get_tower_stats("alpha_loss"))),
        "alpha_value": torch.exp(policy.model.log_alpha),
        "log_alpha_value": policy.model.log_alpha,
        "target_entropy": policy.model.target_entropy,
        "policy_t": torch.mean(torch.stack(policy.get_tower_stats("policy_t"))),
        "mean_q": torch.mean(q_t),
        "max_q": torch.max(q_t),
        "min_q": torch.min(q_t),
    }


def optimizer_fn(policy: Policy, config: TrainerConfigDict) -> Tuple[LocalOptimizer]:
    """Creates all necessary optimizers for SAC learning.
    The 3 or 4 (twin_q=True) optimizers returned here correspond to the
    number of loss terms returned by the loss function.
    Args:
        policy (Policy): The policy object to be trained.
        config (TrainerConfigDict): The Trainer's config dict.
    Returns:
        Tuple[LocalOptimizer]: The local optimizers to use for policy training.
    """
    policy.actor_optim = torch.optim.Adam(
        params=policy.model.policy_variables(),
        lr=config["optimization"]["actor_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )

    critic_split = len(policy.model.q_variables())
    if config["twin_q"]:
        critic_split //= 2

    policy.critic_optims = [
        torch.optim.Adam(
            params=policy.model.q_variables()[:critic_split],
            lr=config["optimization"]["critic_learning_rate"],
            eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
        )
    ]
    if config["twin_q"]:
        policy.critic_optims.append(
            torch.optim.Adam(
                params=policy.model.q_variables()[critic_split:],
                lr=config["optimization"]["critic_learning_rate"],
                eps=1e-7,  # to match tf.keras.optimizers.Adam's eps default
            )
        )
    policy.alpha_optim = torch.optim.Adam(
        params=[policy.model.log_alpha],
        lr=config["optimization"]["entropy_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )

    policy.world_model_optim = torch.optim.Adam(
        params=policy.world_model.parameters(),
        lr=config["optimization"]["actor_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )
    return tuple([policy.actor_optim] + policy.critic_optims + [policy.alpha_optim] + [policy.world_model_optim])

class ComputeTDErrorMixin:
    """Mixin class calculating TD-error (part of critic loss) per batch item.
    - Adds `policy.compute_td_error()` method for TD-error calculation from a
      batch of observations/actions/rewards/etc..
    """

    def __init__(self):
        def compute_td_error(
            obs_t, act_t, rew_t, obs_tp1, done_mask, importance_weights
        ):
            input_dict = self._lazy_tensor_dict(
                {
                    SampleBatch.CUR_OBS: obs_t,
                    SampleBatch.ACTIONS: act_t,
                    SampleBatch.REWARDS: rew_t,
                    SampleBatch.NEXT_OBS: obs_tp1,
                    SampleBatch.DONES: done_mask,
                    PRIO_WEIGHTS: importance_weights,
                }
            )
            # Do forward pass on loss to update td errors attribute
            # (one TD-error value per item in batch to update PR weights).
            actor_critic_loss(self, self.model, None, input_dict)

            # `self.model.td_error` is set within actor_critic_loss call.
            # Return its updated value here.
            return self.model.tower_stats["td_error"]

        # Assign the method to policy (self) for later usage.
        self.compute_td_error = compute_td_error


class TargetNetworkMixin:
    """Mixin class adding a method for (soft) target net(s) synchronizations.
    - Adds the `update_target` method to the policy.
      Calling `update_target` updates all target Q-networks' weights from their
      respective "main" Q-metworks, based on tau (smooth, partial updating).
    """

    def __init__(self):
        # Hard initial update from Q-net(s) to target Q-net(s).
        self.update_target(tau=1.0)

    def update_target(self, tau=None):
        # Update_target_fn will be called periodically to copy Q network to
        # target Q network, using (soft) tau-synching.
        tau = tau or self.config.get("tau")
        model_state_dict = self.model.state_dict()
        # Support partial (soft) synching.
        # If tau == 1.0: Full sync from Q-model to target Q-model.
        target_state_dict = next(iter(self.target_models.values())).state_dict()
        model_state_dict = {
            k: tau * model_state_dict[k] + (1 - tau) * v
            for k, v in target_state_dict.items()
        }

        for target in self.target_models.values():
            target.load_state_dict(model_state_dict)

    @override(TorchPolicy)
    def set_weights(self, weights):
        # Makes sure that whenever we restore weights for this policy's
        # model, we sync the target network (from the main model)
        # at the same time.
        TorchPolicy.set_weights(self, weights)
        self.update_target()


def setup_late_mixins(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: TrainerConfigDict,
) -> None:
    """Call mixin classes' constructors after Policy initialization.
    - Moves the target model(s) to the GPU, if necessary.
    - Adds the `compute_td_error` method to the given policy.
    Calling `compute_td_error` with batch data will re-calculate the loss
    on that batch AND return the per-batch-item TD-error for prioritized
    replay buffer record weight updating (in case a prioritized replay buffer
    is used).
    - Also adds the `update_target` method to the given policy.
    Calling `update_target` updates all target Q-networks' weights from their
    respective "main" Q-metworks, based on tau (smooth, partial updating).
    Args:
        policy (Policy): The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config (TrainerConfigDict): The Policy's config.
    """
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy)


# Build a child class of `TorchPolicy`, given the custom functions defined
# above.
AlignSACPolicy = build_policy_class(
    name="AlignSACPolicy",
    framework='torch',
    loss_fn=actor_critic_loss,
    get_default_config=lambda: ray.rllib.agents.sac.sac.DEFAULT_CONFIG,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory_torch,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_sac_model_and_action_dist,
    make_world_model=build_world_model,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin],
    action_distribution_fn=action_distribution_fn,
)
