from tianshou.policy.base import BasePolicy
from tianshou.policy.imitation.base import ImitationPolicy
from tianshou.policy.modelfree.dqn import DQNPolicy
from tianshou.policy.modelfree.pg import PGPolicy
from tianshou.policy.modelfree.a2c import A2CPolicy
from tianshou.policy.modelfree.ddpg import DDPGPolicy
from tianshou.policy.modelfree.ppo import PPOPolicy
from tianshou.policy.modelfree.td3 import TD3Policy
from tianshou.policy.modelfree.sac import SACPolicy
from tianshou.policy.modelfree.sac_discrete import SACDiscretePolicy
from tianshou.policy.modelfree.sacd_multi import SACDMultiPolicy
from tianshou.policy.modelfree.sacd_multi_vec import SACDMultiVecPolicy
from tianshou.policy.modelfree.sacd_multi_cc import SACDMultiCCPolicy
from tianshou.policy.modelfree.sacd_multi_wm import SACDMultiWMPolicy
from tianshou.policy.modelfree.sacd_multi_wm_vec import SACDMultiWMVecPolicy
from tianshou.policy.modelfree.sacd_multi_cc_wm import SACDMultiCCWMPolicy
from tianshou.policy.modelfree.sacd_multi_cc_wm_others import SACDMultiCCWMOthersPolicy
from tianshou.policy.modelfree.sacd_multi_comm import SACDMultiCommPolicy
from tianshou.policy.modelfree.sacd_multi_comm_wm import SACDMultiCommWMPolicy
from tianshou.policy.modelfree.strategy import StrategyPolicy
from tianshou.policy.modelfree.diayn import DIAYNPolicy
from tianshou.policy.modelfree.maddpg import MADDPG

__all__ = [
    'BasePolicy',
    'ImitationPolicy',
    'DQNPolicy',
    'PGPolicy',
    'A2CPolicy',
    'DDPGPolicy',
    'PPOPolicy',
    'TD3Policy',
    'SACPolicy',
    'SACDiscretePolicy',
    'SACDMultiPolicy',
    'SACDMultiVecPolicy',
    'SACDMultiCCPolicy',
    'SACDMultiWMPolicy',
    'SACDMultiWMVecPolicy',
    'SACDMultiCCWMPolicy',
    'SACDMultiCCWMOthersPolicy',
    'SACDMultiCommPolicy',
    'SACDMultiCommWMPolicy',
    'StrategyPolicy',
    'DIAYNPolicy',
    'MADDPG'
]
