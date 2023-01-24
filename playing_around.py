# from agent.sac import SACAgent
# #Model Path:
# agent = SACAgent()
# agent.load(model_dir="/home/stephane/Desktop/BPref/exp/walker_walk/H1024_L2_B1024_tau0.005/sac_unsup0_topk5_sac_lr0.0005_temp0.1_seed12345/",step=1000000)

import numpy as np
import torch
import torch.nn.functional as F
import utils
from torch import nn

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_dmcontrol_env, make_vec_metaworld_env
from agent.critic import DoubleQCritic
from agent.actor import DiagGaussianActor


class VF():
    def __init__(self, obs_dim,action_dim):
        model_dir = "/home/stephane/Desktop/BPref/exp/walker_walk/H1024_L2_B1024_tau0.005/sac_unsup0_topk5_sac_lr0.0005_temp0.1_seed12345/"
        step = 1000000
        self.critic = DoubleQCritic(obs_dim=obs_dim,action_dim=action_dim, hidden_dim=1024, hidden_depth=2)
        self.critic.load_state_dict(torch.load('%s/critic_target_%s.pt' % (model_dir, step)))

        self.actor = DiagGaussianActor(obs_dim=obs_dim,action_dim=action_dim, hidden_dim=1024, hidden_depth=2,log_std_bounds=[-5, 2])
        self.actor.load_state_dict(torch.load('%s/actor_%s.pt' % (model_dir, step)))

    def V(self,obs,sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1

        Q1, Q2 = self.critic(obs, action)
        return torch.min(Q1,Q2)
        # return utils.to_np(action[0])

 #create enviroment
env = make_vec_dmcontrol_env(
    "walker_walk",     
    n_envs=1, 
    monitor_dir=None,
    seed=1234)

print ("Normalizing env")
env = VecNormalize(env, norm_reward=False)
vf = VF(env.observation_space.shape[0], env.action_space.shape[0])