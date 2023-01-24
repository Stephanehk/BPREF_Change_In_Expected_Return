import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_dmcontrol_env, make_vec_metaworld_env
from stable_baselines3 import PPO_CUSTOM


from StrippedNet import EmbeddingNet

def generate_demos(env, episode_count=1):

    #TODO: add more episodes!!! ALSO ADD PPO AGENT TRAINED FROM DIFFERENT CHECKPOINTS TO CREATE MORE DIVERSE DEMONSTRATIONS
    demonstrations = []
    learning_returns = []
    learning_rewards = []

    checkpoints = []
    for i in range(1,111,5):
        checkpoints.append("saved_models/PPO_walker_walk_" + str(i) + ".zip")
    checkpoints.append("saved_models/PPO_walker_walk.zip")


    rewards = []
    obs = []
    print (len(checkpoints))
    for checkpoint in checkpoints: 
        agent = PPO_CUSTOM.load(checkpoint)
        for i in range (episode_count):
            done = False  
            r = 0
            ob = env.reset()
            steps = 0
            acc_reward = 0
            while True:
                action, _states = agent.predict(ob, deterministic=True) #do we want our agent to be deterministic?
                ob, r, done, info = env.step(action)
                obs.append(ob[0])
                rewards.append(r[0])
                steps += 1
                acc_reward += r[0]
                if done:
                    break
         
    return obs, rewards

device = 'cpu'
state_dims = 24
encoding_dims = 100
env_name = "walker_walk"
pretrained_network = "saved_models/walker_walk_pretrained_100.params"

reward_net = EmbeddingNet(state_dims, encoding_dims)
reward_net.load_state_dict(torch.load(pretrained_network, map_location=device))
num_features = reward_net.fc2.in_features
print("reward is linear combination of ", num_features, "features")


reward_net.fc2 = nn.Linear(num_features, 1, bias=False) #last layer just outputs the scalar reward = w^T \phi(s)
reward_net.to(device)
#freeze all weights so there are no gradients (we'll manually update the last layer via proposals so no grads required)
for name, param in reward_net.named_parameters():
    # if name != "fc2.weight":
    param.requires_grad = False

env = make_vec_dmcontrol_env(
        env_name,     
        n_envs=1, 
        monitor_dir=None,
        seed=1234)
env = VecNormalize(env, norm_reward=False)
gt_obs, gt_rews = generate_demos(env)

print (np.mean(gt_rews))
print (np.std(gt_rews))

pred_rew_features = []
for ob in gt_obs:
    ob = torch.tensor(ob)
    pred = reward_net.state_feature(ob).numpy()
    pred_rew_features.append(pred)


model_ols =  linear_model.LinearRegression()
# model_ols =  linear_model.Ridge()

# pred_rew_features = np.zeros(len(gt_rews)).reshape(-1, 1)
model_ols.fit(pred_rew_features,gt_rews)
coef = model_ols.coef_
intercept = model_ols.intercept_
print('coef= ', coef)
print('intercept= ', intercept)
reward_pred = model_ols.predict(pred_rew_features)
print("Mean squared error= ", mean_squared_error(gt_rews, reward_pred))
print ("Score= ", model_ols.score(pred_rew_features,gt_rews))

print ("=========================")

model_ols =  linear_model.LinearRegression()
pred_rew_features = np.zeros(len(gt_rews)).reshape(-1, 1)
model_ols.fit(pred_rew_features,gt_rews)
coef = model_ols.coef_
intercept = model_ols.intercept_
print('coef= ', coef)
print('intercept= ', intercept)
reward_pred = model_ols.predict(pred_rew_features)
print("Mean squared error= ", mean_squared_error(gt_rews, reward_pred))
print ("Score= ", model_ols.score(pred_rew_features,gt_rews))




