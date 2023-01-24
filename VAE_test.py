import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import argparse
import sys
import pickle
import gym
from gym import spaces
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import torchvision
from torchvision import transforms


from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_dmcontrol_env, make_vec_metaworld_env
from stable_baselines3 import PPO_CUSTOM


def generate_demos(env, episode_count=2):

    #TODO: add more episodes!!! ALSO ADD PPO AGENT TRAINED FROM DIFFERENT CHECKPOINTS TO CREATE MORE DIVERSE DEMONSTRATIONS
    demonstrations = []
    learning_returns = []
    learning_rewards = []
    all_obs = []

    checkpoints = []
    for i in range(1,111,5):
        checkpoints.append("saved_models/PPO_walker_walk_" + str(i) + ".zip")
    checkpoints.append("saved_models/PPO_walker_walk.zip")

    for checkpoint in checkpoints: 
        agent = PPO_CUSTOM.load(checkpoint)

        for i in range (episode_count):
            done = False
            traj = []
            actions = []
            gt_rewards = []
            r = 0

            ob = env.reset()
            steps = 0
            acc_reward = 0
            while True:
                action, _states = agent.predict(ob, deterministic=True) #do we want our agent to be deterministic?
                ob, r, done, info = env.step(action)

    
                traj.append(ob[0])
                all_obs.append(ob[0])
                actions.append(action[0])

                gt_rewards.append(r[0])
                steps += 1
                acc_reward += r[0]
                if done:
                    break
            print ("return: " + str(acc_reward))
            print("traj length", len(traj))
            print("demo length", len(demonstrations))

            demonstrations.append([traj, actions])
            learning_returns.append(acc_reward)
            learning_rewards.append(gt_rewards)
    return demonstrations, learning_returns, learning_rewards, all_obs


def create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length):
    #collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    times = []
    actions = []
    num_demos = len(demonstrations)

    #add full trajs 

    for n in range(num_trajs):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        #create random partial trajs by finding random start frame and random skip frame
        si = np.random.randint(6)
        sj = np.random.randint(6)
        # step = np.random.randint(3,7)
        step = 1 #removed frameskip

        traj_i = demonstrations[ti][0][si::step]  #slice(start,stop,step)
        traj_j = demonstrations[tj][0][sj::step]
        traj_i_actions = demonstrations[ti][1][si::step] 
        traj_j_actions = demonstrations[tj][1][sj::step]

        if ti > tj:
            label = 0
        else:
            label = 1

        #print(len(list(range(si, len(demonstrations[ti][0]), step))), len(traj_i_actions), len(traj_i) )
        times.append((list(range(si, len(demonstrations[ti][0]), step)), list(range(sj, len(demonstrations[tj][0]), step))))
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        actions.append((traj_i_actions, traj_j_actions))
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))



    #fixed size snippets with progress prior
    for n in range(num_snippets):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        #create random snippets
        #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        min_length = min(len(demonstrations[ti][0]), len(demonstrations[tj][0]))
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)
        if ti < tj: #pick tj snippet to be later than ti
            ti_start = np.random.randint(min_length - rand_length + 1)
            #print(ti_start, len(demonstrations[tj]))
            tj_start = np.random.randint(ti_start, len(demonstrations[tj][0]) - rand_length + 1)
        else: #ti is better so pick later snippet in ti
            tj_start = np.random.randint(min_length - rand_length + 1)
            #print(tj_start, len(demonstrations[ti]))
            ti_start = np.random.randint(tj_start, len(demonstrations[ti][0]) - rand_length + 1)
        traj_i = demonstrations[ti][0][ti_start:ti_start+rand_length] #REMOVED THIS: skip everyother framestack to reduce size
        traj_j = demonstrations[tj][0][tj_start:tj_start+rand_length]
        traj_i_actions = demonstrations[ti][1][ti_start:ti_start+rand_length] #REMOVED THIS: skip everyother framestack to reduce size
        traj_j_actions = demonstrations[tj][1][tj_start:tj_start+rand_length]

        # print (traj_i)
        # print ("==============================================")


        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
        if ti > tj:
            label = 0
        else:
            label = 1
        len1 = len(traj_i)
        len2 = len(list(range(ti_start, ti_start+rand_length, 1)))
        if len1 != len2:
            print("---------LENGTH MISMATCH!------")
            assert False

        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        times.append((list(range(ti_start, ti_start+rand_length, 1)), list(range(tj_start, tj_start+rand_length, 1))))
        actions.append((traj_i_actions, traj_j_actions))

    return training_obs, training_labels, times, actions


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, latent_dim):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        # self.linear2 = torch.nn.Linear(H, H*2)
        self.linear2 = torch.nn.Linear(H, latent_dim)

        self.mu = torch.nn.Linear(latent_dim, latent_dim)
        self.log_var = torch.nn.Linear(latent_dim, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        # x = F.tanh(self.linear3(x))

        mu = self.mu(x)
        log_var = self.log_var(x)
        return x, mu, log_var


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, H, orig_dim):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_dim, H)
        # self.linear2 = torch.nn.Linear(2*H, H)
        self.linear2 = torch.nn.Linear(H, orig_dim)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        # x = F.tanh(self.linear2(x))
        return F.leaky_relu(self.linear2(x))


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
      

    def reparameterize(self, mu, log_var):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, state):
        encoded, mu, log_var = self.encoder(state)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return encoded, decoded, mu, log_var


def latent_loss(encoded, decoded, mu, log_var, target):
    target = F.sigmoid(target)
    recons_loss = F.mse_loss(decoded, target)

    log_var = log_var.squeeze()
    mu = mu.squeeze()
   
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    
    loss = recons_loss + kld_loss
    return loss, kld_loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="walker_walk", help="environment ID")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=123)
    parser.add_argument("--normalize", help="Normalization", type=int, default=1)
    args = parser.parse_args()

    #create enviroment
    env = make_vec_dmcontrol_env(
        args.env,     
        n_envs=1, 
        monitor_dir=None,
        seed=args.seed)


    if args.normalize == 1:
        print ("Normalizing env")
        env = VecNormalize(env, norm_reward=False)

 
    # print (env.action_space)
    # print (env.state_space.n)
    ACTION_DIMS = env.action_space._shape[0]
    STATE_DIMS = env.observation_space._shape[0]
    print("Number of action dimensions", ACTION_DIMS)
    print("Number of state dimensions", STATE_DIMS)


    #load PPO model 
    agent = PPO_CUSTOM.load("saved_models/PPO_walker_walk.zip")
    demonstrations, learning_returns, learning_rewards, all_obs = generate_demos(env)
    print ("Number of obs: " + str(len(all_obs)))


    #normalize observations between 0 and 1
    # max_vals = np.amax(all_obs,axis=0)
    # min_vals = np.amin(all_obs,axis=0)

    # all_obs = (all_obs - min_vals)/(max_vals - min_vals)
    # all_obs = list(all_obs)

    # #sort the demonstrations according to ground truth reward to simulate ranked demos

    min_snippet_length = 600 #min length of trajectory for training comparison
    maximum_snippet_length = 900
    num_trajs = 0
    num_snippets = 60000

    demo_lengths = [len(d[0]) for d in demonstrations]
    demo_action_lengths = [len(d[1]) for d in demonstrations]
    for i in range(len(demo_lengths)):
        assert(demo_lengths[i] == demo_action_lengths[i])
    print("demo lengths", demo_lengths)
    max_snippet_length = min(np.min(demo_lengths), maximum_snippet_length)
    print("max snippet length", max_snippet_length)

    print(len(learning_returns))
    print(len(demonstrations))
    print([a[0] for a in zip(learning_returns, demonstrations)])
    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

    sorted_returns = sorted(learning_returns)
    print(sorted_returns)


    training_obs, training_labels, training_times, training_actions = create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length)
    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))
    print("num_times", len(training_times))
    print("num_actions", len(training_actions))

   #============================== Test ON Walker Environment  ==========================================================================================

    input_dim = 24
    batch_size = 64
    encoding_dim = 12
    hidden_size = 32

    encoder = Encoder(input_dim, hidden_size, encoding_dim)
    decoder = Decoder(encoding_dim, hidden_size, input_dim)
    vae = VAE(encoder, decoder)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    losses = []
    klds = []
    n_epochs = 1000

    # trainig_obs_formatted = []
    # for traj_pair in training_obs:
    #     traj1 = torch.tensor(traj_pair[0]).unsqueeze(1)
    #     traj2 = torch.tensor(traj_pair[1]).unsqueeze(1)
    #     trainig_obs_formatted.append([traj1, traj2])
    # trainig_obs_formatted = torch.tensor(trainig_obs_formatted)

    for epoch in range(n_epochs):
        print (epoch)
        batch_loss = 0
        
        
        # inputs = random.sample(all_obs, batch_size)
        # inputs = torch.tensor(inputs).unsqueeze(1)
        

        # optimizer.zero_grad()

        # encoded, decoded, mu, log_var = vae(inputs)
      
        # loss, kld_loss = latent_loss(encoded, decoded, mu, log_var, inputs)
        # loss.backward()
        # optimizer.step()
        # batch_loss+= loss.detach()

        inputs = random.sample(training_obs, batch_size)
        for traj_pair in inputs:
            traj1 = torch.tensor(traj_pair[0]).unsqueeze(1)
            traj2 = torch.tensor(traj_pair[1]).unsqueeze(1)

            optimizer.zero_grad()

            encoded1, decoded1, mu1, log_var1 = vae(traj1)
            loss1,kld_loss1 = latent_loss(encoded1, decoded1, mu1, log_var1, traj1)

            encoded2, decoded2, mu2, log_var2 = vae(traj2)
            loss2,kld_loss2 = latent_loss(encoded2, decoded2, mu2, log_var2, traj2)


            loss = loss1 + loss2
            kld_loss = kld_loss1 + kld_loss2

            loss.backward()
            optimizer.step()
            batch_loss+= loss.detach()
        # print (batch_loss)


        losses.append(batch_loss)
        klds.append(kld_loss)

    f1 = plt.figure()
    # f2 = plt.figure()

    ax1 = f1.add_subplot(211)
    ax1.plot(np.linspace(0,n_epochs-1,n_epochs), losses)
    ax2 = f1.add_subplot(212)
    ax2.plot(np.linspace(0,n_epochs-1,n_epochs), klds)
  
    plt.savefig("VAE_test_losses_norm.png")

    test_set = random.sample(all_obs, 10)
    test_set = torch.tensor(test_set).unsqueeze(1)
    with torch.no_grad():
        encoded, decoded, mu, log_var = vae(test_set)
        encoded = list(encoded.numpy())
        decoded = list(decoded.numpy())
        test_set = list(test_set.numpy())
        for target,enc, dec in zip(test_set,encoded, decoded):
            print ("Target Vector: ")
            print (target)
            print ("Encoded Vector: ")
            print (enc)
            print ("Decoded Vector: ")
            print (dec)
            print ("\n")


    #============================== Test ON MNIST  ==========================================================================================

    # input_dim = 28 * 28
    # batch_size = 32
    # encoding_dim = 8
    # hidden_size = 64

    # transform = transforms.Compose(
    #     [transforms.ToTensor()])
    # mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

    # dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
    #                                          shuffle=True, num_workers=2)

    # print('Number of samples: ', len(mnist))

    # encoder = Encoder(input_dim, hidden_size, encoding_dim)
    # decoder = Decoder(encoding_dim, hidden_size, input_dim)
    # vae = VAE(encoder, decoder)

    # losses = []
    # klds = []

    # criterion = nn.MSELoss()

    # optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    # l = None
    # for epoch in range(100):
    #     batch_loss = 0
    #     print (epoch)
    #     for i, data in enumerate(dataloader, 0):
    #         inputs, classes = data
    #         inputs, classes = Variable(inputs.resize_(batch_size, input_dim)), Variable(classes)
    #         optimizer.zero_grad()
    #         encoded, decoded, mu, log_var = vae(inputs)
    #         loss, kld_loss = latent_loss(encoded, decoded, mu, log_var, inputs)
    #         loss.backward()
    #         optimizer.step()
    #         # l = loss.data[0]
    #         batch_loss += loss.detach()
    #         kld_loss += kld_loss.detach()

    #     losses.append(batch_loss)
    #     klds.append(kld_loss)

    # f1 = plt.figure()
    # ax1 = f1.add_subplot(211)
    # ax1.plot(np.linspace(0,len(losses)-1, len(losses)), losses)
    
    # ax2 = f1.add_subplot(212)
    # ax2.plot(np.linspace(0,len(klds)-1, len(klds)), klds)
  
    # plt.savefig("VAE_MNIST_losses.png")



