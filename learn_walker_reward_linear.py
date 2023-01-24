
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


from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_dmcontrol_env, make_vec_metaworld_env
from stable_baselines3 import PPO_CUSTOM

def generate_demos(env, episode_count=2):

    #TODO: add more episodes!!! ALSO ADD PPO AGENT TRAINED FROM DIFFERENT CHECKPOINTS TO CREATE MORE DIVERSE DEMONSTRATIONS
    demonstrations = []
    learning_returns = []
    learning_rewards = []

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
                traj.append(ob)
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
    return demonstrations, learning_returns, learning_rewards


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

class Net(nn.Module):
    def __init__(self, ENCODING_DIMS, ACTION_DIMS, STATE_DIMS):
        super().__init__()

        # self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        # self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        # self.conv3 = nn.Conv2d(32, 32, 3, stride=1)
        # self.conv4 = nn.Conv2d(32, 16, 3, stride=1)

        intermediate_dimension = 32
        #min(STATE_DIMS, max(12, ENCODING_DIMS*2))
        # intermediate_dimension = 12
        # self.inital_layer = nn.Linear(STATE_DIMS, intermediate_dimension)
        self.fc1 = nn.Linear(STATE_DIMS, intermediate_dimension)
        self.fc_mu = nn.Linear(intermediate_dimension, ENCODING_DIMS)
        self.fc_var = nn.Linear(intermediate_dimension, ENCODING_DIMS)
        self.fc2 = nn.Linear(ENCODING_DIMS, 1)
        self.reconstruct1 = nn.Linear(ENCODING_DIMS, intermediate_dimension)
        self.reconstruct2 = nn.Linear(intermediate_dimension, STATE_DIMS) 

        # self.reconstruct_conv1 = nn.ConvTranspose2d(2, 4, 3, stride=1)
        # self.reconstruct_conv2 = nn.ConvTranspose2d(4, 16, 6, stride=1)
        # self.reconstruct_conv3 = nn.ConvTranspose2d(16, 16, 7, stride=2)
        # self.reconstruct_conv4 = nn.ConvTranspose2d(16, 4, 10, stride=1)

        #QUESTION: Will temporal difference classification work with our state space instead of images?
        #From paper:
        # This task requires an understanding of how visual features move
        # and transform over time, thus encouraging an embedding that learns meaningful abstractions of
        # environment dynamics conditioned on agent interactions
        self.temporal_difference1 = nn.Linear(ENCODING_DIMS*2, 1, bias=False)#ENCODING_DIMS)
        #self.temporal_difference2 = nn.Linear(ENCODING_DIMS, 1)
        self.inverse_dynamics1 = nn.Linear(ENCODING_DIMS*2, ACTION_DIMS, bias=False) #ENCODING_DIMS)
        #self.inverse_dynamics2 = nn.Linear(ENCODING_DIMS, ACTION_SPACE_SIZE)
        self.forward_dynamics1 = nn.Linear(ENCODING_DIMS + ACTION_DIMS, ENCODING_DIMS, bias=False)# (ENCODING_DIMS + ACTION_SPACE_SIZE) * 2)
        self.tanh = nn.Tanh()
        #self.forward_dynamics2 = nn.Linear((ENCODING_DIMS + ACTION_SPACE_SIZE) * 2, (ENCODING_DIMS + ACTION_SPACE_SIZE) * 2)
        #self.forward_dynamics3 = nn.Linear((ENCODING_DIMS + ACTION_SPACE_SIZE) * 2, ENCODING_DIMS)
        self.normal = tdist.Normal(0, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        print("Intermediate dimension calculated to be: " + str(intermediate_dimension))

    def reparameterize(self, mu, var): #var is actually the log variance
        if self.training:
            device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
            # device = "cpu"
            std = var.mul(0.5).exp()
            eps = torch.randn_like(std).to(device)
            #self.normal.sample(mu.shape).to(device)
            return eps.mul(std).add(mu)
        else:
            assert False
            return mu


    def cum_return(self, traj):
        #print("input shape of trajectory:")
        #print(traj.shape)
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        x = traj
    
        x = self.sigmoid(x)
        x = F.leaky_relu(self.fc1(x))
    
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        z = self.reparameterize(mu, var)

        r = self.fc2(z)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards, mu, var, z

    def estimate_temporal_difference(self, z1, z2):
        #QUESTION: Not sure if what I did here with changed the cat dimension is correct
        pair = torch.cat((z1, z2), 2)
        
        x = self.temporal_difference1(pair)
        #x = self.temporal_difference2(x)
        return x

    def forward_dynamics(self, z1, actions):
        
        z1 = torch.squeeze(z1)
    
        x = torch.cat((z1.float(), actions.float()), dim=1)
        x = self.forward_dynamics1(x)
        #x = F.leaky_relu(self.forward_dynamics2(x))
        #x = self.forward_dynamics3(x)
        return x

    def estimate_inverse_dynamics(self, z1, z2):
        concatenation = torch.cat((z1, z2), 2)
        x = self.tanh(self.inverse_dynamics1(concatenation))
        #x = F.leaky_relu(self.inverse_dynamics2(x))
        return x

    def decode(self, encoding):
        x = F.leaky_relu(self.reconstruct1(encoding))
        x = F.leaky_relu(self.reconstruct2(x))
        return x

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i, mu1, var1, z1 = self.cum_return(traj_i)
        cum_r_j, abs_r_j, mu2, var2, z2 = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j, z1, z2, mu1, mu2, var1, var2



def reconstruction_loss(decoded, target, mu, logvar):
    num_elements = decoded.numel()
    target_num_elements = target.numel()
    if num_elements != target_num_elements:
        print("ELEMENT SIZE MISMATCH IN RECONSTRUCTION")
        sys.exit()
        
    # bce = F.binary_cross_entropy(decoded, target)
    target = torch.sigmoid(target)
    bce = F.mse_loss(decoded, target)

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    kld /= num_elements
    # print("bce: " + str(bce) + " kld: " + str(kld))
    return bce + kld, bce, kld

# Train the network
def learn_reward(reward_network, optimizer, training_inputs, training_outputs, training_times, training_actions, num_iter, l1_reg, checkpoint_dir, loss_fn):
    #check if gpu available
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()
    temporal_difference_loss = nn.MSELoss()
    # inverse_dynamics_loss = nn.CrossEntropyLoss()
    inverse_dynamics_loss = nn.MSELoss()

    forward_dynamics_loss = nn.MSELoss()
    last_losses = []

    cum_loss = 0.0
    training_data = list(zip(training_inputs, training_outputs, training_times, training_actions))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels, training_times_sub, training_actions_sub = zip(*training_data)
        validation_split = 1.0
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            times_i, times_j = training_times_sub[i]
            actions_i, actions_j = training_actions_sub[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)
            num_frames_i = len(traj_i)
            num_frames_j = len(traj_j)

            #zero out gradient
            optimizer.zero_grad()

            #==============================================================================================================================
            #TODO: pretty sure something is wrong here
            #forward + backward + optimize
            outputs, abs_rewards, z1, z2, mu1, mu2, logvar1, logvar2 = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)

            decoded1 = reward_network.decode(z1)
            decoded2 = reward_network.decode(z2)

            # # print (decoded1)
            # # print (traj_i)
            # # print ("\n")

            reconstruction_loss_1, bce1, kld1 = reconstruction_loss(decoded1.float(), traj_i.float(), mu1.float(), logvar1.float())
            reconstruction_loss_2, bce2, kld2 = reconstruction_loss(decoded2.float(), traj_j.float(), mu2.float(), logvar2.float())

            reconstruction_loss_1*=10
            reconstruction_loss_2*=10
            # *************************************************************************************************************************************
            # ==============================================================================================================================
            t1_i = np.random.randint(0, len(times_i))
            t2_i = np.random.randint(0, len(times_i))
            t1_j = np.random.randint(0, len(times_j))
            t2_j = np.random.randint(0, len(times_j))

            est_dt_i = reward_network.estimate_temporal_difference(mu1[t1_i].unsqueeze(0), mu1[t2_i].unsqueeze(0))
            est_dt_j = reward_network.estimate_temporal_difference(mu2[t1_j].unsqueeze(0), mu2[t2_j].unsqueeze(0))
            real_dt_i = (times_i[t2_i] - times_i[t1_i])/900.0
            real_dt_j = (times_j[t2_j] - times_j[t1_j])/900.0

            actions_1 = reward_network.estimate_inverse_dynamics(mu1[0:-1], mu1[1:]).squeeze()
            actions_2 = reward_network.estimate_inverse_dynamics(mu2[0:-1], mu2[1:]).squeeze()
            target_actions_1 = torch.LongTensor(actions_i[1:]).to(device)
            target_actions_2 = torch.LongTensor(actions_j[1:]).to(device)
            #print((actions_1, target_actions_1))
            #print((actions_2, target_actions_2))

            # print (actions_1.shape)#pretty sure this is right
            # print (target_actions_1.shape)#pretty sure this is not right
            
            # target_actions_1 = target_actions_1.unsqueeze(1)

            inverse_dynamics_loss_1 = inverse_dynamics_loss(actions_1.float(), target_actions_1.float())/1.9
            inverse_dynamics_loss_2 = inverse_dynamics_loss(actions_2.float(), target_actions_2.float())/1.9

            #===============================================================================================================================================================================

            forward_dynamics_distance = 5 #1 if epoch <= 1 else np.random.randint(1, min(1, max(epoch, 4)))
            forward_dynamics_actions1 = target_actions_1
            forward_dynamics_actions2 = target_actions_2
            # forward_dynamics_onehot_actions_1 = torch.zeros((num_frames_i-1, ACTION_DIMS), dtype=torch.float32, device=device)
            # forward_dynamics_onehot_actions_2 = torch.zeros((num_frames_j-1, ACTION_DIMS), dtype=torch.float32, device=device)
    

            # forward_dynamics_onehot_actions_1.scatter_(1, forward_dynamics_actions1.unsqueeze(1), 1.0)
            # forward_dynamics_onehot_actions_2.scatter_(1, forward_dynamics_actions2.unsqueeze(1), 1.0)
        
            forward_dynamics_1 = reward_network.forward_dynamics(mu1[:-forward_dynamics_distance].float(), forward_dynamics_actions1[:(num_frames_i-forward_dynamics_distance)].float())
            forward_dynamics_2 = reward_network.forward_dynamics(mu2[:-forward_dynamics_distance].float(), forward_dynamics_actions2[:(num_frames_j-forward_dynamics_distance)].float())
            for fd_i in range(forward_dynamics_distance-1):
                forward_dynamics_1 = reward_network.forward_dynamics(forward_dynamics_1.float(), forward_dynamics_actions1[fd_i+1:(num_frames_i-forward_dynamics_distance+fd_i+1)].float())
                forward_dynamics_2 = reward_network.forward_dynamics(forward_dynamics_2.float(), forward_dynamics_actions2[fd_i+1:(num_frames_j-forward_dynamics_distance+fd_i+1)].float())

            forward_dynamics_loss_1 = 100 * forward_dynamics_loss(forward_dynamics_1.float(), torch.squeeze(mu1[forward_dynamics_distance:]).float())
            forward_dynamics_loss_2 = 100 * forward_dynamics_loss(forward_dynamics_2.float(), torch.squeeze(mu2[forward_dynamics_distance:]).float())

            #===============================================================================================================================================================================
            #TODO: Not sure if this makes sense in our domain
           
            # print("est_dt: " + str(est_dt_i) + ", real_dt: " + str(real_dt_i))
            # print("est_dt: " + str(est_dt_j) + ", real_dt: " + str(real_dt_j))

            dt_loss_i = 4*temporal_difference_loss(torch.squeeze(est_dt_i).float(), torch.tensor(((real_dt_i,),), dtype=torch.float32, device=device).squeeze())
            dt_loss_j = 4*temporal_difference_loss(torch.squeeze(est_dt_j).float(), torch.tensor(((real_dt_j,),), dtype=torch.float32, device=device).squeeze())
            #===============================================================================================================================================================================
            #I think something is wrong here
            trex_loss = loss_criterion(outputs, labels)
            #===============================================================================================================================================================================

            # # print (dt_loss_i)
            # # print (inverse_dynamics_loss_1)
            # # print (forward_dynamics_loss_1)
            # # print (reconstruction_loss_1)
            # # print (trex_loss)
            # # print ("\n")

            if loss_fn == "trex": #only use trex loss
                loss = trex_loss
            elif loss_fn == "ss": #only use self-supervised loss
                loss = dt_loss_i + dt_loss_j + (inverse_dynamics_loss_1 + inverse_dynamics_loss_2) + forward_dynamics_loss_1 + forward_dynamics_loss_2 + reconstruction_loss_1 + reconstruction_loss_2
            elif loss_fn == "trex+ss":
                loss = dt_loss_i + dt_loss_j + (inverse_dynamics_loss_1 + inverse_dynamics_loss_2) + forward_dynamics_loss_1 + forward_dynamics_loss_2 + reconstruction_loss_1 + reconstruction_loss_2 + trex_loss
                # loss = reconstruction_loss_1 + reconstruction_loss_2


            # last_losses.append([reconstruction_loss_1, reconstruction_loss_2, bce1, bce2, kld1, kld2])
            last_losses.append([dt_loss_i + dt_loss_j, inverse_dynamics_loss_1 + inverse_dynamics_loss_2, forward_dynamics_loss_1 + forward_dynamics_loss_2,reconstruction_loss_1 + reconstruction_loss_2, kld1 + kld2, trex_loss])

            #*************************************************************************************************************************************

            if i < len(training_labels) * validation_split:
                loss.backward()
                optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            #print("total", item_loss)
            cum_loss += item_loss

            if i % 500 == 499:
                #print(i)
                print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
                print(abs_rewards)
                cum_loss = 0.0
                print("check pointing")
                torch.save(reward_net.state_dict(), checkpoint_dir)


    print("finished training")
    np.save("last_losses_autoencoder_components_playing.npy",last_losses)

def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    loss_criterion = nn.CrossEntropyLoss()
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return, z1, z2, _, _, _, _ = reward_network.forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs,0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)

def predict_reward_sequence(net, traj):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            rewards_from_obs.append(r)
    return rewards_from_obs

def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))


#python learn_walker_reward_linear.py --env=walker_walk --seed=1234
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
    demonstrations, learning_returns, learning_rewards = generate_demos(env)


    #sort the demonstrations according to ground truth reward to simulate ranked demos

    #QUESTION: where do these max and min snippet lengths come from?
    #Why does Enduro use full trajectories?
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


    # # Now we create a reward network and optimize it using the training data.

    # #QUESTION: Any suggestion for choosing this parameter?
    encoding_dims = 100
    lr = 0.0001
    weight_decay = 0.001
    num_iter = 2
    l1_reg=0.0
    stochastic = True
    loss_fn = "trex+ss"
    reward_model_path = "saved_models/walker_walk_rew_feature.params"

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    reward_net = Net(encoding_dims, ACTION_DIMS, STATE_DIMS)
    reward_net.to(device)
    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    
    
    
    learn_reward(reward_net, optimizer, training_obs, training_labels, training_times, training_actions, num_iter, l1_reg, reward_model_path, loss_fn)
    #save reward network
    torch.save(reward_net.state_dict(), "saved_models/walker_walk_rew_feature_state_dict_" + str(encoding_dims))

    #print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj[0]) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    print("accuracy", calc_accuracy(reward_net, training_obs, training_labels))

    
