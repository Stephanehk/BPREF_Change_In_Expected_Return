#----------------------------------------------------------------------------------------
# Long imports
import torch
import torch.nn as nn
import torch.nn.functional as F
#----------------------------------------------------------------------------------------


class EmbeddingNet(nn.Module):
    def __init__(self, STATE_DIMS, ENCODING_DIMS):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ENCODING_DIMS = ENCODING_DIMS

        intermediate_dimension = 32
        self.fc1 = nn.Linear(STATE_DIMS, intermediate_dimension)
        self.fc_mu = nn.Linear(intermediate_dimension, ENCODING_DIMS)
        self.fc2 = nn.Linear(ENCODING_DIMS, 1)
        self.sigmoid = nn.Sigmoid()


    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''

        sum_rewards = 0
        sum_abs_rewards = 0
        x = traj
    
        x = self.sigmoid(x)
        x = F.leaky_relu(self.fc1(x))
    
        mu = self.fc_mu(x)
       
        r = self.fc2(mu)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards

    def get_rewards(self,traj):
        sum_rewards = 0
        sum_abs_rewards = 0
        x = traj
    
        x = self.sigmoid(x)
        x = F.leaky_relu(self.fc1(x))
    
        mu = self.fc_mu(x)
       
        r = self.fc2(mu)
        return r

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i, mu1 = self.cum_return(traj_i)
        cum_r_j, abs_r_j, mu2 = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j, mu1, mu2


    def state_features(self, traj):

        with torch.no_grad():
            accum = torch.zeros(1,self.ENCODING_DIMS).float().to(self.device)
            for x in traj:
                x = self.sigmoid(x)
                x = F.leaky_relu(self.fc1(x))
                mu = self.fc_mu(x)
                accum.add_(mu)
                #print(accum)
        return accum

    def state_feature(self, obs):
        with torch.no_grad():
            x = obs
            x = self.sigmoid(x)
            x = F.leaky_relu(self.fc1(x))
            mu = self.fc_mu(x)
        return mu


### Usage example
#net = Net()
#net.load_state_dict(torch.load(model_path))