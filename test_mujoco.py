import torch
import torch.nn as nn
from StrippedNet import EmbeddingNet

device = "cuda:0"
reward_net = EmbeddingNet(24,12)
reward_net.load_state_dict(torch.load("saved_models/walker_walk_pretrained.params", map_location=device))
num_features = reward_net.fc2.in_features

print("reward is linear combination of ", num_features, "features")
reward_net.fc2 = nn.Linear(num_features, 1, bias=False) #last layer just outputs the scalar reward = w^T \phi(s)
reward_net.to(device)
#freeze all weights so there are no gradients (we'll manually update the last layer via proposals so no grads required)
for name, param in reward_net.named_parameters():
    if name != "fc2.weight":
        param.requires_grad = False
    