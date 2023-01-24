ENCODING_DIMS = 12
STATE_DIMS = 24
print("stripping network to encode to ", ENCODING_DIMS, "dimensional space")

import hashlib
import torch
import torch.nn as nn
from StrippedNet import EmbeddingNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("saved_models/walker_walk_rew_feature_state_dict", map_location=device)

net = EmbeddingNet(STATE_DIMS,ENCODING_DIMS)
sd = net.state_dict()
sd.update({k:v for k,v in model.items() if k in net.state_dict()})

torch.save(sd, "saved_models/walker_walk_pretrained.params")