# example run of Mini version of Unit model with layers implemented from scratch 

from model import Model
from torch import empty, zeros , nn 
import math
import torch


# Project requirements 
torch.set_grad_enabled(False)


# Dummy Data  of shape (N,C,H,W)
input_data = torch.empty(444,3,32, 32).uniform_(0, 1)
target_data = torch.empty(444, 3,32, 32).uniform_(0, 1)

# Unet Model
model = Model()

# Train model 
#model.train(input_data, target_data)
cnn = nn.Conv2d(3, 3, 3, stride=1)
# Test model with dummy input 
input_test = torch.empty(1,3,32, 32).uniform_(0, 255)
output_m = model.predict(input_test)
output_p = cnn.forward(input_test)
print(output_m)





