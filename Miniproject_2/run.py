# example run of Mini version of Unit model with layers implemented from scratch 

from model import Unet

# Dummy Data  of shape (N,C,H,W)


input_data = torch.empty(444,3,100, 100).uniform_(0, 1)
target_data = torch.empty(444, 5,100, 100).uniform_(0, 1)

# Unet Model
model = Unet()

# Train model 
model.train()

