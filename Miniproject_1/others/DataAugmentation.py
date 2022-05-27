## loading packages

import torch
import math
import torchvision


from torch import optim
from torch import Tensor
from torch import nn

from torch.autograd import Variable
from torch.nn import Linear, ReLU,LeakyReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Upsample, Module, Softmax, BatchNorm2d, Dropout,ConvTranspose2d, MSELoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split 

import random # to choose randomnly the sample

import numpy as np 
np.random.seed(2013)


# path where we load the new data
path= '/content/drive/Shareddrives/DeepLearning/dataset/'

# load data
noisy_imgs_1 , noisy_imgs_2 = torch.load(path+'train_data.pkl')
noisy_imgs , clean_imgs = torch.load(path+'val_data.pkl')
train_x, train_y = noisy_imgs_1 , noisy_imgs_2
test_x, test_y = noisy_imgs , clean_imgs

# transformig into float to be ready to train
train_x, train_y = train_x.float(), train_y.float()
test_x, test_y = test_x.float(), test_y.float()

# define the function that transforms the data
transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomInvert(0.5),
    torchvision.transforms.CenterCrop(16),
    torchvision.transforms.RandomGrayscale(0.5),
    torchvision.transforms.Resize(32),
    torchvision.transforms.RandomAdjustSharpness(0.5)
])

# loading images : here it's the variable where we will load the augmented data, we add the initial data here so we can just append the results
noisy_imgs_1_augmented , noisy_imgs_2_augmented= noisy_imgs_1.float(), noisy_imgs_2.float()
noisy_imgs_1_augmented.shape, noisy_imgs_2_augmented.shape


# picking the random sample

indice = random.sample(range(50000), 10000)
indice = torch.tensor(indice)

noisy_imgs_1_sampled=noisy_imgs_1[indice]
noisy_imgs_2_sampled=noisy_imgs_2[indice]



# doing the transformation

noisy_imgs_1_augmented_results,noisy_imgs_2_augmented_results= transforms(noisy_imgs_1_sampled), transforms(noisy_imgs_2_sampled)

noisy_imgs_1_augmented=torch.concat((noisy_imgs_1_augmented,noisy_imgs_1_augmented_results))
noisy_imgs_2_augmented=torch.concat((noisy_imgs_2_augmented,noisy_imgs_2_augmented_results))


# checking the size
noisy_imgs_1_augmented.shape, noisy_imgs_2_augmented.shape

# saving the images
torch.save(noisy_imgs_1_augmented, '/content/drive/Shareddrives/DeepLearning/dataset/noisy_imgs_1_augmented_torch_vision.pkl')
torch.save(noisy_imgs_2_augmented, '/content/drive/Shareddrives/DeepLearning/dataset/noisy_imgs_2_augmented_torch_vision.pkl')


