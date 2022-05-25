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

# saving the model
torch.save(noisy_imgs_1_augmented, '/content/drive/Shareddrives/DeepLearning/dataset/noisy_imgs_1_augmented_torch_vision.pkl')
torch.save(noisy_imgs_2_augmented, '/content/drive/Shareddrives/DeepLearning/dataset/noisy_imgs_2_augmented_torch_vision.pkl')



## below is the code we used before checking with the TA, we were using the package skimage before moving to pytorch
## we put the code here commented: until doing the transformation, the code is the same:

# Data Augmentation

# #needed packages 
# import skimage.io as io
# from skimage.transform import rotate, AffineTransform, warp
# from skimage.util import random_noise
# from skimage.filters import gaussian
# import random

# indice = random.sample(range(50000), 200)
# indice = torch.tensor(indice)

# noisy_imgs_1_sampled=noisy_imgs_1[indice]
# noisy_imgs_2_sampled=noisy_imgs_2[indice]
# loading images

# noisy_imgs_1_augmented , noisy_imgs_2_augmented= noisy_imgs_1.float(), noisy_imgs_2.float()



# # Augmenting

# for i in range(200):
#     image_1= noisy_imgs_1_sampled[i]
#     image_2= noisy_imgs_2_sampled[i]

#     # doing rotations 45deg
#     rotated_1 = rotate(image_1, angle=45, mode = 'wrap')
#     rotated_2 = rotate(image_2, angle=45, mode = 'wrap')

#     noisy_imgs_1_augmented=torch.concat((noisy_imgs_1_augmented,torch.from_numpy(rotated_1).unsqueeze(0)))
#     noisy_imgs_2_augmented=torch.concat((noisy_imgs_2_augmented,torch.from_numpy(rotated_2).unsqueeze(0)))

#     # shifted Images 45deg
#     transform = AffineTransform(translation=(45,45))

#     wrapShift_1 = warp(image_1,transform,mode='wrap')
#     wrapShift_2 = warp(image_2,transform,mode='wrap')

#     noisy_imgs_1_augmented=torch.concat((noisy_imgs_1_augmented,torch.from_numpy(wrapShift_1).unsqueeze(0)))
#     noisy_imgs_2_augmented=torch.concat((noisy_imgs_2_augmented,torch.from_numpy(wrapShift_2).unsqueeze(0)))

#     #flip the image: left to right
#     flipped_lr_1= np.fliplr(image_1)
#     flipped_lr_2= np.fliplr(image_2)

#     noisy_imgs_1_augmented=torch.concat((noisy_imgs_1_augmented,torch.from_numpy(flipped_lr_1.copy()).unsqueeze(0)))
#     noisy_imgs_2_augmented=torch.concat((noisy_imgs_2_augmented,torch.from_numpy(flipped_lr_2.copy()).unsqueeze(0)))

#     #flip the image: up to down  
#     flipped_ud_1= np.flipud(image_1)
#     flipped_ud_2= np.flipud(image_2)

#     noisy_imgs_1_augmented=torch.concat((noisy_imgs_1_augmented,torch.from_numpy(flipped_ud_1.copy()).unsqueeze(0)))
#     noisy_imgs_2_augmented=torch.concat((noisy_imgs_2_augmented,torch.from_numpy(flipped_ud_2.copy()).unsqueeze(0)))

#     # adding the noise but to only one picture
#     sigma=0.155
#     image_1_noisy_Random = random_noise(image_1,var=sigma**2)

#     noisy_imgs_1_augmented=torch.concat((noisy_imgs_1_augmented,torch.from_numpy(image_1_noisy_Random).unsqueeze(0)))
#     noisy_imgs_2_augmented=torch.concat((noisy_imgs_2_augmented,image_2.unsqueeze(0)))

#     #blurring
#     blurred_1 = gaussian(image_1,sigma=1,multichannel=True)
#     blurred_2 = gaussian(image_2,sigma=1,multichannel=True)

#     noisy_imgs_1_augmented=torch.concat((noisy_imgs_1_augmented,torch.from_numpy(blurred_1).unsqueeze(0)))
#     noisy_imgs_2_augmented=torch.concat((noisy_imgs_2_augmented,torch.from_numpy(blurred_2).unsqueeze(0)))

# # noisy_imgs_1_augmented.shape, noisy_imgs_2_augmented.shape

# torch.save(noisy_imgs_1_augmented, '/content/drive/Shareddrives/DeepLearning/dataset/noisy_imgs_1_augmented.pkl')
# torch.save(noisy_imgs_2_augmented, '/content/drive/Shareddrives/DeepLearning/dataset/noisy_imgs_2_augmented.pkl')