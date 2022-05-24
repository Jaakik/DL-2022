import re
import sys
import unittest
import importlib
from pathlib import Path

import torch
import math
import torchvision

import numpy as np
from torch import optim
from torch import Tensor
from torch import nn

from torch.autograd import Variable
from torch.nn import Linear, ReLU,LeakyReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Upsample, Module, Softmax, BatchNorm2d, Dropout,ConvTranspose2d, MSELoss,Hardtanh
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split

from datetime import datetime
import time

# inspired by https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/ and https://github.com/joeylitalien/noise2noise-pytorch/blob/master/src/unet.py
# we firstly build the similar architecture as described in the paper as the base model and to see if we can further optimise it.

# model definition
class UNet(nn.Module):
    # define model elements
    def __init__(self):
        super(UNet, self).__init__()

        self.struct1=Sequential(
        #ENC CONV0
        Conv2d(3, 48, kernel_size=3, padding='same'),
        #(W – F + 2P) / S + 1 =(48-3+2)/1+1=48
        LeakyReLU(0.1, inplace=True),
        #ENC CONV1
        Conv2d(48, 48, kernel_size=3, padding=1),
        #(W – F + 2P) / S + 1 =(48-3+2)/1+1=48
        LeakyReLU(0.1, inplace=True),
        #POOL1
        MaxPool2d(2)
        )

        self.struct2=Sequential(
        # ENC CONV2
        Conv2d(48, 48, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),
        # POOL2
        MaxPool2d(2)
        )

        self.struct3=Sequential(
        # ENC CONV3
        Conv2d(48, 48, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),

        # POOL3
        MaxPool2d(2)
        )

        self.struct4=Sequential(
        # ENC CONV4
        Conv2d(48, 48, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),

        # POOL4
        MaxPool2d(2)
        )

        self.struct5=Sequential(

        # ENC CONV5
        Conv2d(48, 48, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),

        # POOL5
        MaxPool2d(2)
        )

        self.struct6=Sequential(
        #ENC CONV6
        Conv2d(48, 48, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),
        #UPSAMPLE5
        ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
        # nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.struct7=Sequential(
        #DEC CONV5A
        Conv2d(96, 96, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),
        #DEC CONV5B
        Conv2d(96, 96, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),
        #UPSAMPLE4
        ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
        # nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.struct8=Sequential(
        #DEC CONV4A
        Conv2d(144, 96, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),

        #DEC CONV4B
        Conv2d(96, 96, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),

        #UPSAMPLE3
        ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
        # nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.struct9=Sequential(

        #DEC CONV3A
        Conv2d(144, 96, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),

        #DEC CONV3B
        Conv2d(96, 96, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),

        #UPSAMPLE2
        ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1)

        # nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.struct10=Sequential(
        #DEC CONV2A
        Conv2d(144, 96, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),

        #DEC CONV2B
        Conv2d(96, 96, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),
        ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1),

        #UPSAMPLE1
        # nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.struct11=Sequential(
        # I think this is not needed: at least not in the github
        # Conv2d(96, 99, kernel_size=3, padding=1),
        # LeakyReLU(0.1, inplace=True),
        Conv2d(99, 64, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),
        Conv2d(64, 32, kernel_size=3, padding=1),
        LeakyReLU(0.1, inplace=True),
        Conv2d(32, 3, kernel_size=3, padding=1),
        Hardtanh(min_val=0, max_val=255.)
        )


        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initializes weights using He et al. (2015) as described in the orginal paper
        # self.modules() returns an iterable to the many layers or “modules” defined in the model class.
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                #Fills the input Tensor with values according to the method described in “Delving deep into rectifiers: Surpassing human-level
                #performance on ImageNet classification” - He, K.et al. (2015), using a normal distribution.
                #Also known as He initialization. No gradient will be recorded for this operation.
                nn.init.kaiming_normal_(m.weight.data)
                #Fills the given tensor with the 0 in-place, and returns it. 
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
          # Encoder
        p1 = self.struct1(x)
        # print("p1: ", p1.size())
        p2 = self.struct2(p1)
        # print("p2: ", p2.size())
        p3 = self.struct3(p2)
        # print("p3: ", p3.size())
        p4 = self.struct4(p3)
        # print("p4: ", p4.size())
        p5 = self.struct5(p4)
        # print("p5: ", p5.size())

        # Decoder

        # print("----------")
        usp5 = self.struct6(p5)
        # print('usp5: ', usp5.size())
        # print('p4: ', p4.size())
        cct5 = torch.cat((usp5, p4), dim=1) # this is where the error comes from !
        # print('cct5: ', cct5.size())

        # print("----------")
        usp4 = self.struct7(cct5)
        # print('usp4: ',usp4.size())
        # print('p3: ', p3.size())
        cct4 = torch.cat((usp4, p3), dim=1)
        # print('cct4: ', cct4.size())

        # print("----------")
        usp3 = self.struct8(cct4)
        # print('usp3: ',usp3.size())
        # print('p2: ', p2.size())
        cct3 = torch.cat((usp3, p2), dim=1)
        # print('cct3: ', cct3.size())

        # print("----------")
        usp2 = self.struct9(cct3)
        # print('usp2: ',usp2.size())
        # print('p1: ', p1.size())
        cct2 = torch.cat((usp2, p1), dim=1)
        # print('cct2: ', cct2.size())

        # print("----------")
        usp1 = self.struct10(cct2)
        cct1 = torch.cat((usp1, x), dim=1)
        return  self.struct11(cct1)


## For mini-project 1
class Model():
  def __init__(self) -> None:
      ## instantiate model + optimizer + loss function + any other stuff you need
      # self.model=UNet()
      self.device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
      self.model=UNet()
      # put model on cuda
      self.model=self.model.to(self.device)
      self.bestpth = './bestmodel.pth'
      # TOBE DELETED: optimizer_name only used for finding best model
      self.optimizer_name="Adam"

      self.optimizer = Adam(self.model.parameters(), 0.001, betas=(0.9, 0.99), eps=1e-8)
      self.criterion = nn.MSELoss()
      self.batch_size = 128

  def load_pretrained_model ( self ) -> None :
      self.model.load_state_dict(torch.load(self.bestpth,map_location=self.device))

  def train (self , train_input , train_target , num_epochs ) -> None :
    #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images
    #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images , which only differs from the input by their noise
    
    #: The following code is adpted from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    all_data=TensorDataset(train_input,train_target)
    # set the seed number for reproducibility
    torch.manual_seed(4)
    # random split into train and validation: using 80% of data to train and 20% to validate
    train_data, val_data = random_split(all_data, (int(0.8*len(all_data)), int(0.2*len(all_data))))
    # use DataLoader to get batches of data
    train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size , shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=self.batch_size , shuffle=True)
    # initialise the best validation loss as infinite
    best_vloss = np.inf
    # initialise the model
    model=self.model

    # POSSIBLE NEED TO BE DELETED: time start to train the model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    start = time.time()
    for epoch_number in range(num_epochs):

        train_loss = 0.0

        # Perform mini-batch training
        for i,data in enumerate(train_loader):
            # for DEBUG
            # print("STEP ",i,"------\n")

            # Make sure gradient tracking is on, and do a pass over the data
            model.train()
            x_train, y_train = data
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
            # Gradients are accumulated. So, every time we use the gradients to update the parameters, we need to zero the gradients afterwards.
            self.optimizer.zero_grad()
            y_output = model(x_train)
            tloss = self.criterion(y_output, y_train)
            # backward(): Computes the sum of gradients of given tensors with respect to graph leaves.
            tloss.backward()
            # step(): updating the parameters using the computed gradients
            self.optimizer.step()
            train_loss+=tloss.item()

        # We don't need gradients on to do reporting
        model.train(False)

        # Perform mini-batch validation
        validation_loss = 0.0
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels =  vinputs.to(self.device), vlabels.to(self.device)
            voutputs = model(vinputs)
            vloss = self.criterion(voutputs, vlabels)

            validation_loss += vloss.item()

        # Computer the average loss for train and validation
        avg_vloss=validation_loss/(len(val_loader))
        avg_tloss = train_loss / (len(train_loader))

        print(f'Epoch {epoch_number+1} \t\t Training Loss: {avg_tloss }\t\t Validation Loss: {avg_vloss}')
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            print(f'Validation Loss Decreased({best_vloss:.6f}--->{avg_vloss:.6f}) \t Saving The Model')
            best_vloss = avg_vloss
            model_path = './model_{}_opt-{}_bat-{}_epo-{}.pth'.format(timestamp,self.optimizer_name, self.batch_size, epoch_number)
            torch.save(model.state_dict(), model_path)


    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

  def predict (self , test_input ) -> torch . Tensor :
      #: test_input : tensor of size (N1 , C, H, W) with values in range 0 -255 that has to be denoised by the trained or the loaded network
      #: returns a tensor of the size (N1 , C, H, W) with values in range 0 -255.
      test_input=test_input.to(self.device)
      
      self.model.eval()
      return self.model(test_input)
