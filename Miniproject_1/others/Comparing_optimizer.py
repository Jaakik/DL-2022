#import packages

import torch
import math
import torchvision


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
from tqdm import tqdm
import pickle
import numpy as np 
np.random.seed(2013)

# shared path
path= '/content/drive/Shareddrives/DeepLearning/dataset/'

# load data

# !!!!!-------------
# to use the original data: uncomment the following line : 
# noisy_imgs_1 , noisy_imgs_2 = torch.load(path+'train_data.pkl')

# to use the augmented data, uncomment the following two lines : 
# noisy_imgs_1= torch.load(path+'noisy_imgs_1_augmented_torch_vision.pkl')
# noisy_imgs_2= torch.load(path+'noisy_imgs_2_augmented_torch_vision.pkl')

noisy_imgs , clean_imgs = torch.load(path+'val_data.pkl')

# prepare the data
train_x, train_y = noisy_imgs_1 , noisy_imgs_2
test_x, test_y = noisy_imgs , clean_imgs
train_x, train_y = train_x.float(), train_y.float()
test_x, test_y = test_x.float(), test_y.float()


# evaluation metric
def compute_psnr(x, y, max_range=1.0):
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()



# inspired by https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
# we firstly build the same architecture as described in the paper as the base model and to see if we can further optimise it.

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
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
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


    
##------------------------------------------------------------------------------
## SGD

device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
lr=0.000001
model = UNet()
model = model.to(device)
wd=0.01
opt="SGD"
optimizer = optim.SGD(model.parameters(), lr, weight_decay = wd)
bz=128
# train_step = make_train_step(model, loss_fn, optimizer)
criterion = nn.MSELoss() 
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
EPOCHS = 20
device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()
model = model.to(device)
criterion = nn.MSELoss() 
train_loader = DataLoader(dataset=train_data, batch_size=bz, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=bz, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=bz, shuffle=True)
start=time.time()
v_loss,t_loss,psnr=train_validate_model(EPOCHS=EPOCHS,bz=bz, lr=lr, wd=wd, opt=opt,train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,start=start)
records={"config":[],"validation_losses":[],"training_losses":[],"PSNR":[]}
records["config"].append([opt,lr,bz,wd,EPOCHS])
records["validation_losses"].append(v_loss)
records["training_losses"].append(t_loss)
records["PSNR"].append(psnr)
name=str("result+opt_"+opt+"_lr_"+str(lr)+"_bz_"+str(bz)+"_wd_"+str(wd))
with open('/content/drive/Shareddrives/DeepLearning/FinalResults/'+name+'.pickle', 'wb') as handle:
    pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
##------------------------------------------------------------------------------
## RMSprop

device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
lr=0.001
model = UNet()
model = model.to(device)
wd=0.01
opt="RMSProp"
optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
bz=128
criterion = nn.MSELoss() 
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
EPOCHS = 20
device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()
model = model.to(device)
criterion = nn.MSELoss() 
train_loader = DataLoader(dataset=train_data, batch_size=bz, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=bz, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=bz, shuffle=True)
start=time.time()
v_loss,t_loss,psnr=train_validate_model(EPOCHS=EPOCHS,bz=bz, lr=lr, wd=wd, opt=opt,train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,start=start)
records={"config":[],"validation_losses":[],"training_losses":[],"PSNR":[]}
records["config"].append([opt,lr,bz,wd,EPOCHS])
records["validation_losses"].append(v_loss)
records["training_losses"].append(t_loss)
records["PSNR"].append(psnr)
name=str("result+opt_"+opt+"_lr_"+str(lr)+"_bz_"+str(bz)+"_wd_"+str(wd))
with open('/content/drive/Shareddrives/DeepLearning/FinalResults/'+name+'.pickle', 'wb') as handle:
    pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
