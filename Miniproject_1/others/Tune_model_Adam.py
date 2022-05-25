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


    
    
# put the data in tensors
torch.manual_seed(4)

all_data=TensorDataset(train_x,train_y)
train_data, val_data = random_split(all_data, (int(0.8*len(all_data)), int(0.2*len(all_data))))
test_data=TensorDataset(test_x,test_y)


# training function 
def train_per_epoch():
    batch_losses = 0.0
    for i,data in enumerate(train_loader):
        # for debug purpose
        # print("STEP ",i,"------\n")
        model.train()
        x_train, y_train = data
        x_train, y_train = x_train.to(device), y_train.to(device)
        # Gradients are accumulated. So, every time we use the gradients to update the parameters, we need to zero the gradients afterwards.
        optimizer.zero_grad()
        y_output = model(x_train)
        loss = criterion(y_output, y_train) 
        # backward(): Computes the sum of gradients of given tensors with respect to graph leaves.
        loss.backward()
        # step(): updating the parameters using the computed gradients
        optimizer.step()
        
        batch_losses+=loss.item()
        
    return batch_losses


# function to tune our model and save the results
def train_validate_model(EPOCHS,bz, lr, wd, opt,train_loader,val_loader,test_loader,start):
  best_vloss = np.inf
  start = time.time()
  tlosses=[]
  vlosses=[]
  for epoch_number in range(EPOCHS):
      # Make sure gradient tracking is on, and do a pass over the data
      # model.train(True)
      train_loss = train_per_epoch()

      # We don't need gradients on to do reporting
      model.train(False)

      validation_loss = 0.0
      for i, vdata in enumerate(val_loader):
          vinputs, vlabels = vdata
          vinputs, vlabels =  vinputs.to(device), vlabels.to(device)
          voutputs = model(vinputs)
          vloss = criterion(voutputs, vlabels)
          validation_loss += vloss.item()
      
      avg_vloss=validation_loss/len(val_loader)
      avg_tloss = train_loss / (len(train_loader))
      vlosses.append(avg_vloss)
      tlosses.append(avg_tloss)
      print(f'Epoch {epoch_number+1} \t\t Training Loss: {avg_tloss }\t\t Validation Loss: {avg_vloss}')
      # Track best performance, and save the model's state
      if avg_vloss < best_vloss:
          print(f'Validation Loss Decreased({best_vloss:.6f}--->{avg_vloss:.6f}) \t Saving The Model')
          best_vloss = avg_vloss
          model_path = '/content/drive/Shareddrives/DeepLearning/FinalModels/OriginalData/Tuning_model_batch{}_lr{}_wd{}_opt{}'.format(bz, lr, wd, opt)
          torch.save(model.state_dict(), model_path)     
  time_elapsed = time.time() - start
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  mini_batch_size = 100
  model_outputs = []
  test_model = UNet()
  # paths_to_save_results= '/content/drive/Shareddrives/DeepLearning/models/Tuning_model_batch{}_lr{}'.format(bz, lr)
  test_model.load_state_dict(torch.load(model_path))
  # test_model.to(device)
  test_model.eval()
  for b in tqdm(range(0, test_x.size(0), mini_batch_size)):
      # b=b.to(device)
      output = test_model(test_x.narrow(0, b, mini_batch_size))
      model_outputs.append(output.cpu())
  model_outputs = torch.cat(model_outputs, dim=0)

  output_psnr = compute_psnr(model_outputs, test_y, 255.0)
  print(f"[PSNR: {output_psnr:.2f} dB]")
  return vlosses, tlosses, output_psnr



# tuning : 

device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
lr=0.001
model = UNet()
model = model.to(device)
opt="Adam"
BATCH_SIZE_list=[32,64,128]
WD_list=[0.0001,0.001,0.01,0.1,1,5]
for bz in BATCH_SIZE_list:
  for wd in WD_list:
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.99),weight_decay=wd,eps=1e-8)
    # train_step = make_train_step(model, loss_fn, optimizer)
    criterion = nn.MSELoss() 
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    EPOCHS = 20
    device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr, weight_decay=wd, betas=(0.9, 0.99), eps=1e-8)
    # train_step = make_train_step(model, loss_fn, optimizer)
    criterion = nn.MSELoss() 
    train_loader = DataLoader(dataset=train_data, batch_size=bz, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=bz, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=bz, shuffle=True)
    start=time.time()
    v_loss,t_loss,psnr=train_validate_model(EPOCHS=EPOCHS,bz=bz, lr=lr, wd=wd, opt="Adam",train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,start=start)
    records={"config":[],"validation_losses":[],"training_losses":[],"PSNR":[]}
    records["config"].append([opt,lr,bz,wd,EPOCHS])
    records["validation_losses"].append(v_loss)
    records["training_losses"].append(t_loss)
    records["PSNR"].append(psnr)
    name=str("result+opt_"+opt+"_lr_"+str(lr)+"_bz_"+str(bz)+"_wd_"+str(wd))
    with open('/content/drive/Shareddrives/DeepLearning/FinalResults/OriginalData/'+name+'.pickle', 'wb') as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)