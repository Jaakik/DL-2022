from modules import Conv2d, Sequential, MaxPool2d
from criterion import LossMSE
from opt import SGD
import pickle as pkl 

### For mini - project 2

class Model(object) :
    
    def __init__ ( self )  :
       ## instantiate model + optimizer + loss function + any other stuff you need
       pass

    def load_pretrained_model ( self ) :
      ## This loads the parameters saved in bestmodel .pth into the model
       pass

    def train ( self , train_input , train_target , num_epochs ) :
      #: train˙input : tensor of size (N, C, H, W) containing a noisy version of the images

      #: train˙target : tensor of size (N, C, H, W) containing another noisy version of the same images , which only differs from the input by their noise .
     pass
 
    def predict ( self , test_input ) :
      #: test˙input : tensor of size (N1 , C, H, W) with values in range 0 -255 that has to be denoised by the trained or the loaded network .
      #: returns a tensor of the size (N1 , C, H, W) with values in range 0 -255.
      pass

class Unet(Model)

    def __init__ (self,nb_epochs =50 , mini_batch_size = 1, , lr = 0.0008 criterion = LossMSE())  :
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.mini_batch_size = mini_batch_size
        self.model = Sequential(Conv2d(5,3,1,3), ReLU(), Conv2d(5,3,1,3), ReLU(),MaxPooling2d(),Upsampling(),Conv2d(5,3,1,3),ReLU(), Conv2d(5,3,1,3), ReLU())
        self.optimizer = SGD(model=self.model, nb_epochs=nb_epochs, mini_batch_size=mini_batch_size,
                             lr=lr, criterion=criterion)
    
    
    def train (self , train_input , train_target) :
        
        for e in range(self.nb_epochs):
            sum_loss = 0.

            for b in range(0, train_input.size(0), self.mini_batch_size):
                self.model.zero_grad()

                output = self.model.forward(train_input.narrow(0, b, self.mini_batch_size))
                loss = self.criterion.forward(output, train_target.narrow(0, b, self.mini_batch_size))

                sum_loss += loss

                l_grad = self.criterion.backward()
                l_grad = l_grad.squeeze()
                self.model.backward(l_grad)
                self.optimizer.step()

            if verbose:
                print("{} iteration: loss={}".format(e, sum_loss))
                
        return self.model

    
    def predict ( self , test_input ) :
        
        output = self.model.forward(test_input)
        return output

        
    
    
    def save_model(save_path):
        
        with open(save_path, 'wb') as handle:
            pickle.dump(self.model.param(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_pretrained_model (model_path, model) :
        
        with open(model_path, 'rb') as handle:
            weights = pickle.load(handle)
        
        for params, model in enumerate(weights, self.model.modules) : 
            model.init_params(params)
        
        return model 
            
    
    
        
        
        
        
    
    
        
        


        
    
    
        
        
    
