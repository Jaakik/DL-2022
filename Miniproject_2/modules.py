""" Module containing implementations of layers and model architectures """

from torch import empty, zeros
import math
import torch


class Module(object):
    """
    Module class - interface that all other model architecture classes in the framework should inherit
    """

    def __init__(self):
        pass

    def forward(self, *input_):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def update(self, lr):
        pass

    def zero_grad(self):
        pass

    def init_params(self):
        pass

class Sequential(Module):
    """
    Class implementing the sequential deep model architecture
    """

    def __init__(self, *modules):
        """
        Sequential constructor

        :param modules: list of layer modules, list of nn.Module objects
        """

        Module.__init__(self)
        self.modules = list(modules)
    
    def init_params():
        for module in self.modules:
            module.init_params()
        

    def forward(self, *input_):
        """
        Sequential model prediction
        x_0 = input
        x_l+1 = LayerForward(x_l)

        :param input_: train input, torch.Tensor

        :returns: final layer output, torch.Tensor
        """

        x = input_[0].clone()

        for m in self.modules:
            x = m.forward(x)
        

        return x

    def backward(self, *gradwrtoutput):
        """
        Sequential model gradient accumulation
        Grad(x_L) = input
        Grad(x_l-1) = LayerBackward(Grad(x_l))

        :param gradwrtoutput: loss gradient, torch.Tensor
        """

        
        x = gradwrtoutput[0].clone()

        for i, m in enumerate(reversed(self.modules)):
            x = m.backward(x)

    def param(self):
        """
        Retrieve parameters from all layers

        :returns: list of Module.param outputs
        """

        params = []

        for m in self.modules:
            for param in m.param():
                params.append(param)

        return params

    def update(self, lr):
        """
        Perform the gradient descent parameter update for all layers

        :param lr: learning rate, positive float
        """

        for m in self.modules:
            m.update(lr)

    def zero_grad(self):
        """
        Reset the gradients to zero for all layer parameters
        """

        for m in self.modules:
            m.zero_grad()

    def append_layer(self, module):
        """
        Append a new layer at the end of the architecture

        :param module: layer to append, nn.Module object
        """

        self.modules.append(module)



class  Conv2d(Module):
    """
        Applies a 2D convolution over an input signal composed of several input planes. Applies padding = 1 by default. 

        :param stride: controls the stride for the cross-correlation, a single number or a tuple
        :param size: size of filter
        :param n_channels: number of input channels 
        :param num_filters: number of output channels 
    
        """
    def __init__(self, num_filters=5, n_channels = 3, stride=1, size=3):
        super().__init__()
        self.filters = torch.empty((num_filters,size,size))
        self.stride = stride
        self.n_channels = n_channels 
        self.size = size
        self.num_filters = num_filters
        self.dfilt = zeros((num_filters,size,size))
        

    
    def init_params(self, normal = True):
        if normal : 
            self.filters = torch.normal(0, 1, size=(self.num_filters,self.n_channels ,self.size,self.size))

        
    def forward(self, x):
        x = x.squeeze()
        dim = x.size(dim=1)
        self.sz = dim
        self.in_dim = x.size()

        
        padded_shape = dim + 2     # input padded shape , padding = 1 by default 
        
        in_tensor = zeros(self.n_channels,padded_shape,padded_shape)

        in_tensor[0][:dim,:dim] = x[0]    # Input channels being 3 for our dataset
        in_tensor[1][:dim,:dim] = x[1]
        in_tensor[2][:dim,:dim] = x[2]
        


        

        self.input = in_tensor     # keep track of last input for later backward propagation
        

        input_dimension = in_tensor.size(dim=1)                                              # input dimension

        output_dimension = int((input_dimension - self.size) / self.stride) + 1         # output dimension
        

        out = torch.empty((self.num_filters, output_dimension, output_dimension))     # create the matrix to hold the
                                                                                        # values of the convolution

        for f in range(self.num_filters):              # convolve each filter over the image,
            tmp_y = out_y = 0                               # moving it vertically first and then horizontally
            while tmp_y + self.size <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dimension:
                    patch = in_tensor[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    out[f, out_y, out_x] += torch.sum(self.filters[f] * patch)
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
        
        return out

    def backward(self, din):
        input_dimension = self.sz        # input dimension
        
        
        
        dout = torch.zeros(self.num_filters,input_dimension,input_dimension)  # loss gradient of the input to the convolution operation
        
        
        
        dfilt = torch.zeros(self.filters.size())                # loss gradient of filter

        for f in range(self.num_filters):              # loop through all filters
            tmp_y = out_y = 0
            while tmp_y + self.size <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dimension:
                    patch = self.input[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    dfilt[f] += torch.sum(din[f, out_y, out_x] * patch, axis=0)
                    dout[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size] += din[f, out_y, out_x] * self.filters[f]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
        self.dfilt = dfilt
        return dout
    
    def init_params(self, weights):
        self.filters = weights 


    def update(self, lr):

        self.filters -= lr * self.dfilt                  # update filters using GD


    def param(self):
          return [self.filters]


    def zero_grad(self):
        self.dfilt = zeros((self.num_filters,self.size,self.size))

class MAXPOOL2D(Module):
    """
    Applies a 2D max pooling over an input signal composed of several input planes.
    Notice that this class has weights
    
    :param stride: controls the stride for the cross-correlation, a single number or a tuple
    :param size: downsampling factor 

    """    
    def __init__(self, stride=2, size=2):
        self.input = None
        self.stride = stride
        self.size = size
    
    def unravel_index(self,index, shape):
        coord = []

        for dim in reversed(shape):
            coord.append(index % dim)
            index = index // dim

        coord = torch.stack(coord[::-1], dim=-1)

        return coord
    
    def forward(self, x):
        self.input = x                            # keep track of last input for later backward propagation

        num_channels, h_prev, w_prev = x.size()
        h = int((h_prev - self.size) / self.stride) + 1     # compute output dimensions after the max pooling
        w = int((w_prev - self.size) / self.stride) + 1

        downsampled = torch.zeros((num_channels, h, w))        # hold the values of the max pooling

        for i in range(num_channels):                       # slide the window over every part of the image and
            curr_y = out_y = 0                              # take the maximum value at each step
            while curr_y + self.size <= h_prev:             # slide the max pooling window vertically across the image
                curr_x = out_x = 0
                while curr_x + self.size <= w_prev:         # slide the max pooling window horizontally across the image
                    patch = x[i, curr_y:curr_y + self.size, curr_x:curr_x + self.size]
                    downsampled[i, out_y, out_x] = torch.max(patch)       # choose the maximum value within the window
                    curr_x += self.stride                              # at each step and store it to the output matrix
                    out_x += 1
                curr_y += self.stride
                out_y += 1

        return downsampled
    
    
    
    def backward(self, din):
        num_channels, orig_dim, *_ = self.input.shape      # gradients are passed through the indices of greatest
                                                                # value in the original pooling during the forward step

        dout = torch.zeros(self.input.shape)                  # initialize derivative

        for c in range(num_channels):
            tmp_y = out_y = 0
            while tmp_y + self.size <= orig_dim:
                tmp_x = out_x = 0
                while tmp_x + self.size <= orig_dim:
                    patch = self.input[c, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size] # obtain index of largest
                    
                    (x, y) = self.unravel_index(torch.argmax(patch),patch.shape)                     # value in patch
                    dout[c, tmp_y + x, tmp_x + y] += din[c, out_y, out_x]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1

        return dout
        
    def zero_grad(self):                      # pooling layers have no weights
        pass
    
    def update(self, lr):                     # pooling layers have no weights
        pass

    def param(self):                          # pooling layers have no weights
        return []
    
    def init_params(self, weights):
        pass

class MaxPool2d(Module):
    ''' 
    Upsampling layer with nearest mode which simply repeats up to a scale factor the elements around their neighborhood 
    :param scale_factor: upsampling factor 

    '''
    def __init__(self, scale_factor=2):
        self.input = None
        self.scale_factor = scale_factor
    
    def forward(self, x):
        self.input = x     
        num_channels, h, w = x.size()
        upsampled = x[0].repeat_interleave(self.scale_factor, dim=0).repeat_interleave(self.scale_factor, dim=1)
        upsampled = torch.unsqueeze(upsampled, 0)
        return upsampled 
    
    
    def backward(self, din):    
        dout = torch.zeros(self.input.shape) 
        dout[0] = din[0][0::2,0::2]
        return dout 
    
      def param(self):                          # pooling layers have no weights
        return []
    
    def init_params(self, weights):
        pass
        
        
    
        
        
          
    

    
    
                 


    
    
    