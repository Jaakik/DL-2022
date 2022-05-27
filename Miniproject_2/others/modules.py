""" Module containing implementations of layers and model architectures """

from torch import empty, zeros
import math
import torch
import torch.nn as nn


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

    def init_params(self,weights):
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
        :param kernel_size: size of filter
        :param in_channel: number of input channels 
        :param out_channel: number of output channels 
        :param padding: padding mode 
        
    
        """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True):
        super(Conv2d, self).__init__()
        self.k = kernel_size
        self.in_c = in_channel
        self.out_c = out_channel
        self.stride = stride
        self.padding = padding
        self.conv = torch.empty(out_channel, in_channel, kernel_size, kernel_size).uniform_(-1/10, 1/10)
        

    
    
    def forward(self, inp):
        
        self.input = inp # for backpropagation
        
        k = self.k
        stride = self.stride
        h_in  = inp.size(2)
        w_in  = inp.size(2)

        padding = self.padding  # + k//2
        batch_size = inp.shape[0]

        h_out = (h_in + 2 * padding - (k - 1) - 1) / stride + 1
        w_out = (w_in + 2 * padding - (k - 1) - 1) / stride + 1
        h_out, w_out = int(h_out), int(w_out)

        inp_unf = torch.nn.functional.unfold(inp, (k, k), padding=padding)
        out_unf = inp_unf.transpose(1, 2).matmul(self.conv.view(self.conv.size(0), -1).t()).transpose(1, 2)
        out_ = torch.nn.functional.fold(out_unf, (h_out, w_out), (1, 1))
        
        return out_

    def backward(self, din):
        input_dimension = din.size(-1)
        
        dout = torch.zeros(self.out_c,input_dimension,input_dimension)  # loss gradient of the input to the convolution operation
        
        dfilt = torch.zeros(self.conv.size())                # loss gradient of filter
        
        for f in range(self.out_c):              # loop through all filters
            tmp_y = out_y = 0
            while tmp_y + self.k <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.k <= input_dimension:
                    patch = self.input[0][:, tmp_y:tmp_y + self.k, tmp_x:tmp_x + self.k]
                    dfilt[f] += torch.sum(din[0][f, out_y, out_x] * patch, axis=0)
                    dout[:, tmp_y:tmp_y + self.k, tmp_x:tmp_x + self.k] += din[0][f, out_y, out_x] * self.conv[f]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
        self.dfilt = dfilt
        return dout.unsqueeze(0)
    
    def init_params(self, weights):
        self.conv = weights 


    def update(self, lr):

        self.conv -= lr * self.dfilt               # update filters using GD


    def param(self):
          return self.conv


    def zero_grad(self):
        self.conv = torch.zeros(self.out_c, self.in_c, self.k, self.k)

class MaxPooling2D(Module):
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
        
        print(x.size())

        _,num_channels, h_prev, w_prev = x.size()
        h = int((h_prev - self.size) / self.stride) + 1     # compute output dimensions after the max pooling
        w = int((w_prev - self.size) / self.stride) + 1

        downsampled = torch.zeros((num_channels, h, w))        # hold the values of the max pooling

        for i in range(num_channels):                       # slide the window over every part of the image and
            curr_y = out_y = 0                              # take the maximum value at each step
            while curr_y + self.size <= h_prev:             # slide the max pooling window vertically across the image
                curr_x = out_x = 0
                while curr_x + self.size <= w_prev:         # slide the max pooling window horizontally across the image
                    patch = x[0][i, curr_y:curr_y + self.size, curr_x:curr_x + self.size]
                    downsampled[i, out_y, out_x] = torch.max(patch)       # choose the maximum value within the window
                    curr_x += self.stride                              # at each step and store it to the output matrix
                    out_x += 1
                curr_y += self.stride
                out_y += 1

        return downsampled.unsqueeze(0)
    
    
    
    def backward(self, din):
        print(self.input.shape)
        _,num_channels, orig_dim, *_ = self.input.shape      # gradients are passed through the indices of greatest
                                                                # value in the original pooling during the forward step

        dout = torch.zeros(self.input.shape)                  # initialize derivative

        for c in range(num_channels):
            tmp_y = out_y = 0
            while tmp_y + self.size <= orig_dim:
                tmp_x = out_x = 0
                while tmp_x + self.size <= orig_dim:
                    patch = self.input[0][c, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size] # obtain index of largest
                    
                    (x, y) = self.unravel_index(torch.argmax(patch),patch.shape)                     # value in patch
                    dout[0][c, tmp_y + x, tmp_x + y] += din[0][c, out_y, out_x]
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

class Upsampling(Module):
    ''' 
    Upsampling layer with nearest mode which simply repeats up to a scale factor the elements around their neighborhood 
    :param scale_factor: upsampling factor 

    '''
    def __init__(self, scale_factor=2):
        self.input = None
        self.scale_factor = scale_factor
    
    def forward(self, x):
        self.input = x     
        _, num_channels, h, w = x.size()
        upsampled = torch.zeros((num_channels, h *self.scale_factor, w*self.scale_factor)) 
        upsampled[0] = x[0][0].repeat_interleave(self.scale_factor, dim=0).repeat_interleave(self.scale_factor, dim=1)
        upsampled[1] = x[0][1].repeat_interleave(self.scale_factor, dim=0).repeat_interleave(self.scale_factor, dim=1)
        upsampled[2] = x[0][2].repeat_interleave(self.scale_factor, dim=0).repeat_interleave(self.scale_factor, dim=1)


        return upsampled.unsqueeze(0)
    
    
    def backward(self, din):    
        dout = torch.zeros(self.input.shape) 
        dout[0][0] = din[0][0][0::2,0::2]
        dout[0][1] = din[0][1][0::2,0::2]
        dout[0][2] = din[0][2][0::2,0::2]


        return dout 
    
    def param(self):                          # pooling layers have no weights
        return []
    
    def init_params(self, weights):
        pass
        
        
    
        
        
          
    

    
    
                 


    
    
    