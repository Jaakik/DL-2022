import torch
import math

torch.set_grad_enabled(False)


class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        return


class Linear(Module):
    def __init__(self, input_layer_size, output_layer_size, paramInit = "Normal"):
        super().__init__()
        # Weights "w" is a 2d tensor [input_layer_size, output_layer_size]
        # which is the transpose of what might seem "logical"
        # Thanks to broadcasting "w" is going to increase to a 3d tensor when we receive a batch of inputs
        # Bias "b" is a 1d tensor [output_layer_size]
    
        var = 1 # normal by default
        self.w = torch.empty(input_layer_size, output_layer_size).normal_(0, var) 
        
        # Weight initialization 
        if paramInit == "He": 
            var = math.sqrt(2/(input_layer_size))
        if paramInit == "Xavier":
            var =  math.sqrt(2/(input_layer_size + output_layer_size))
            
        self.w = torch.empty(input_layer_size, output_layer_size).normal_(0, var)
        
        self.b = torch.empty(output_layer_size).normal_(0, var)
        # Gradient vector is just empty for now
        # Each channel represents one of the inputs we receive in the batch
        # And within each channel, each entry represents "how much" the weight should change according to that x
        self.grad_w = torch.empty(self.w.size()).fill_(0)
        self.grad_b = torch.empty(self.b.size()).fill_(0)


    def forward(self, x):
        # We record the input for later use
        self.input = x  
        # It's just a matrix-vector product plus bias after it
        return (x @ self.w) + self.b

    def backward(self, dl_dout):
        self.grad_w.add_(self.input.t() @ dl_dout)
        self.grad_b.add_(dl_dout.sum(0))
        return dl_dout @ self.w.t()

    def param(self):

        return [
                (self.w, self.grad_w),
                (self.b, self.grad_b)
                ]

    def zero_grad(self):
        self.grad_w.zero_()
        self.grad_b.zero_()
        
class CNN(Module):
     def __init__(self, num_filters=16, stride=1, size=3):
        super().__init__()
        #self.filters = np.random.randn(num_filters, 3, 3).normal_(0, 1)
        self.filters = torch.empty((num_filters,size,size))
        self.stride = stride
        self.size = size
        
    def forward(self, x):
        self.input = x                             # keep track of last input for later backward propagation

        input_dimension = x.size(dim=1)                                              # input dimension
        output_dimension = int((input_dimension - self.size) / self.stride) + 1         # output dimension

        out = torch.empty((self.filters.size(dim=0), output_dimension, output_dimension))     # create the matrix to hold the
                                                                                        # values of the convolution

        for f in range(self.filters.size(dim=0)):              # convolve each filter over the image,
            tmp_y = out_y = 0                               # moving it vertically first and then horizontally
            while tmp_y + self.size <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dimension:
                    patch = x[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    out[f, out_y, out_x] += torch.sum(self.filters[f] * patch)
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
        
        return out

    def backward(self, dl_out):
        input_dimension = self.input.size(dim=1)         # input dimension


        dout = torch.zeros(self.input.size(dim=1))              # loss gradient of the input to the convolution operation
        dfilt = torch.zeros(self.filters.size())                # loss gradient of filter

        for f in range(self.filters.size(dim=0)):              # loop through all filters
            tmp_y = out_y = 0
            while tmp_y + self.size <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dimension:
                    patch = self.last_input[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    dfilt[f] += torch.sum(din[f, out_y, out_x] * patch, axis=0)
                    dout[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size] += din[f, out_y, out_x] * self.filters[f]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
        #self.filters -= learn_rate * dfilt                  # update filters using SGD
        self.dfilt = dfilt
        return dout, dfilt

    def param(self):
          return [(self.filters, self.dfilt)]


    def zero_grad(self):
        self.dfilt.zero()

class Pooling(Module):
    ## class has no weights 
     def __init__(self, stride=2, size=2):
        self.input = None
        self.stride = stride
        self.size = size
    
     def forward(self, x):
            self.input = x                            # keep track of last input for later backward propagation

        num_channels, h_prev, w_prev = image.size()
        h = int((h_prev - self.size) / self.stride) + 1     # compute output dimensions after the max pooling
        w = int((w_prev - self.size) / self.stride) + 1

        downsampled = torch.zeros((num_channels, h, w))        # hold the values of the max pooling

        for i in range(num_channels):                       # slide the window over every part of the image and
            curr_y = out_y = 0                              # take the maximum value at each step
            while curr_y + self.size <= h_prev:             # slide the max pooling window vertically across the image
                curr_x = out_x = 0
                while curr_x + self.size <= w_prev:         # slide the max pooling window horizontally across the image
                    patch = image[i, curr_y:curr_y + self.size, curr_x:curr_x + self.size]
                    downsampled[i, out_y, out_x] = torch.max(patch)       # choose the maximum value within the window
                    curr_x += self.stride                              # at each step and store it to the output matrix
                    out_x += 1
                curr_y += self.stride
                out_y += 1

        return downsampled
    
    def backward(self, din, learning_rate):
        #### TODO 

    def param(self):                          # pooling layers have no weights
        return None



# Module to combines several other modules
class Sequential(Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def forward(self, x):
        self.x = x
        for m in self.modules:
            x = m.forward(x)
        return x

    def backward(self, dl_dout):
        self.dl_dout = dl_dout
        for m in reversed(self.modules):
            dl_dout = m.backward(dl_dout)
        return dl_dout

    def param(self):
        param = []
        for m in self.modules:
            param.extend(m.param())
        return param

    def zero_grad(self):
        for m in self.modules:
            m.zero_grad()

#==================================================================================

# Modules for activations functions

class ReLU(Module):
    def forward(self, x):
        self.x = x
        return torch.max(torch.zeros_like(x), x)

    def backward(self, dl_dout):
        # clamp forces negative elements to 0.0
        return torch.clamp(self.x.sign(), 0.0, 1.0) * dl_dout


