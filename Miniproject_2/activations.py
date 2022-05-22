""" Module containing implementations of activation functions """

from modules import Module


class ReLU(Module):
    """ Module class performing ReLU activation """

    def __init__(self, ):
        """
        ReLU constructor
        """

        super(ReLU, self).__init__()
        self.s = None

    def forward(self, *input_):
        """
        ReLU forward pass

        :param input_: output of the previous layer, torch.Tensor

        :returns: ReLU(x_l) = max(x_l, 0)
        """

        s = input_[0].clone()
        self.s = s

        s[s < 0] = 0.

        return s

    def backward(self, *gradwrtoutput):
        """
        ReLU backward pass

        :param gradwrtoutput: gradient of the next layer, torch.Tensor

        :returns: Grad(ReLU(x_l)) = I(x_l > 0) * Grad(x_l+1)
        """

        input_ = gradwrtoutput[0].clone()

        out = self.s.clone()
        out[out > 0] = 1
        out[out < 0] = 0

        return out.mul(input_)


class LeakyReLU(Module):
    """ Module class performing LeakyReLU activation """

    def __init__(self, alpha=0.01):
        """
        LeakyReLU constructor

        :param alpha: non-negative float

        :raises ValueError, if `alpha` is not a non-negative float
        """

        if not isinstance(alpha, float) or alpha < 0:
            raise ValueError("LeakyReLU alpha must be a non-negative float")

        Module.__init__(self)
        self.alpha = alpha
        self.s = None

    def forward(self, *input_):
        """
        LeakyReLU forward pass

        :param input_: output of the previous layer, torch.Tensor

        :returns: LeakyReLU(x_l) = max(x_l, alpha)
        """

        s = input_[0].clone()
        self.s = s

        s[s < 0] = self.alpha * s[s < 0]

        return s

    def backward(self, *gradwrtoutput):
        """
        LeakyReLU backward pass

        :param gradwrtoutput: gradient of the next layer, torch.Tensor

        :returns: Grad(LeakyReLU(x_l)) = (I(x_l > 0) + alpha * I(x_l < 0)) * Grad(x_l+1)
        """

        input_ = gradwrtoutput[0].clone()

        out = self.s.clone()
        out[out > 0] = 1
        out[out < 0] = self.alpha

        return out.mul(input_)