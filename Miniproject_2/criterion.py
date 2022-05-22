""" Module containing implementations of loss functions """

from torch import zeros, softmax, log_softmax
from models import Module


class LossMSE(Module):
    """ Module class performing MSE loss computation """

    def __init__(self):
        """
        LossMSE constructor
        """

        Module.__init__(self)
        self.y = None
        self.target = None
        self.e = None
        self.n = None

    def forward(self, y, target):
        """
        MSE computation

        :param y: output of the final layer, torch.Tensor
        :param target: target data, torch.Tensor

        :returns: MSE(f(x), y) = Sum(e^2) / n, e = y - f(x)
        """

        self.y = y.clone()
        self.target = target

        self.e = (self.y - self.target)
        self.n = self.y.size(0)

        return self.e.pow(2).mean()

    def backward(self):
        """
        MSE gradient computation

        :returns: Grad(MSE(f(x), y)) = 2e / n, e = y - f(x)
        """

        return 2 * self.e / self.n

      def param(self):                          # Relu layers have no weights
        return []