""" Module providing implementations of the SGD and Adam optimization algorithms as well as model evaluation """

import torch
import math
from .criterion import LossMSE





class __Optimizer:
    """
    Private class serving as an interface for all optimizers which should inherit it
    """

    def __init__(self, model, nb_epochs, mini_batch_size, lr, criterion):
        """
        __Optimizer constructor

        :param model: the model to train, models.Sequential object (only one currently possible)
        :param nb_epochs: maximum number of training epochs, positive int
        :param mini_batch_size: number of samples per mini-batch, int in [1, num_train_samples]
        :param lr: learning rate, positive float
        :param criterion: loss function to optimize, criteria.LossMSE or criteria.LossCrossEntropy object

        :raises ValueError if:
                - `nb_epochs` is not a positive int
                - `mini_batch_size` is not a positive int
                - `lr` is not a positive float
        """

        if not isinstance(nb_epochs, int) or nb_epochs <= 0:
            raise ValueError("Number of training epochs must be a positive integer")
        if not isinstance(mini_batch_size, int) or mini_batch_size <= 0:
            raise ValueError("Mini-batch size must be a positive integer")
        if not isinstance(lr, float) or lr <= 0:
            raise ValueError("Learning rate must be a positive number")

        self.model = model
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.mini_batch_size = mini_batch_size
        self.criterion = criterion

    def train(self, train_input, train_target, verbose=True):
        """
        Function implementing the mini-batch training procedure, the same for all optimizers

        :param train_input: torch.Tensor with train input data
        :param train_target: torch.Tensor with train target data
        :param verbose: whether to print total loss values after each epoch, boolean, optional, default is True

        :returns: the trained models.Sequential model
        """

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
                self.step()

            if verbose:
                print("{} iteration: loss={}".format(e, sum_loss))
        return self.model

    def step(self):
        """
        Function that implements the gradient update step of the optimizer
        """

        raise NotImplementedError


class SGD(__Optimizer):
    """
    Class implementing mini-batch SGD optimization
    """

    def __init__(self, model, nb_epochs=50, mini_batch_size=1, lr=1e-2, criterion=LossMSE()):
        """
        SGD constructor

        :param model: the model to train, models.Sequential object (only one currently possible)
        :param nb_epochs: maximum number of training epochs, positive int, optional, default is 50
        :param mini_batch_size: number of samples per mini-batch, int in [1, num_train_samples], optional, default is 1
        :param lr: learning rate, positive float, optional, default is 1e-2
        :param criterion: loss function to optimize, models.Module object, optional, default is criteria.LossMSE

        :raises ValueError if:
                - `nb_epochs` is not a positive int
                - `mini_batch_size` is not a positive int
                - `lr` is not a positive float
        """

        super().__init__(model, nb_epochs, mini_batch_size, lr, criterion)

    def step(self):
        """
        Overloads __Optimizer.step
        """

        self.model.update(self.lr)