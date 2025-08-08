import torch
import torch.nn as nn
from math import *


class PolicyGradientAgent(nn.Module):
    def __init__(self,
                 board_size: int,
                 n_pieces: int,
                 channel_num_list: list[int],
                 kernel_size_list: list[int],
                 activation_fn: str = "ReLU",
                 value_head: bool = False):
        """
        We are going to be using a purely convolutional neural network to represent the policy.
        the input is a tensor of shape (batch, n_pieces + 1, board_size, board_size)
        the output is a tensor of shape (batch, 1, board_size, board_size) not probability normalized.
        Args:
            board_size: int, the size of the board
            n_pieces: int, the number of pieces
            channel_num_list: list[int], the number of channels for each layer
            kernel_size_list: list[int], the kernel size for each layer
            activation_fn: str, name of the activation function
            value_head: bool, whether to use a value head
        """
        super().__init__()
        self.board_size = board_size
        self.n_pieces = n_pieces
        self.channel_num_list = [n_pieces + 1] + channel_num_list + [1]
        self.kernel_size_list = kernel_size_list
        self.activation_fn = getattr(nn, activation_fn)
        self.value_head = value_head

        if len(kernel_size_list) == 1:
            self.kernel_size_list = [
                kernel_size_list[0]] * (len(self.channel_num_list) - 1)
        elif len(kernel_size_list) != len(self.channel_num_list) - 1:
            raise ValueError(
                "kernel_size_list must be the same length as channel_num_list")

        self.layer_dict = nn.ModuleDict()
        for i, kernel_size in enumerate(self.kernel_size_list):
            self.layer_dict[f"conv_{i+1}"] = nn.Conv2d(self.channel_num_list[i],
                                                       self.channel_num_list[i+1],
                                                       kernel_size=kernel_size,
                                                       padding = floor(kernel_size/2))
        self.layer_dict[f"activation"] = self.activation_fn()

        if value_head:
            self.value_network = nn.LazyLinear(1)
        else:
            self.value_network = None


    def forward(self, x):
        """
        Args:
            x: torch.Tensor, shape (batch, n_pieces + 1, board_size, board_size)
        Returns:
            x: torch.Tensor, shape (batch, 1, board_size, board_size)
            x is not probability normalized.
            v: torch.Tensor, shape (batch, 1)
            v is a scalar value produced by the value head.
        """
        batch_size, _, board_size, board_size = x.shape
        h = x
        for i in range(len(self.kernel_size_list) - 1):
            h = self.layer_dict[f"conv_{i+1}"](h)
            h = self.layer_dict[f"activation"](h)
        
        x = self.layer_dict[f'conv_{i + 2}'](h)
        if self.value_head:
            v = self.value_network(h.reshape(batch_size, -1))
            return x, v
        else:
            return x

    def act(self, x):
        """
        Args:
            x: torch.Tensor, shape (batch, n_pieces + 1, board_size, board_size)
        Returns:
            action: torch.Tensor, shape (batch, 1)
            logprob: torch.Tensor, shape (batch, 1, board_size, board_size)
            v: torch.Tensor, shape (batch, 1)
        """
        batch_size, _, board_size, board_size = x.shape
        if self.value_head:
            x, v = self.forward(x)
        else:
            x = self.forward(x)
        result = x.reshape(batch_size, board_size * board_size)
        # sample from the softmax of the result
        action = torch.multinomial(torch.nn.Softmax(dim=-1)(result), 1)
        logprob = torch.nn.LogSoftmax(dim=-1)(result).gather(-1, action)

        if self.value_head:
            return action, logprob, v
        else:
            return action, logprob

    def get_value(self, x):
        """
        Args:
            x: torch.Tensor, shape (batch, n_pieces + 1, board_size, board_size)
        Returns:
            x: torch.Tensor, shape (batch, 1)
        """
        if not self.value_head:
            raise ValueError("Value head is not provided")
        else:
            _, v = self.forward(x)
            return v
