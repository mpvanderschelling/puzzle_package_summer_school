import torch 
import torch.nn as nn
from abc import ABC, abstractmethod
from math import floor

class PolicyGradientAgent(nn.Module):
    def __init__(self,
                 board_size: int,
                 n_pieces: int,
                 channel_num_list: list[int],
                 kernel_size_list: list[int],
                 activation_fn:str = "ReLU",
                 value_network: nn.Module = None):
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
        """
        super().__init__()
        self.board_size = board_size
        self.n_pieces = n_pieces
        self.channel_num_list = [n_pieces + 1] + channel_num_list + [1]
        self.kernel_size_list = kernel_size_list
        self.activation_fn = getattr(nn, activation_fn)
        self.value_network = value_network
        
        if len(kernel_size_list) == 1:
            self.kernel_size_list = [kernel_size_list[0]] * (len(channel_num_list) - 1)
        elif len(kernel_size_list) != len(channel_num_list) - 1:
            raise ValueError("kernel_size_list must be the same length as channel_num_list")
        
        self.layer_dict = nn.ModuleDict()
        for i, kernel_size in enumerate(self.kernel_size_list):
            self.layer_dict[f"conv_{i+1}"] = nn.Conv2d(self.channel_num_list[i], 
                                                    self.channel_num_list[i+1], 
                                                    kernel_size=kernel_size)
        self.layer_dict[f"activation"] = self.activation_fn()
        
    def forward(self, x):
        """
        Args:
            x: torch.Tensor, shape (batch, n_pieces + 1, board_size, board_size)
        Returns:
            x: torch.Tensor, shape (batch, 1, board_size, board_size)
            x is not probability normalized.
        """
        for i in range(1, len(self.kernel_size_list)):
            x = self.layer_dict[f"conv_{i}"](x)
            x = self.layer_dict[f"activation"](x)
        x = self.layer_dict[f"conv_{len(self.kernel_size_list)}"](x)
        return x
    
    def act(self, x):
        """
        Args:
            x: torch.Tensor, shape (batch, n_pieces + 1, board_size, board_size)
        Returns:
            action: torch.Tensor, shape (batch, 1)
            logprob: torch.Tensor, shape (batch, 1, board_size, board_size)
        """
        batch_size, channels, board_size, board_size = x.shape
        result = self.forward(x)
        result = result.reshape(batch_size, board_size * board_size)
        # sample from the softmax of the result
        action = torch.multinomial(torch.nn.Softmax(dim = -1)(result), 1)
        logprob = torch.nn.LogSoftmax(dim = -1)(result).gather(-1, action)

        row, col = divmod(action, board_size)
        if self.value_network is None:
            return (row, col), logprob
        else:
            return (row, col), logprob, self.get_value(x)
    
    def get_value(self, x):
        """
        Args:
            x: torch.Tensor, shape (batch, n_pieces + 1, board_size, board_size)
        Returns:
            x: torch.Tensor, shape (batch, 1)
        """
        if self.value_network is None:
            raise ValueError("Value network is not provided")
        return self.value_network(x)