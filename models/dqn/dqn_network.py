import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DQN(nn.Module):
    """DQN Neural network."""
    def __init__(self, board_size, output_dim, dropout_rate=0.2):
        super(DQN, self).__init__()
        
        self.board_size = board_size
        self.output_dim = output_dim
        
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError
