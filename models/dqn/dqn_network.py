import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DQN(nn.Module):
    """
    Dueling DQN with CNN architecture for Corridor game.
    Processes 6-channel spatial representation of the board state.
    """
    def __init__(self, board_size, output_dim, dropout_rate=0.2):
        super(DQN, self).__init__()
        
        self.output_dim = output_dim
        
        self.board_size = board_size
        self.channels = 6
        
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.flatten_size = 64 * self.board_size * self.board_size
        
        # Dueling architecture
        # Value stream (estimates the board state)
        self.value_fc1 = nn.Linear(self.flatten_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Advantage stream (choses the best action)
        self.advantage_fc1 = nn.Linear(self.flatten_size, 256)
        self.advantage_fc2 = nn.Linear(256, output_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Convolutional feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten for fully connected layers
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        
        # Value stream
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)
        
        # Combine value and advantage using dueling architecture formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
