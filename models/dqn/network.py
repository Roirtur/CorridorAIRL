import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, input_channels: int, action_dim: int, board_size: int = 9):
        super(DuelingDQN, self).__init__()
        
        # CNN Feature Extractor
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size: 64 * N * N
        self.flatten_dim = 64 * board_size * board_size
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU()
        )

        # Value Stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage Stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.flatten_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        # state: (B, C, H, W)
        # action: (B, ActionDim)
        
        s_feat = self.conv_net(state)
        a_feat = self.action_encoder(action)
        
        val = self.value_stream(s_feat)
        
        sa_feat = torch.cat([s_feat, a_feat], dim=1)
        adv = self.advantage_stream(sa_feat)
        
        q_val = val + adv
        return q_val.squeeze(1)
