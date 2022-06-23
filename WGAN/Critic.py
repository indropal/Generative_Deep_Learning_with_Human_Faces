import torch, torchvision, os, PIL, pdb
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import numpy as np

class Critic(torch.nn.Module):
    
    def __init__(self, hidden_conv_channels=64):
        super(Critic, self).__init__()

        self.critic_model = torch.nn.Sequential(
            # Input is a RGB image: 3 channels, output channel: 16,  kernel size: 4, paddding: 2, stride: 1 
            torch.nn.Conv2d(3, int(hidden_conv_channels*0.25), 4, 2, 1), # new dimension (64, 64, 16) -> 64X64 image with channels: 16
            torch.nn.InstanceNorm2d(int(hidden_conv_channels*0.25)), # Checking & stabilising by Instance 
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(int(hidden_conv_channels*0.25), int(hidden_conv_channels*0.5), 4, 2, 1), # new dimension (32, 32, 32) -> 32X32 image with channels: 32
            torch.nn.InstanceNorm2d(int(hidden_conv_channels*0.5)), 
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(int(hidden_conv_channels*0.5), hidden_conv_channels, 4, 2, 1), # new dimension (16, 16, 64) -> 16X16 image with channels: 64
            torch.nn.InstanceNorm2d(hidden_conv_channels), 
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(hidden_conv_channels, hidden_conv_channels*2, 4, 2, 1), # new dimension (8, 8, 128) -> 8X8 image with channels: 128
            torch.nn.InstanceNorm2d(hidden_conv_channels*2), 
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(hidden_conv_channels*2, hidden_conv_channels*4, 4, 2, 1), # new dimension (4, 4, 256) -> 4X4 image with channels: 256
            torch.nn.InstanceNorm2d(hidden_conv_channels*4), 
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(hidden_conv_channels*4, 1, 4, 1, 0), # new dimension (1, 1, 1) -> 1X1 image with channels: 1
        )
  
    def forward(self, image_data):

        critic_prediction = self.critic_model(image_data) # expected output Tensor dims -> ('batch_size', 1, 1, 1)

        return critic_prediction.view(len(critic_prediction), -1) # dims ~ ('batch_size' X 1)