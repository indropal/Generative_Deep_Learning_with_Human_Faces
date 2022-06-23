import torch, torchvision, os, PIL, pdb
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import numpy as np


class Generator(torch.nn.Module):
    
    def __init__(self, latent_space_dim=256, hidden_conv_channels=64):

        super(Generator, self).__init__()

        # Latent Space Dimensions -> Noise Vector Dimension (channels) used as input to Generator
        self.latent_space_dim = latent_space_dim

        # ConvTranspose 2d -> Increse dimension by: (original_dimension-1)*stride + kernel_size
        self.generator_model = torch.nn.Sequential(

            # Layer Details: input_channels=latent_space_dim | output_channels=hidden_conv_channels*8 | kernel_size=4 | stride=1 | padding=0
            torch.nn.ConvTranspose2d(latent_space_dim, hidden_conv_channels*8, 4, 1, 0), # new dimension (4, 4, 512) -> 4X4 image with channels: 512
            torch.nn.BatchNorm2d(hidden_conv_channels*8),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(hidden_conv_channels*8, hidden_conv_channels*4, 4, 2, 1), # new dimension (8, 8, 256) -> 8X8 image with channels: 256
            torch.nn.BatchNorm2d(hidden_conv_channels*4),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(hidden_conv_channels*4, hidden_conv_channels*2, 4, 2, 1), # new dimension (16, 16, 128) -> 16X16 image with channels: 128
            torch.nn.BatchNorm2d(hidden_conv_channels*2),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(hidden_conv_channels*2, hidden_conv_channels*1, 4, 2, 1), # new dimension (32, 32, 64) -> 32X32 image with channels: 64
            torch.nn.BatchNorm2d(hidden_conv_channels*1),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(hidden_conv_channels*1, int(hidden_conv_channels*0.5), 4, 2, 1), # new dimension (64, 64, 32) -> 64X64 image with channels: 32
            torch.nn.BatchNorm2d(int(hidden_conv_channels*0.5)),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(int(hidden_conv_channels*0.5), 3, 4, 2, 1), # new dimension (128, 128, 3) -> 128X128 image with channels: 3
            torch.nn.Tanh(),

        )

    def forward(self, noise_latent_tensor):
        
        # we want Latent Tensor to have following dims: (batch_size X num_channels_latent_space X 1 X 1)
        latent_tensor = noise_latent_tensor.view(len(noise_latent_tensor), self.latent_space_dim, 1, 1) 

        return self.generator_model(latent_tensor)
