import torch
from torch import nn
from torch.nn import Conv2d, ReLU


class CharacterRecognizer(nn.Module):
    def __init__(self, image_size, num_labels, kernel_size, channels, pooling_kernel_size, stride):
        super().__init__()
        self.image_size = image_size
        self.convolution_out_size = (image_size - kernel_size) // stride + 1
        self.maxpool_out_size = (self.convolution_out_size - pooling_kernel_size) // pooling_kernel_size + 1
        self.num_heads = self.maxpool_out_size
        # ^ forced by pytorch's attention requirements (embedding dim must be divisible by no. of heads)
        self.dim_before_affine = self.maxpool_out_size ** 2 * channels
        self.convolution = Conv2d(in_channels=1,  # greyscale
                                  out_channels=channels,
                                  kernel_size=(kernel_size, kernel_size),
                                  stride=(stride,))
        self.relu = ReLU()
        self.max_pool = nn.MaxPool2d(pooling_kernel_size)
        self.affine = nn.Linear(self.dim_before_affine, self.dim_before_affine)
        self.to_logits = nn.Linear(self.dim_before_affine, num_labels)

    def forward(self, batch):
        batch = self.convolution(torch.unsqueeze(batch, dim=1))
        batch = self.relu(batch)
        batch = self.max_pool(batch)
        batch = torch.flatten(batch, start_dim=1)
        batch = self.affine(batch)
        batch = self.relu(batch)
        batch = self.to_logits(batch)
        return batch

