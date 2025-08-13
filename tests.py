import torch
from utils.modules import *


#Test that Conv3d layer does not change input shape
input = torch.rand(64, 1, 28, 28, 28)

model = ConvolutionalSelfAttention3d(1, (28*28), 4)

output = model(input)
assert output.shape == input.shape
print('conv3d CSA passed')

#Test that Conv2d layer does not change input shape
input = torch.rand(64, 1, 28, 28)

model = ConvolutionalSelfAttention2d(1, (28*28), 4)

output = model(input)
assert output.shape == input.shape
print('conv2d CSA passed')

input = torch.rand(64, 1, 28, 28)

model = SelfAttentionModule((28*28))

output = model(input.flatten(start_dim=1))
assert output.shape == input.shape
print('SA test passed')

print('all tests passed')
