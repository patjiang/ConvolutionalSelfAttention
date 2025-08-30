import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from ConvolutionalSelfAttention.utils.modules import *
from ConvolutionalSelfAttention.utils.benchmark_models import *
from ConvolutionalSelfAttention.utils.benchmark_pipe import *
from ConvolutionalSelfAttention.utils.data_load_test import *


transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('MNIST_data_train/', download=True, train=True, transform=transform)
testset = datasets.MNIST('MNIST_data_test/', download=True, train=False, transform=transform)

model = MNIST_CSA_1_layer(1, (28*28)).to('cuda')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
csa_1_loss, csa_1_time = run_test(model, trainloader, testloader, '')
csa_1 = sum(param.numel() for param in model.parameters())

model = MNIST_CSA_2_layer(1, (28*28)).to('cuda')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
csa_2_loss, csa_2_time = run_test(model, trainloader, testloader, '')
csa_2 = sum(param.numel() for param in model.parameters())

model = MNIST_CSA_1_layer_full(1, (28*28)).to('cuda')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
csa_1f_loss, csa_1f_time = run_test(model, trainloader, testloader, '')
csa_1f = sum(param.numel() for param in model.parameters())

model = MNIST_CSA_2_layer_full(1, (28*28)).to('cuda')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
csa_2f_loss, csa_2f_time = run_test(model, trainloader, testloader, '')
csa_2f = sum(param.numel() for param in model.parameters())

model = MNIST_SA_1_layer(1, (28*28)).to('cuda')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
lsa_1_loss, lsa_1_time = run_test(model, trainloader, testloader, folder = '', nlr = 0.00001)
lsa_1 = sum(param.numel() for param in model.parameters())

model = MNIST_SA_2_layer(1, (28*28)).to('cuda')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
lsa_2_loss, lsa_2_time = run_test(model, trainloader, testloader, folder = '', nlr = 0.00001)
lsa_2 = sum(param.numel() for param in model.parameters())

model = CNN().to('cuda')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
cnn_1_loss, cnn_1_time = run_test(model, trainloader, testloader, '')
cnn_1 = sum(param.numel() for param in model.parameters())

model = CNN_torch().to('cuda')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
cnn_2_loss, cnn_2_time = run_test(model, trainloader, testloader, folder='', nlr = 0.00002)
cnn_2 = sum(param.numel() for param in model.parameters())

model = LeNet().to('cuda')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
lenet_loss, lenet_time = run_test(model, trainloader, testloader, folder = '', nlr = 0.00002)
lenet = sum(param.numel() for param in model.parameters())

import matplotlib.pyplot as plt
import numpy as np

models = ['CSA 1 Layer', 'CSA 2 Layers', 'CSA 1 Layer full', 'CSA 2 Layers full',
          'LSA 1 Layer', 'LSA 2 Layers', 'CNN Light', 'CNN Torch Example', 'LeNet']
log_params = [csa_1, csa_2, csa_1f, csa_2f, lsa_1, lsa_2, cnn_1, cnn_2, lenet]
best_nll = [csa_1_loss, csa_2_loss, csa_1f_loss, csa_2f_loss, lsa_1_loss, lsa_2_loss,
            cnn_1_loss, cnn_2_loss, lenet_loss]
time_spent = [csa_1_time, csa_2_time, csa_1f_time, csa_2f_time, lsa_1_time, lsa_2_time,
              cnn_1_time, cnn_2_time, lenet_time]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

bars1 = ax1.bar(models, log_params, color=colors)
ax1.set_title('Number of Parameters Amount Comparison')
ax1.tick_params(axis='x', rotation=90)
ax1.set_ylabel('Param Count (symlog scale)')
ax1.set_yscale('symlog')

bars2 = ax2.bar(models, -np.log(best_nll), color=colors)
ax2.set_title('Best Negative Log(NLL) Comparison')
ax2.tick_params(axis='x', rotation=90)
ax2.set_ylabel('Best Negative Log(NLL)')

bars3 = ax3.bar(models, time_spent, color=colors)
ax3.set_title('Training Time Comparison')
ax3.tick_params(axis='x', rotation=90)
ax3.set_ylabel('Time (s)')

fig.legend(bars1, models, loc="center right", bbox_to_anchor=(1.1, 0.5))

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig('benchmarks.png', dpi=200, bbox_inches="tight")

plt.show()
