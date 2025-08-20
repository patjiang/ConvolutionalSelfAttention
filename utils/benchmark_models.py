from ConvolutionalSelfAttention.utils.modules import *
import torch.nn

class MNIST_CSA_1_layer(nn.Module):
  def __init__(self, in_ch, dimprod):
    super().__init__()
    self.downsamp_1 = nn.MaxPool2d(2)
    self.CSA_1 = ConvolutionalSelfAttention2d(in_ch, dimprod // 4, 2)
    self.out = nn.LazyLinear(10)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

  def forward(self, x):
    x = self.downsamp_1(x)
    x = self.CSA_1(x)
    pred = self.out(x.flatten(start_dim=1))
    return pred, x

class MNIST_CSA_2_layer(nn.Module):
  def __init__(self, in_ch, dimprod):
    super().__init__()
    self.downsamp_1 = nn.MaxPool2d(2)
    self.CSA_1 = ConvolutionalSelfAttention2d(in_ch, dimprod // 4, 2)
    self.downsamp_2 = nn.MaxPool2d(2)
    self.CSA_2 = ConvolutionalSelfAttention2d(in_ch, (dimprod // 16))
    self.out = nn.LazyLinear(10)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

  def forward(self, x):
    x = self.downsamp_1(x)
    x = self.CSA_1(x)
    x = self.downsamp_2(x)
    x = self.CSA_2(x)
    pred = self.out(x.flatten(start_dim=1))
    return pred, x

class MNIST_CSA_1_layer_full(nn.Module):
  def __init__(self, in_ch, dimprod):
    super().__init__()
    self.downsamp_1 = nn.MaxPool2d(2)
    self.CSA_1 = ConvolutionalSelfAttention2d(in_ch, dimprod // 4)
    self.out = nn.LazyLinear(10)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

  def forward(self, x):
    x = self.downsamp_1(x)
    x = self.CSA_1(x)
    pred = self.out(x.flatten(start_dim=1))
    return pred, x

class MNIST_CSA_2_layer_full(nn.Module):
  def __init__(self, in_ch, dimprod):
    super().__init__()
    self.downsamp_1 = nn.MaxPool2d(2)
    self.CSA_1 = ConvolutionalSelfAttention2d(in_ch, dimprod // 4)
    self.downsamp_2 = nn.MaxPool2d(2)
    self.CSA_2 = ConvolutionalSelfAttention2d(in_ch, (dimprod // 16))
    self.out = nn.LazyLinear(10)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

  def forward(self, x):
    x = self.downsamp_1(x)
    x = self.CSA_1(x)
    x = self.downsamp_2(x)
    x = self.CSA_2(x)
    pred = self.out(x.flatten(start_dim=1))
    return pred, x

class MNIST_SA_1_layer(nn.Module):
  def __init__(self, in_ch, dimprod):
    super().__init__()
    self.downsamp_1 = nn.Linear(dimprod, dimprod // 4)
    self.SA_1 = SelfAttentionModule(dimprod // 4)
    self.out = nn.LazyLinear(10)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

  def forward(self, x):
    x = x.flatten(start_dim=1)
    x = self.downsamp_1(x)
    x = self.SA_1(x)
    pred = self.out(x)
    return pred, x

class MNIST_SA_2_layer(nn.Module):
  def __init__(self, in_ch, dimprod):
    super().__init__()
    self.downsamp_1 = nn.Linear(dimprod, dimprod // 4)
    self.SA_1 = SelfAttentionModule(dimprod // 4)
    self.downsamp_2 = nn.Linear(dimprod // 4, dimprod // 16)
    self.SA_2 = SelfAttentionModule((dimprod // 16))
    self.out = nn.LazyLinear(10)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

  def forward(self, x):
    x = x.flatten(start_dim=1)
    x = self.downsamp_1(x)
    x = self.SA_1(x)
    x = self.downsamp_2(x)
    x = self.SA_2(x)
    pred = self.out(x)
    return pred, x

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        pred = self.fc1(x)
        return pred, x

class CNN_torch(nn.Module):
    def __init__(self):
        super(CNN_torch, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        pred = self.fc2(x)
        return pred, x

class LeNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(1, 20, 5, 1)
      self.conv2 = nn.Conv2d(20, 50, 5, 1)
      self.fc1 = nn.Linear(4*4*50, 500)
      self.dropout1 = nn.Dropout(0.5)
      self.fc2 = nn.Linear(500, 10)
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 4*4*50)
      x = F.relu(self.fc1(x))
      x = self.dropout1(x)
      x = self.fc2(x)
      return x
