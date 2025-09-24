import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalSelfAttention3d(nn.Module):
  def __init__(self, in_ch, dimprod, scale_reduce = None):
    super().__init__()
    self.dw_conv = nn.Conv3d(in_ch, in_ch, 5, padding = 2)
    if(scale_reduce == None):
      self.pw_conv_1 = nn.Conv3d(in_ch, dimprod, 1)
      self.pw_conv_2 = nn.Sequential(
          nn.Conv3d(dimprod, in_ch, 1),
          nn.Sigmoid()
      )
    else:
      scaled_hw = (dimprod // (scale_reduce**3))
      self.pw_conv_1 = nn.Sequential(
          nn.AvgPool3d(scale_reduce),
          nn.Conv3d(in_ch, scaled_hw, 1)
      )
      self.pw_conv_2 = nn.Sequential(
          nn.Conv3d(scaled_hw, in_ch, 1),
          nn.Sigmoid(),
          nn.Upsample(scale_factor = scale_reduce)
      )
    self.pw_conv_3 = nn.Sequential(
        nn.Conv3d(in_ch, 3*in_ch, 1),
        nn.Conv3d(3*in_ch, in_ch, 1)
    )

  def forward(self, x):
    q, k, v = self.dw_conv(x), self.dw_conv(x), self.dw_conv(x)
    q, k = self.pw_conv_1(q), self.pw_conv_1(k)
    k = k.flatten(start_dim=1).T.reshape(q.shape)
    qk = self.pw_conv_2((q * k))
    qkv = v * qk
    out_feat = self.pw_conv_3(qkv)
    return x * out_feat

class ConvolutionalSelfAttention2d(nn.Module):
  def __init__(self, in_ch, dimprod, activation = 'sigmoid', pool = 'avg', scale_reduce = None):
    super().__init__()
    self.dw_conv = nn.Conv2d(in_ch, in_ch, 5, padding = 2)

    if activation == 'sigmoid':
      act = nn.Sigmoid()
    elif activation == 'gelu':
      act = nn.GELU()
    elif activation = 'relu':
      act = nn.ReLU()

    if pool == 'avg':
      pool = nn.AvgPool2d(scale_reduce)
    elif pool == 'max':
      pool = nn.MaxPool2d(scale_reduce)
    elif pool == 'adavg':
      pool = nn.AdaptiveAvgPool2d(scale_reduce)
    elif pool == 'adamax':
      pool = nn.AdaptiveMaxPool2d(scale_reduce)
    
    if(scale_reduce == None):
      self.pw_conv_1 = nn.Conv2d(in_ch, dimprod, 1)
      self.pw_conv_2 = nn.Sequential(
          nn.Conv2d(dimprod, in_ch, 1),
          nn.Sigmoid()
      )

    else:
      scaled_hw = (dimprod // (scale_reduce**2))
      self.pw_conv_1 = nn.Sequential(
          pool,
          nn.Conv2d(in_ch, scaled_hw, 1)
      )
      self.pw_conv_2 = nn.Sequential(
          nn.Conv2d(scaled_hw, in_ch, 1),
          nn.Sigmoid(),
          nn.Upsample(scale_factor = scale_reduce)
      )
    self.pw_conv_3 = nn.Sequential(
        nn.Conv2d(in_ch, 3*in_ch, 1),
        act,
        nn.Dropout(0.1),
        nn.Conv2d(3*in_ch, in_ch, 1)
    )

  def forward(self, x):
    q, k, v = self.dw_conv(x), self.dw_conv(x), self.dw_conv(x)
    q, k = self.pw_conv_1(q), self.pw_conv_1(k)
    k = k.flatten(start_dim=1).T.reshape(q.shape)
    qk = self.pw_conv_2((q * k))
    qkv = v * qk
    out_feat = self.pw_conv_3(qkv)
    return x * out_feat

class ConvolutionalSelfAttention(nn.Module):
  def __init__(self, in_ch, dimprod, dims = 2, activation = 'sigmoid', pool = 'avg', scale_reduce = None):
    super().__init__()
    if dims == 2:
      Convolution = nn.Conv2d
    elif dims == 3:
      Convolusion = nn.Conv3d
    self.dw_conv = Convolution(in_ch, in_ch, 5, padding = 2)

    if activation == 'sigmoid':
      act = nn.Sigmoid()
    elif activation == 'gelu':
      act = nn.GELU()
    elif activation = 'relu':
      act = nn.ReLU()

    if pool == 'avg':
      pool = nn.AvgPool2d(scale_reduce)
    elif pool == 'max':
      pool = nn.MaxPool2d(scale_reduce)
    elif pool == 'adavg':
      pool = nn.AdaptiveAvgPool2d(scale_reduce)
    elif pool == 'adamax':
      pool = nn.AdaptiveMaxPool2d(scale_reduce)
    
    if(scale_reduce == None):
      self.pw_conv_1 = Convolution(in_ch, dimprod, 1)
      self.pw_conv_2 = nn.Sequential(
          Convolution(dimprod, in_ch, 1),
          nn.Sigmoid()
      )

    else:
      scaled_hw = (dimprod // (scale_reduce**2))
      self.pw_conv_1 = nn.Sequential(
          pool,
          Convolution(in_ch, scaled_hw, 1)
      )
      self.pw_conv_2 = nn.Sequential(
          nn.Convolution(scaled_hw, in_ch, 1),
          nn.Sigmoid(),
          nn.Upsample(scale_factor = scale_reduce)
      )
    self.pw_conv_3 = nn.Sequential(
        nn.Convolution(in_ch, 3*in_ch, 1),
        act,
        nn.Dropout(0.1),
        nn.Convolution(3*in_ch, in_ch, 1)
    )

  def forward(self, x):
    q, k, v = self.dw_conv(x), self.dw_conv(x), self.dw_conv(x)
    q, k = self.pw_conv_1(q), self.pw_conv_1(k)
    k = k.flatten(start_dim=1).T.reshape(q.shape)
    qk = self.pw_conv_2((q * k))
    qkv = v * qk
    out_feat = self.pw_conv_3(qkv)
    return x * out_feat
    
class SelfAttentionModule(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.self_attention = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)

  def forward(self, x):
    output, _ = self.self_attention(query=x, key=x, value=x)
    return output
