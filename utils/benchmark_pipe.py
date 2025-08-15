from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

def do_pca(lats, labs, folder, epc):
  data = np.concat(lats)
  labels = np.concat(labs)

  pca = PCA(n_components=2)
  principal_components = pca.fit_transform(data)

  explained_variance_ratio = pca.explained_variance_ratio_
  #print("Explained variance ratio for each component:", explained_variance_ratio)
  #print("Cumulative explained variance:", explained_variance_ratio.cumsum())

  plt.figure(figsize=(8, 6))
  scatter = plt.scatter(
      principal_components[:, 0],
      principal_components[:, 1],
      c=labels,
      cmap='tab10',
      s=50,
      alpha=0.7
  )
  plt.colorbar(scatter, label='Label')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.title(f'PCA of Data Colored by Labels {epc}')
  plt.grid(True)
  #plt.show()
  tmp = str(epc)
  if(len(tmp) == 1):
    tmp2 = '00' + tmp
  elif(len(tmp) == 2):
    tmp2 = '0' + tmp
  else:
    tmp2 = tmp
  plt.savefig(f'{folder}/frame_{tmp2}.png')
  plt.close()

def run_test(model, trainloader, folder = 'lats', nlr = 0.001):
  np.random.seed(42)
  torch.manual_seed(42) 
  #for reproducibility
  os.makedirs(folder, exist_ok=True)
  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr=nlr)
  plot_latents = False
  epochs = 200
  lats, labs, losses = [], [], []
  for e in tqdm(range(epochs)):
      running_loss = 0
      if(e % 1 == 0):
        plot_latents = True
        lats, labs = [], []
      for images, labels in (trainloader):
          optimizer.zero_grad()

          pred, emb = model(images.to(model.device))
          output = F.log_softmax(pred, dim=1)
          nlloss = criterion(output, labels.to(model.device))
          loss = nlloss
          loss.backward()
          optimizer.step()
          if(plot_latents):
            lats.append(emb.detach().cpu().squeeze().flatten(start_dim=1).numpy())
            labs.append(labels)
          running_loss += loss.item()
      losses.append(running_loss/len(trainloader))
      
      if(plot_latents):
        do_pca(lats, labs, folder, e)
        plot_latents = False
  print('\n best NLL: ', min(losses))
  plt.plot(np.arange(1, 201), losses)
  plt.title(f'Loss over epochs')
  plt.show()
  plt.close()
