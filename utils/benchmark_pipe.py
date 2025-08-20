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
import time

def do_pca(lats, labs, folder, epc, acc):
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
  plt.title(f'PCA of Test Data Colored by Labels epc: {epc}, acc: {acc}')
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

def run_test(model, trainloader, testloader, folder = 'lats', nlr = 0.001):
  np.random.seed(42)
  torch.manual_seed(42) 
  start = time.time()
  #for reproducibility
  os.makedirs(folder, exist_ok=True)
  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr=nlr)
  plot_latents = False
  epochs = 200
  lats, labs, losses, accur, tlosses, taccur = [], [], [], [], [], []
  for e in tqdm(range(epochs)):
      running_loss, trn_corr = 0, 0
      if(e % 1 == 0):
        plot_latents = True
        lats, labs = [], []
      for images, labels in (trainloader):
          if(images.shape[0] != 64):
            continue
          optimizer.zero_grad()
          pred, emb = model(images.to(model.device))
          output = F.log_softmax(pred, dim=1)
          loss = criterion(output, labels.to(model.device))
          loss.backward()
          optimizer.step()
          _, train_out = torch.max(output, 1)
          running_loss += loss.item()
          trn_corr += torch.sum(train_out.detach().cpu() == labels.data)
      
      with torch.no_grad():
        test_run, test_corr = 0, 0
        for images, labels in (testloader):
          if(images.shape[0] != 64):
            continue
          pred, emb = model(images.to(model.device))
          output = F.log_softmax(pred, dim=1)
          loss = criterion(output, labels.to(model.device))
          _, test_out = torch.max(output, 1)
          test_run += loss.item()
          test_corr += torch.sum(test_out.detach().cpu() == labels.data)
          if(plot_latents):
              lats.append(emb.detach().cpu().squeeze().flatten(start_dim=1).numpy())
              labs.append(labels)
      
      test_accuracy = num_correct.float()/ len(testloader)
      tlosses.append(test_run/len(testloader))
      taccur.append(test_accuracy)
      losses.append(running_loss/len(trainloader))
      accur.append(trn_corr.float() / len(trainloader))
      
      if(plot_latents):
        do_pca(lats, labs, folder, e, test_accuracy)
        plot_latents = False
  print('\n best Train Loss: ', min(losses), '\t best Train Accuracy: ', min(accur))
  print('\n best Test Loss: ', min(losses), '\t best Test Accuracy: ', min(taccur))
  plt.plot(np.arange(1, 201), losses)
  plt.title(f'Loss over epochs')
  plt.show()
  plt.close()
  return min(losses), time.time() - start
