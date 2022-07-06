# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:05:03 2022

@author: MauritsvandenOeverPr
"""
import os
import random
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


from datasets import load_dataset
from models import GaussianVAE, StudentsTVAE
from data.datafuncs import GetData, GenerateAllDataSets

X_train, weights = GetData('returns')

layers = 3
n_in = X_train.shape[1]
n_latent = 5
n_h = round((n_in + n_latent)/2)

epochs = 500

dist = 'normal'
# dist = 't'

if dist == 'normal':
    model = GaussianVAE(n_in, n_latent, n_h, layers)
elif dist == 't':
    model = StudentsTVAE(n_in, n_latent, n_h, layers)

save_path = os.path.join(os.getcwd(), r'results')

model.fit(X_train, k=1, batch_size=100,
          learning_rate=1e-4, n_epoch=epochs,
          warm_up=False, is_stoppable=False,
          X_valid=X_train, path=save_path)

if dist == 'normal':
    reconstruct, _ = model.reconstruct(X_train)
elif dist == 't':
    _, reconstruct, _ = model.reconstruct(X_train)

print(((X_train - reconstruct)**2).mean())

# check correlation between LVs, LVs and data, LVs and pca, LVs  and reconstructed
decomp = PCA(n_components=n_latent)
decomp.fit(X_train)
PCs = decomp.transform(X_train)

z = model.encode(X_train)[0]

PCs_Z = np.append(z, PCs, axis=1)
X_Z   = np.append(X_train, z, axis=1)
REC_Z = np.append(reconstruct, z, axis=1)
X_REC = np.append(X_train, reconstruct, axis=1)

#%%

sb.heatmap(np.corrcoef(z, rowvar=False))

sb.heatmap(np.corrcoef(X_train, rowvar=False))

sb.heatmap(np.corrcoef(reconstruct, rowvar = False))

sb.heatmap(np.corrcoef(X_Z, rowvar=False))

sb.heatmap(np.corrcoef(X_REC, rowvar=False))

sb.heatmap(np.corrcoef(REC_Z, rowvar=False))

sb.heatmap(np.corrcoef(PCs_Z, rowvar=False))


#%%

plt.plot(range(len(X_Z)), X_Z[:,16] / np.std(X_Z[:,16]))
plt.plot(range(len(X_Z)), X_Z[:,39])
plt.show()
