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

from datasets import load_dataset
from models import GaussianVAE, StudentsTVAE
from data.datafuncs import GetData, GenerateAllDataSets

X_train, weights = GetData('returns')

n_in = X_train.shape[1]
n_latent = 10
n_h = round((n_in + n_latent)/2)

dist = 'normal'
dist = 't'

if dist == 'normal':
    model = GaussianVAE(n_in, n_latent, n_h)

save_path = os.path.join(os.getcwd(), r'results')

model.fit(X_train, k=1, batch_size=100,
          learning_rate=1e-4, n_epoch=500,
          warm_up=False, is_stoppable=True,
          X_valid=X_train, path=save_path)


z, _ = model.encode(X_train)

reconstruct, _ = model.decode(z)

print(((X_train - reconstruct)**2).mean())

