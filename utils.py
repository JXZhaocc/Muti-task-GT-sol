from __future__ import annotations

import os , csv, pickle
from tqdm import tqdm
import networkx as nx
from torch_geometric.utils import to_networkx
from typing import Tuple, Dict, List
from torch_geometric.data import Data
import torch
from torch.nn import LeakyReLU, ReLU
import torch.nn.functional as F

import numpy as np
import torch, os ,sys
import time
from tqdm import tqdm
import numpy as np
import subprocess, psutil
import pylab as pl
import pandas as pd

def process_hop(sph, gamma, hop, slope=0.1):
    leakyReLU = LeakyReLU(negative_slope=slope)
    sph = sph.unsqueeze(1)
    sph = sph - hop
    sph = leakyReLU(sph)
    sp = torch.pow(gamma, sph)
    return sp


def process_sph(args, data, split=None):
    os.makedirs(f'./sph', exist_ok=True)
    if split is None:
        file = f'./sph/{args.dataset}.pkl'
    else:
        file = f'./sph/{args.dataset}_{split}.pkl'
    if not os.path.exists(file):
        print('pre-process start!')
        progress_bar = tqdm(desc='pre-processing Data', total=len(data), ncols=70)
        for i in range(len(data)):
            data.process(i)
            progress_bar.update(1)
        progress_bar.close()
        pickle.dump(data.sph, open(file, 'wb'))
        print('pre-process down!')
    else:
        data.sph = pickle.load(open(file, 'rb'))
        print('load sph down!')
        
def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0.0:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(torch.nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
