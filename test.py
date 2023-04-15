import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#import opts
import pdb
import warnings

warnings.filterwarnings('ignore')
import time
from GOOD.data.good_datasets.good_cmnist import GOODCMNIST

from collections import Counter
import torch
import numpy as np
import matplotlib.pyplot as plt

dataset_dg, meta_info = GOODCMNIST.load("dataset",
                                        domain="color",
                                        shift="concept",
                                        generate=False)
print('Dataset information:')
print(dataset_dg['train'].data)
print(meta_info)
print(dataset_dg)


    