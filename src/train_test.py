from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import time
import torch
import torch.nn as nn
import optimizers
from Config_File import Config
import numpy as np

criterion = nn.CrossEntropyLoss()

def transpose_batch(batch_data):
    ret={}
    for key in batch_data:
        ret[key]=batch_data[key].t()
    return ret

