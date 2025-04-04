import time
from CSTGConv2d_KAN_model import *
from sklearn.model_selection import KFold
import torch.nn as nn
from data_load import *
import numpy as np
import argparse
from torch.utils.data import DataLoader
from utils.logger import init_logger
from torch.utils.tensorboard import SummaryWriter
import warnings
import optuna
import plotly
from tslearn.metrics import SoftDTWLossPyTorch

soft_dtw_loss = SoftDTWLossPyTorch(gamma=0.1)  

class CombinedLoss(nn.Module):
        ""
        The program will be uploaded as soon as the article is published
        ""
        return  

def Training_Bayes(trial):
        ""
        The program will be uploaded as soon as the article is published
        ""
    return 
