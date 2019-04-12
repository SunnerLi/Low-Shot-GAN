import torch.nn as nn
import sys

"""
    This script defines some fundamental function which will be used in other scripts.

    @Author: Cheng-Che Lee
"""

def num2Str(n):
    """
        Fill the number as the same string length

        Arg:    n   (Int)   - The number you want to assign
    """
    if n < 10:
        return '0000' + str(n)
    elif n < 100:
        return '000' + str(n)
    elif n < 1000:
        return '00' + str(n)
    elif n < 10000:
        return '0' + str(n)
    else:
        return str(n)

def weights_init_Xavier(m):
    """
        Initialize the module parameter

        Arg:    m   (torch.nn.Module)   - The module you want to initilize
    """
    if isinstance(m, nn.Conv2d):
        if hasattr(m, 'weight'):
            nn.init.xavier_uniform_(m.weight.data, 1.)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
    elif isinstance(m, nn.Linear):
        if hasattr(m, 'weight'):
            nn.init.xavier_uniform_(m.weight.data, 1.)