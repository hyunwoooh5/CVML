#!/usr/bin/env python

import argparse
import itertools
import pickle
import sys
import time
from typing import Callable, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
# Don't print annoying CPU warning.
jax.config.update('jax_platform_name', 'cpu')

from mc import metropolis,replica
from cv import *

parser = argparse.ArgumentParser(description="Print network parameters")
parser.add_argument('cv', type=str, help="cv filename")
args = parser.parse_args()

with open(args.cv, 'rb') as f:
    g, g_params = pickle.load(f)  


print("cv params:", g_params)
print("\n\n")
print("cv structure:", jax.tree_util.tree_map(lambda x: x.shape, g_params))

print(g)

def l2(x):
    return (x**2).mean()

def square(p):
    _, y = g.apply(p, jnp.zeros(g.volume))
    
    return sum(l2(w) for w in jax.tree_util.tree_leaves(p["params"]))-y[0]**2, y[0] 

gradsquare, y = square(g_params)

print(f'sum of squares of paramters: {gradsquare}')
print(f'mu: {y}')
