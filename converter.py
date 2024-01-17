#!/usr/bin/env python

import pickle
import argparse
import ast
import jax
import jax.numpy as jnp
import numpy as np


parser = argparse.ArgumentParser(
    description="""Define geometry""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    fromfile_prefix_chars='@'
)

parser.add_argument('geom', type=str, help='geometry')
parser.add_argument(
    'dof', type=int, help='degree of freedom for each configuration')
parser.add_argument('input', type=str, help='text file')
parser.add_argument('output', type=str, help='pickle file')
parser.add_argument('-n', '--nconfig', type=int,
                    default=-1, help='number of configs')

args = parser.parse_args()

GEOM = ast.literal_eval(args.geom)
n_lat = np.prod(GEOM, dtype=int)

with open(args.input, 'rb') as f:
    conf = jnp.array([float(x) for x in f.readlines()])

n = len(conf)//args.dof
n_conf = n//n_lat

conf = conf.reshape([n, args.dof])

key = jax.random.PRNGKey(0)
conf_shuffled = jax.random.permutation(key, conf)

conf_shuffled = conf_shuffled.reshape([n_conf, args.dof*n_lat])

with open(args.output, 'wb') as f:
    pickle.dump(conf_shuffled[:args.nconfig], f)
