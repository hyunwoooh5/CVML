#!/usr/bin/env python

import pickle
import argparse
import ast
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name","cpu")

parser = argparse.ArgumentParser(
    description="""Define geometry""",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    fromfile_prefix_chars='@'
)

parser.add_argument('geom', type=str, help='geometry')
parser.add_argument('input', type=str, help='text file')
parser.add_argument('output', type=str, help='pickle file')
parser.add_argument('-n', '--n_config', type=int,
                    default=-1, help='number of configs')
parser.add_argument('-s', '--shuffle', type=bool,
                    default=False, help="shuffle the Monte Carlo chain fully (for U(1) OBC)")

args = parser.parse_args()

GEOM = ast.literal_eval(args.geom)
dof = np.prod(GEOM, dtype=int)

with open(args.input, 'rb') as f:
    conf = jnp.array([float(x) for x in f.readlines()])

n_conf = int(len(conf)/dof)

conf = conf.reshape([n_conf, dof])

if args.shuffle:
    key = jax.random.PRNGKey(0)
    conf = jax.random.permutation(key, conf.ravel())
    conf = conf.reshape([n_conf, dof])

if args.n_config != -1:
    conf = conf[:args.n_config]

with open(args.output, 'wb') as f:
    pickle.dump(conf, f)
