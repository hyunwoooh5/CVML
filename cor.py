#!/usr/bin/env python

from models import scalar
import pickle
import jax.numpy as jnp
import jax

import argparse
from cv import *

# Don't print annoying CPU warning.
jax.config.update('jax_platform_name', 'cpu')

parser = argparse.ArgumentParser(
    description="Train g",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    fromfile_prefix_chars='@')
parser.add_argument('model', type=str, help="model filename")
parser.add_argument('cv', type=str, help="cv filename")
parser.add_argument('config', type=str, help="config filename")
parser.add_argument('n_train', type=int)
parser.add_argument('n_test', type=int)


args = parser.parse_args()

with open(args.model, 'rb') as f:
    model=eval(f.read())

with open(args.cv, 'rb') as f:
    g1, g_params = pickle.load(f)

with open(args.config, 'rb') as f:
    conf=pickle.load(f)

V=model.dof
L=model.L
nt=model.NT

index = jnp.array([(-i, -j) for i in range(nt) for j in range(L)])

@jax.jit
def g(x, p):
    def g_(x, p, ind):
        return g1.apply(p, jnp.roll(x.reshape([nt, L]), ind, axis=(0, 1)).reshape(V))[0]
    return jnp.ravel(jax.vmap(lambda ind: g_(x, p, ind))(index).T)

g1_grad = jax.jit(jax.grad(lambda y, p: g1.apply(p, y)[0][0], argnums=0))
dS = jax.jit(jax.grad(lambda y: model.action(y).real))

@jax.jit
def f(x, p):
    # diagonal sum (Stein's identity)
    def diag_(ind):
        return g1_grad(jnp.roll(x.reshape([nt, L]), ind, axis=(0, 1)).reshape(V), p)
    j = jax.vmap(diag_)(index)[:, 0].sum()
    return j - g(x, p)@dS(x)

conf=jnp.array(conf)
obs_full = jax.vmap(model.observe)(conf)

conf_partial=jnp.concatenate([conf[:args.n_test], conf[-args.n_train:]])
cv=jax.vmap(lambda x: f(x, g_params))(conf_partial[:args.n_test])
obs = jax.vmap(model.observe)(conf_partial)

print("nocv")
jack=[0]*nt
cov=[0]*(nt**2)
avs = jax.vmap(jnp.mean)(obs.T)

for i in range(nt):
    jack[i] = jackknife(obs[:,i])
print(f"jackknife:\n{jack}")

print("covariance:")
for i in range(nt):
    for j in range(nt):
        cov[i*nt+j] = (obs[:,i]-avs[i])@(obs[:,j]-avs[j])/(args.n_test+args.n_train-1)/(args.n_test+args.n_train)

for i in range(nt):
    print(f"{jnp.array(cov).reshape([nt,nt])[i]}")

print("nocv large")
jack_full=[0]*nt
cov_full=[0]*(nt**2)
avs_full = jax.vmap(jnp.mean)(obs_full.T)

for i in range(nt):
    jack_full[i] = jackknife(obs_full[:,i], Bs=1000)
print(f"jackknife:\n{jack_full}")

print(f"covariance:")
for i in range(nt):
    for j in range(nt):
        cov_full[i*nt+j] = (obs_full[:,i]-avs_full[i])@(obs_full[:,j]-avs_full[j])/(len(obs_full)-1)/len(obs_full)

for i in range(nt):
    print(f"{jnp.array(cov_full).reshape([nt,nt])[i]}")

print("\ncv")
jack_cv = [0]*nt
cov_cv = [0] * (nt**2)
obs_cv = obs

for i in range(args.n_test):
    obs_cv = obs_cv.at[(i,nt//2)].add(-cv[i])

avs_cv = jax.vmap(jnp.mean)(obs_cv[:args.n_test].T)

for i in range(nt):
    jack_cv[i] = jackknife(obs_cv[:args.n_test,i])
print(f"jackknife:\n{jack_cv}")

for i in range(nt):
    for j in range(nt):
        cov_cv[i*nt+j] = (obs_cv[:args.n_test,i]-avs_cv[i])@(obs_cv[:args.n_test,j]-avs_cv[j])/(args.n_test-1)/args.n_test
print(f"covariance:")
for i in range(nt):
    print(f"{jnp.array(cov_cv).reshape([nt,nt])[i]}")

print("\ncv all")
jack_cv_all = [0]*nt
cov_cv_all = [0]* (nt**2)
obs_cv_all = obs[:args.n_test] - jnp.concatenate([cv]*nt).reshape([nt, args.n_test]).T
avs_cv_all = jax.vmap(jnp.mean)(obs_cv_all.T)

for i in range(nt):
    jack_cv_all[i] = jackknife(obs_cv_all[:,i])
print(f"jackknife:\n{jack_cv_all}")

for i in range(nt):
    for j in range(nt):
        cov_cv_all[i*nt+j] = (obs_cv_all[:,i]-avs_cv_all[i])@(obs_cv_all[:,j]-avs_cv_all[j])/(args.n_test-1)/args.n_test
print(f"covariance:")
for i in range(nt):
    print(f"{jnp.array(cov_cv_all).reshape([nt,nt])[i]}")

