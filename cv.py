#!/usr/bin/env python

from models import scalar, gauge
import pickle
import sys
import time
from typing import Callable, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from util import *
from itertools import product

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)


@jax.jit
def arcsinh(x: any) -> any:
    return jnp.arcsinh(x)


@jax.jit
def sinh(x: any) -> any:
    return jnp.sinh(x)


class MLP(nn.Module):
    volume: int
    features: Sequence[int]
    kernel_init: Callable = nn.initializers.variance_scaling(
        2, "fan_in", "truncated_normal")  # for ReLU / CELU
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat, use_bias=False,
                         kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            x = arcsinh(x)
        x = nn.Dense(1, use_bias=False,
                     kernel_init=self.kernel_init)(x)
        return x


class CV_MLP(nn.Module):
    volume: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = MLP(self.volume, self.features)(x)
        y = self.param('bias', nn.initializers.zeros, (1,))
        return x, y


class CV_MLP_Periodic(nn.Module):
    volume: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        input = jnp.ravel(jnp.array([jnp.sin(x), jnp.cos(x)]))

        x = MLP(self.volume, self.features)(input)
        y = self.param('bias', nn.initializers.zeros, (1,))
        return x, y


class CNN(nn.Module):
    volume: int
    features: Sequence[int]
    kernel_init: Callable = nn.initializers.variance_scaling(
        2, "fan_in", "truncated_normal")  # for ReLU / CELU
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), use_bias=False, kernel_init=self.kernel_init,
                        bias_init=self.bias_init, padding='CIRCULAR')(x)  # Periodic boundary
            x = arcsinh(x)
        x = nn.Conv(1, kernel_size=(3, 3), use_bias=False, kernel_init=self.kernel_init,
                    bias_init=self.bias_init, padding='CIRCULAR')(x)

        y = self.param('bias', nn.initializers.zeros, (1,))
        return jnp.ravel(x), y


class CV_CNN(nn.Module):
    volume: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, x, shape):
        x = x.reshape(shape)
        x = CNN(self.volume, self.features)(x)
        return x


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Train g",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument('model', type=str, help="model filename")
    parser.add_argument('cv', type=str, help="cv filename")
    parser.add_argument('cf', type=str, help="configurations file name")
    parser.add_argument('-i', '--init', action='store_true',
                        help="re-initialize even if cv already exists")
    parser.add_argument('-f', '--from', dest='fromfile', type=str,
                        help="initialize from other file")
    parser.add_argument('-l', '--layers', type=int, default=0,
                        help='number of (hidden) layers')
    parser.add_argument('-w', '--width', type=int, default=1,
                        help='width (scaling)')
    parser.add_argument('-r', '--replica', action='store_true',
                        help="use replica exchange")
    parser.add_argument('-nrep', '--nreplicas', type=int, default=30,
                        help="number of replicas (with -r)")
    parser.add_argument('-maxh', '--max-hbar', type=float, default=10.,
                        help="maximum hbar (with -r)")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--seed-time', action='store_true',
                        help="seed PRNG with current time")
    parser.add_argument('--dp', action='store_true',
                        help="turn on double precision")
    parser.add_argument('-lr', '--learningrate', type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument('-N', '--nstochastic', default=1, type=int,
                        help="number of samples to estimate gradient")
    parser.add_argument('-o',  '--optimizer', choices=['adam', 'sgd', 'yogi'], default='adam',
                        help='optimizer to use')
    parser.add_argument('-s', '--schedule', action='store_true',
                        help="scheduled learning rate")
    parser.add_argument('-C', '--care', type=float, default=1,
                        help='scaling for learning schedule')
    parser.add_argument('--b1', type=float, default=0.9,
                        help="b1 parameter for adam")
    parser.add_argument('--b2', type=float, default=0.999,
                        help="b2 parameter for adam")
    parser.add_argument('-nt', '--n_train', type=int, default=1000,
                        help="number of training set")
    parser.add_argument('-ns', '--n_test', type=int, default=1000,
                        help="number of test set")
    parser.add_argument('--l2', type=float, default=0.0,
                        help="l2 regularization")
    parser.add_argument('-opt', '--optuna', action='store_true',
                        help="Use optuna")
    parser.add_argument('--cnn',  action='store_true',
                        help="Use CNN")

    args = parser.parse_args()

    if args.dp:
        jax.config.update('jax_enable_x64', True)

    seed = args.seed
    if args.seed_time:
        seed = time.time_ns()
    key = jax.random.PRNGKey(seed)

    with open(args.model, 'rb') as f:
        model = eval(f.read())
    V = model.dof

    g_ikey, chain_key = jax.random.split(key, 2)

    # define control variates
    loaded = False
    if not args.init and not args.fromfile:
        try:
            with open(args.cv, 'rb') as f:
                g, g_params = pickle.load(f)
            loaded = True
        except FileNotFoundError:
            pass
    if args.fromfile:
        with open(args.fromfile, 'rb') as f:
            g, g_params = pickle.load(f)
        loaded = True
    if not loaded:
        if args.cnn:
            shape = list(model.shape)
            shape.append(1)

            g = CV_CNN(V, [args.width]*args.layers)
            g_params = g.init(g_ikey, jnp.zeros(V), shape)

            dS = jax.jit(jax.grad(lambda y: model.action(y).real))
            j = jax.jit(jax.jacfwd(
                lambda x, p: g.apply(p, x, shape)[0], argnums=0))

            @jax.jit
            def f(x, p):
                dg = jnp.trace(j(x, p))
                ds = dS(x)
                gx, _ = g.apply(p, x, shape)

                return dg - gx@ds

            # define loss function
            @jax.jit
            def Loss(x, p):
                _, y = g.apply(p, x, shape)

                # shift is not regularized
                return jnp.abs(model.observe(x) - f(x, p) - y[0])**2 + sum(l2_loss(w, alpha=args.l2) for w in jax.tree_util.tree_leaves(p["params"])) - args.l2 * y[0]**2

            def save():
                with open(args.cv, 'wb') as aa:
                    pickle.dump((g, g_params), aa)

        else:
            if model.periodic:
                g1 = CV_MLP_Periodic(V, [args.width]*args.layers)
                g_params = g1.init(g_ikey, jnp.zeros(V))
            else:
                g1 = CV_MLP(V, [args.width]*args.layers)
                g_params = g1.init(g_ikey, jnp.zeros(V))

            # g(Tx) = Tg(x)
            index = jnp.array(
                [(-i, -j) for i, j in product(*list(map(lambda y: range(y), model.shape)))])

            @jax.jit
            def g(x, p):
                def g_(x, p, ind):
                    return g1.apply(p, jnp.roll(x.reshape(model.shape), ind, axis=(0, 1)).reshape(V))[0]
                return jnp.ravel(jax.vmap(lambda ind: g_(x, p, ind))(index).T)

            g1_grad = jax.jit(
                jax.grad(lambda y, p: g1.apply(p, y)[0][0], argnums=0))
            dS = jax.jit(jax.grad(lambda y: model.action(y).real))
            jaco = jax.jit(jax.jacfwd(g, argnums=0))

            # define subtraction function
            @jax.jit
            def f(x, p):
                # diagonal sum (Stein's identity)
                def diag_(ind):
                    return g1_grad(jnp.roll(x.reshape(model.shape), ind, axis=(0, 1)).reshape(V), p)
                j = jax.vmap(diag_)(index)[:, 0].sum()
                return j - g(x, p)@dS(x)

            # define loss function
            @jax.jit
            def Loss(x, p):
                _, y = g1.apply(p, x)

                # shift is not regularized
                return jnp.abs(model.observe(x) - f(x, p) - y[0])**2 + sum(l2_loss(w, alpha=args.l2) for w in jax.tree_util.tree_leaves(p["params"])) - args.l2 * y[0]**2

            def save():
                with open(args.cv, 'wb') as aa:
                    pickle.dump((g1, g_params), aa)

    Loss_grad = jax.jit(jax.grad(lambda x, p: Loss(x, p), argnums=1))

    if args.schedule:
        sched = optax.exponential_decay(
            init_value=args.learningrate,
            transition_steps=int(args.care),
            decay_rate=0.99,
            end_value=1e-6)
    else:
        sched = optax.constant_schedule(args.learningrate)
    opt = getattr(optax, args.optimizer)(sched, args.b1, args.b2)
    opt_state = opt.init(g_params)
    opt_update_jit = jax.jit(opt.update)

    # measurement
    with open(args.cf, 'rb') as aa:  # variable name aa should be different from f
        configs = pickle.load(aa)

    # separate configurations for training and test
    configs = jnp.array(configs)
    configs_test = configs[:args.n_test]
    configs = configs[-args.n_train:]

    obs = jax.vmap(model.observe)(configs_test)
    obs_av = jackknife(np.array(obs))

    # Training
    while True:
        g_ikey, subkey = jax.random.split(g_ikey)
        configs = jax.random.permutation(key, configs)
        for s in range(args.n_train//args.nstochastic):  # one epoch
            grads = jax.vmap(lambda y: Loss_grad(y, g_params))(
                configs[args.nstochastic*s: args.nstochastic*(s+1)])

            grad = jax.tree_util.tree_map(
                lambda x: jnp.mean(x, axis=0), grads)
            updates, opt_state = opt_update_jit(grad, opt_state)
            g_params = optax.apply_updates(g_params, updates)

        fs = jax.vmap(lambda x: f(x, g_params))(configs_test)

        print(
            f"{obs_av} {jackknife(np.array(obs-fs))} {jackknife(np.array(fs))}", flush=True)
        save()
