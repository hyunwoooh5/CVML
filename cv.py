#!/usr/bin/env python


from models import scalar
from mc import metropolis, replica
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
# Don't print annoying CPU warning.
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)


class MLP(nn.Module):
    volume: int
    features: Sequence[int]
    kernel_init: Callable = nn.initializers.variance_scaling(
        2, "fan_in", "truncated_normal")  # for ReLU / CELU
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat, kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            x = nn.celu(x)
        x = nn.Dense(self.volume, use_bias=False,
                     kernel_init=self.kernel_init)(x)
        return x


class CV_MLP(nn.Module):
    volume: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = MLP(self.volume, self.features)(x)
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
    parser.add_argument('--weight', type=str, default='jnp.ones(len(grads))',
                        help="weight for gradients")
    parser.add_argument('-nt', '--n_train', type=int, default=1000,
                        help="number of training set")
    parser.add_argument('-ns', '--n_test', type=int, default=1000,
                        help="number of test set")

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

    # define the function g
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
        g = CV_MLP(V, [args.width*V]*args.layers)
        g_params = g.init(g_ikey, jnp.zeros(V))

    # define subtraction function
    @jax.jit
    def f(x, p):
        # diagonal sum (Stein's identity)
        j = jax.jacfwd(lambda y: g.apply(p, y))(x)
        return j.diagonal().sum() - g.apply(p, x)@jax.grad(lambda y: model.action(y).real)(x)

        # g: R^V -> R
        # return jnp.sum(jax.grad(lambda x, p: g.apply(p, x)[0], argnums=0)(x, p) - jax.grad(lambda y: model.action(y).real)(x) * g.apply(p, x))

        # All possible sum
        # return j.sum() - jnp.kron(g.apply(p, x), jax.grad(lambda y: model.action(y).real)(x)).sum()

    # regularization, not yet implemented
    def ridge(p):
        sum = 0
        layers = len(p['params']['MLP_0'])-1

        def sum_at_i(i, sum):
            print(type(str(i)))
            print('Dense_'+str(i))
            return sum + (p['params']['MLP_0']['Dense_'+str(i)]['kernel']**2).sum() + (p['params']['MLP_0']['Dense_'+str(i)]['bias']**2).sum()

        sum = jax.lax.fori_loop(0, layers-1, sum_at_i, sum)
        sum += (p['params']['MLP_0']['Dense_' +
                str(layers)]['kernel']**2).sum()

        return sum

    # define loss function
    @jax.jit
    def Loss(x, p):
        return (model.observe(x).real - f(x, p) - mu)**2

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

    def save():
        with open(args.cv, 'wb') as aa:
            pickle.dump((g, g_params), aa)

    def Grad_Mean(grads, weight):
        """
        Params:
            grads: Gradients
            weight: Weights
        """
        grads_w = [jax.tree_util.tree_map(
            lambda x: w*x, g) for w, g in zip(weight, grads)]
        w_mean = jnp.mean(weight)
        grad_mean = jax.tree_util.tree_map(
            lambda *x: jnp.mean(jnp.array(x), axis=0)/w_mean, *grads_w)
        return grad_mean

    def bootstrap(xs, ws=None, N=100, Bs=50):
        if Bs > len(xs):
            Bs = len(xs)
        B = len(xs)//Bs
        if ws is None:
            ws = xs*0 + 1
        # Block
        x, w = [], []
        for i in range(Bs):
            x.append(sum(xs[i*B:i*B+B]*ws[i*B:i*B+B])/sum(ws[i*B:i*B+B]))
            w.append(sum(ws[i*B:i*B+B]))
        x = np.array(x)
        w = np.array(w)
        # Regular bootstrap
        y = x * w
        m = (sum(y) / sum(w))
        ms = []
        for _ in range(N):
            # Using numpy random seed so it might change
            s = np.random.choice(range(len(x)), len(x))
            ms.append((sum(y[s]) / sum(w[s])))
        ms = np.array(ms)
        return m, np.std(ms.real) + 1j*np.std(ms.imag)

    steps = 10000 // args.nstochastic
    grads = [0] * args.nstochastic
    weight = eval(args.weight)

    # measurement
    obs = [0] * args.n_test
    cvs = [0] * args.n_test

    with open(args.cf, 'rb') as aa:  # variable name aa should be different from f
        configs = pickle.load(aa)

    # separate configurations for training and test
    configs_test = configs[:args.n_test]
    configs = configs[-args.n_train:]

    # Unbiased estimation for preventing overfitting
    mu = float(np.mean([model.observe(configs[i])
                        for i in range(args.n_train)]))

    # Training
    while True:
        g_ikey, subkey = jax.random.split(g_ikey)
        rands = jax.random.choice(g_ikey, len(configs), (10000,))
        for s in range(steps):
            for l in range(args.nstochastic):
                grads[l] = Loss_grad(
                    configs[rands[args.nstochastic*s+l]], g_params)

            grad = Grad_Mean(grads, weight)
            updates, opt_state = opt_update_jit(grad, opt_state)
            g_params = optax.apply_updates(g_params, updates)

        for i in range(args.n_test):
            obs[i] = model.observe(configs_test[i])
            cvs[i] = model.observe(configs_test[i]) - \
                f(configs_test[i], g_params)

        print(
            f"{bootstrap(np.array(obs))} {bootstrap(np.array(cvs))}", flush=True)
        save()
