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
    features: Sequence[int]
    kernel_init: Callable = nn.initializers.variance_scaling(
        2, "fan_in", "truncated_normal") # for ReLU
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat, kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            x = nn.relu(x)
        x = nn.Dense(1, kernel_init=self.kernel_init,
                     bias_init=self.bias_init)(x)
        return x


class CV_MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        u = MLP(self.features)(x)
        return u


class Naive(nn.Module):
    def __call__(self, x):
        return jnp.array([0.])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Train g",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument('model', type=str, help="model filename")
    parser.add_argument('cv', type=str, help="cv filename")
    parser.add_argument('-R', '--real', action='store_true',
                        help="output the real plane")
    parser.add_argument('-i', '--init', action='store_true',
                        help="re-initialize even if contour already exists")
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
    parser.add_argument('-lr', '--learningrate', type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument('-S', '--skip', default=30, type=int,
                        help='number of steps to skip')
    parser.add_argument('-N', '--nstochastic', default=1, type=int,
                        help="number of samples to estimate gradient")
    parser.add_argument('-T', '--thermalize', default=0, type=int,
                        help="number of MC steps (* d.o.f) to thermalize")
    parser.add_argument('-Nt', '--tsteps', default=10000000, type=int,
                        help="number of training")
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

    args = parser.parse_args()

    seed = args.seed
    if args.seed_time:
        seed = time.time_ns()
    key = jax.random.PRNGKey(seed)

    with open(args.model, 'rb') as f:
        model = eval(f.read())
    V = model.dof

    skip = args.skip
    if args.skip == 30:
        skip = V

    g_ikey, chain_key = jax.random.split(key, 2)

    # define the function g
    if args.real:
        # Output real plane and quit
        g = Naive()
        g_params = g.init(g_ikey, jnp.zeros(V))
        with open(args.cv, 'wb') as f:
            pickle.dump((g, g_params), f)
        sys.exit(0)

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
        g = CV_MLP([args.width*V]*args.layers)
        g_params = g.init(g_ikey, jnp.zeros(V))

    # setup metropolis
    if args.replica:
        chain = replica.ReplicaExchange(model.action, jnp.zeros(
            V), chain_key, delta=1./jnp.sqrt(V), max_hbar=args.max_hbar, Nreplicas=args.nreplicas)
    else:
        chain = metropolis.Chain(model.action, jnp.zeros(
            V), chain_key, delta=1./jnp.sqrt(V))

    # define subtraction function
    @jax.jit
    def f(x, p):
        return (jax.grad(lambda x, p: g.apply(p, x)[0],
                argnums=0)(x, p)[0] - jax.grad(lambda y: model.action(y).real)(x)[0] * g.apply(p, x))[0]

    # define loss function
    @jax.jit
    def Loss(x, p):
        return - 2. * model.observe(x).real * f(x, p) + f(x, p)**2

    Loss_grad = jax.jit(jax.grad(lambda x, p: Loss(x, p), argnums=1))

    if args.schedule:
        sched = optax.exponential_decay(
            init_value=args.learningrate,
            transition_steps=int(args.care*1000),
            decay_rate=0.99,
            end_value=2e-5)
    else:
        sched = optax.constant_schedule(args.learningrate)
    opt = getattr(optax, args.optimizer)(sched, args.b1, args.b2)
    opt_state = opt.init(g_params)
    opt_update_jit = jax.jit(opt.update)

    def save():
        with open(args.cv, 'wb') as f:
            pickle.dump((g, g_params), f)

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
            s = np.random.choice(range(len(x)), len(x))
            ms.append((sum(y[s]) / sum(w[s])))
        ms = np.array(ms)
        return m, np.std(ms.real) + 1j*np.std(ms.imag)

    steps = int(10000 / args.nstochastic)
    grads = [0] * args.nstochastic
    weight = eval(args.weight)

    # measurement
    obs = [0] * 10000
    cvs = [0] * 10000

    chain.calibrate()
    chain.step(N=args.thermalize*V)
    try:
        for t in range(args.tsteps):
            for s in range(steps):
                chain.calibrate()
                for l in range(args.nstochastic):
                    chain.step(N=skip)
                    grads[l] = Loss_grad(chain.x, g_params)  # gradient descent

                grad = Grad_Mean(grads, weight)
                updates, opt_state = opt_update_jit(grad, opt_state)
                g_params = optax.apply_updates(g_params, updates)

            '''
            # tracking the size of gradient
            grad_abs = 0.
        
            grad_abs += np.linalg.norm(grad['params']
                                       ['Dense_'+str(0)]['kernel'])
            grad_abs += np.linalg.norm(grad['params']
                                           ['Dense_'+str(0)]['bias'])

                for j in range(args.layers):
                    grad_abs += np.linalg.norm(grad['params']
                                               ['MLP_'+str(0)]['Dense_'+str(j)]['kernel'])
                    grad_abs += np.linalg.norm(grad['params']
                                               ['MLP_'+str(0)]['Dense_'+str(j)]['bias'])
            '''

            # measurement once in a while
            for i in range(len(obs)):
                chain.step(N=skip)
                obs[i] = model.observe(chain.x).real
                cvs[i] = model.observe(chain.x).real - f(chain.x, g_params)

            # print(f'{np.mean(phases).real} {np.abs(np.mean(phases))} {bootstrap(np.array(phases))} ({np.mean(np.abs(chain.x))} {np.real(np.mean(acts))} {np.mean(acts)} {grad_abs} {chain.acceptance_rate()})', flush=True)
            print(f'{bootstrap(np.array(obs))} {bootstrap(np.array(cvs))} ({np.mean(np.abs(chain.x))} {chain.acceptance_rate()})', flush=True)

            save()

    except KeyboardInterrupt:
        print()
        save()
