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
import sympy
import optax
import optuna
from util import *

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
        input = jnp.sin(x)
        input = jnp.ravel(jnp.array([jnp.cos(x), jnp.sin(x)]))

        x1 = MLP(self.volume, self.features)(input)
        x2 = MLP(self.volume, self.features)(input)
        y = self.param('bias', nn.initializers.zeros, (1,))
        return x1, x2, y


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
    length: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x.reshape(self.length, self.length, 1)
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
    nt, L = model.shape

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
            g = CV_CNN(V, L, [args.width]*args.layer)
            g_params = g.init(g_ikey, jnp.zeros(V))

            dS = jax.jit(jax.grad(lambda y: model.action(y).real))
            j = jax.jit(jax.jacfwd(lambda x, p: g.apply(p, x)[0], argnums=0))

            @jax.jit
            def f(x, p):
                d_g = jnp.trace(j(x, p))
                d_act = dS(x)
                # x = x.reshape([nt, L])
                g_x, _ = g.apply(p, x)

                return d_g - g_x@d_act

            # define loss function
            @jax.jit
            def Loss(x, p):
                _, y = g.apply(p, x)

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
            index = jnp.array([(-i, -j) for i in range(nt) for j in range(L)])

            @jax.jit
            def g(x, p):
                def g_(x, p, ind):
                    return g1.apply(p, jnp.roll(x.reshape([nt, L]), ind, axis=(0, 1)).reshape(V))[0]
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
                    return g1_grad(jnp.roll(x.reshape([nt, L]), ind, axis=(0, 1)).reshape(V), p)
                j = jax.vmap(diag_)(index)[:, 0].sum()
                return j - g(x, p)@dS(x)

                # j = jaco(x, p)
                # return j.diagonal().sum() - g(x, p)@dS(x)

                # g: R^V -> R
                # return jnp.sum(jax.grad(lambda x, p: g.apply(p, x)[0], argnums=0)(x, p) - jax.grad(lambda y: model.action(y).real)(x) * g.apply(p, x))

                # All possible sum
                # return j.sum() - jnp.kron(g.apply(p, x), jax.grad(lambda y: model.action(y).real)(x)).sum()
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

    def objective(trial):
        layers = trial.suggest_int('layers', 1, 5)
        width = trial.suggest_int('width', 2, V//4, step=2)

        l2 = trial.suggest_float('l2', 0, 0.2)
        lr = trial.suggest_float('lr', 1e-4, 1e-3)
        end = trial.suggest_float('end', 1e-8, 1e-6)
        b1 = trial.suggest_float('b1', 0.9, 1)
        b2 = trial.suggest_float('b2', 0.9, 1)
        care = trial.suggest_int('care', 100, 10000, step=100)
        N = trial.suggest_categorical('N', sympy.divisors(100))

        g_ikey = jax.random.PRNGKey(0)

        g1 = CV_MLP(V, L, [width]*layers)
        g_params = g1.init(g_ikey, jnp.zeros(V))

        @jax.jit
        def g(x, p):
            def g_(x, p, ind):
                return g1.apply(p, jnp.roll(x.reshape([nt, L]), ind, axis=(0, 1)).reshape(V))[0]
            return jnp.ravel(jax.vmap(lambda ind: g_(x, p, ind))(index).T)

        @jax.jit
        def f(x, p):
            j = jax.jacfwd(lambda y: g(y, p))(x)
            return j.diagonal().sum() - g(x, p)@jax.grad(lambda y: model.action(y).real)(x)

        @jax.jit
        def Loss(x, p):
            _, y = g1.apply(p, x)
            # shift is not regularized
            return jnp.abs(model.observe(x) - f(x, p) - y[0])**2 + sum(l2_loss(w, alpha=l2) for w in jax.tree_util.tree_leaves(p["params"])) - l2 * y[0]**2

        Loss_grad = jax.jit(jax.grad(lambda x, p: Loss(x, p), argnums=1))

        sched = optax.exponential_decay(
            init_value=lr,
            transition_steps=care,
            decay_rate=0.99,
            end_value=end)

        opt = getattr(optax, 'adam')(sched, b1, b2)
        opt_state = opt.init(g_params)
        opt_update_jit = jax.jit(opt.update)

        for step in range(500):  # 500 epochs
            for s in range(args.n_train//N):
                grads = jax.vmap(lambda y: Loss_grad(y, g_params))(
                    configs[N*s: N*(s+1)])

                grad = jax.tree_util.tree_map(
                    lambda x: jnp.mean(x, axis=0), grads)
                updates, opt_state = opt_update_jit(grad, opt_state)
                g_params = optax.apply_updates(g_params, updates)

            fs = jax.vmap(lambda x: f(x, g_params))(configs_test)

            ob, err = jackknife(np.array(obs)-np.array(fs))
            intermediate_value = err.real

            trial.report(intermediate_value, step)

            if trial.should_prune():
                raise optuna.TrialPruned()

        fs = jax.vmap(lambda x: f(x, g_params))(configs_test)

        ob, err = jackknife(np.array(obs-fs))

        return err.real
    '''
    from flax.training import train_state

    class TrainState(train_state.TrainState):
        key: jax.Array

    if args.optuna:
        study = optuna.create_study(
            study_name=args.cv, direction='minimize', sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=200, interval_steps=1, n_min_trials=5))  # single-objective optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=10)
        print("Best Score:", study.best_value)
        print("Best trial:", study.best_trial.params)
        print("Parameter importance: ",
              optuna.importance.get_param_importances(study))
    else:
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
