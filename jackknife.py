#!/usr/bin/env python

import numpy as np
import sys
from cmath import phase


def jackknife(xs, ws=None, Bs=50):  # Bs: Block size
    B = len(xs)//Bs  # number of blocks
    if ws is None:  # for reweighting
        ws = xs*0 + 1
    # Block
    '''
    x, w = [], []
    for i in range(Bs):
        x.append(sum(xs[i*B:i*B+B]*ws[i*B:i*B+B])/sum(ws[i*B:i*B+B]))
        w.append(sum(ws[i*B:i*B+B]))
    x = np.array(x)
    w = np.array(w)
    '''
    x = np.array(xs[:B*Bs])
    w = np.array(ws[:B*Bs])

    m = np.mean(x*w)/np.mean(w)

    # partition
    block = [[B, Bs], list(x.shape[1:])]
    block_f = [i for sublist in block for i in sublist]
    x = x.reshape(block_f)
    w = w.reshape(block_f)

    # jackknife
    vals = [np.mean(np.delete(x, i, axis=0)*np.delete(w, i, axis=0)) /
            np.mean(np.delete(w, i)) for i in range(B)]
    vals = np.array(vals)

    return m, (np.std(vals.real) + 1j*np.std(vals.imag))*(np.sqrt(len(vals))-1)


lines = [l for l in sys.stdin.readlines() if l[0] != '#']
dat = np.array([[complex(x) for x in l.split()] for l in lines if l[0] != '#'])

boltz = dat[:, 0]  # reweighting factor
dat = dat[:, 1:]  # Others

# Reweighting (equivalent to sign problem)
rew, rew_err = jackknife(boltz)
print(rew, rew_err)
print(f'# Reweighting: {rew} {rew_err} {abs(rew)} {phase(rew)}')

# Observables
for i in range(dat.shape[1]):
    print(*jackknife(dat[:, i], boltz))
