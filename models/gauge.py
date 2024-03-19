from dataclasses import dataclass
from typing import Tuple
from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class Lattice:
    shape: Tuple

    def __post_init__(self):
        self.dof = np.prod(self.shape, dtype=int)
        self.V = self.dof//self.shape[2]

    def idx(self, *args):
        n = len(args)
        return jnp.ravel_multi_index(args, self.shape[-n:], mode='wrap')

    def plaquettes(self):
        index = []
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                index.append([[x, y, 0], [(x+1) % self.shape[0], y, 1],
                             [x, (y+1) % self.shape[1], 0], [x, y, 1]])
        return jnp.array(index)


@dataclass
class U1_2D_OBC:
    geom: Tuple[int]
    beta: float

    def __post_init__(self):
        self.dof = np.prod(self.geom, dtype=int)
        self.shape = self.geom

    def action(self, phi):
        return -self.beta*jnp.cos(phi).sum()

    def observe(self, phi):
        phi = phi.reshape(self.shape)
        # return jnp.array([jnp.prod(jnp.exp(1j*phi[:k, :k])) for k in range(1, self.shape[0]+1)])
        # return jnp.array([jnp.mean(jnp.array([jnp.prod(jnp.exp(1j*(jnp.roll(phi, (i, j), axis=(0, 1))[:k, :k])))
        #                                     for i in range(self.shape[0]) for j in range(self.shape[1])])) for k in range(1, self.shape[0]+1)])
        return jnp.prod(jnp.exp(1j*phi))
        # Move the area and take average, full area
        # obs = jnp.array([jnp.mean(jnp.array([jnp.prod(jnp.exp(1j*(jnp.roll(phi, (i, j), axis=(0, 1))[:k, :k])))
        #                                     for i in range(self.shape[0]) for j in range(self.shape[1])])) for k in range(self.shape[0])])
        return obs


@dataclass
class U1_2D_PBC:
    geom: Tuple[int]
    beta: float

    def __post_init__(self):
        self.shape = (self.geom[0], self.geom[1], 1)

        self.lattice = Lattice(self.shape)
        self.dof = self.lattice.dof
        self.V = self.lattice.V

        self.plaq = self.lattice.plaquettes()
        self.periodic = True

    def plaquette(self, phi):
        # phi = jnp.exp(1j*phi)
        phi = phi.reshape(self.shape1)
        return jax.vmap(lambda y: y[0]*y[1]/y[2]/y[3])(jax.vmap(jax.vmap(lambda y: phi[*y]))(self.plaq))
        x = jnp.array([j[0]*j[1]/j[2]/j[3] for j in [jax.vmap(lambda y: phi[*y])(self.plaq[i])
                      for i in range(self.V)]])
        return x

        x = jnp.array([jax.vmap(lambda y: phi[*y])(self.plaq[i])
                      for i in range(self.V)])
        y = jnp.array([x[i][0]*x[i][1]/x[i][2]/x[i][3] for i in range(self.V)])
        return y

    def action(self, phi):
        return self.beta*jnp.sum(1-jnp.cos(1*phi))

    def observe(self, phi, i):
        x = jnp.exp(1j*phi)
        return jnp.prod(x[:i])

    def action1(self, phi):
        return self.beta*jnp.sum(1-self.plaquette(phi)).real
        for i in range(self.V):
            u1, u2, u3, u4 = jax.vmap(lambda y: phi[*y])(self.plaq[i])
            s += 2.*(u1*u2/u3/u4).real
        return -self.beta*s

    def observe1(self, phi, i):
        x = self.plaquette(phi)
        return jnp.prod(x[:i])
        plaq = self.lattice.plaquettes()

        return jnp.prod(plaq[:i])

        phi = phi.reshape(self.shape)
        return jnp.prod(jnp.exp(1j*phi))

    def plaquette3(self, phi):
        phi = phi.reshape(self.shape)
        return jax.vmap(lambda y: y[0]*y[1]/y[2]/y[3])(jax.vmap(jax.vmap(lambda y: phi[*y]))(self.plaq))
        x = jnp.array([j[0]*j[1]/j[2]/j[3] for j in [jax.vmap(lambda y: phi[*y])(self.plaq[i])
                      for i in range(self.V)]])
        return x

        x = jnp.array([jax.vmap(lambda y: phi[*y])(self.plaq[i])
                      for i in range(self.V)])
        y = jnp.array([x[i][0]*x[i][1]/x[i][2]/x[i][3] for i in range(self.V)])
        return y

    def action3(self, phi):
        return -self.beta*jnp.sum(self.plaquette(phi)).real
        for i in range(self.V):
            u1, u2, u3, u4 = jax.vmap(lambda y: phi[*y])(self.plaq[i])
            s += 2.*(u1*u2/u3/u4).real
        return -self.beta*s

    def observe3(self, phi, i):
        x = self.plaquette(phi)
        return jnp.prod(x[:i])
        plaq = self.lattice.plaquettes()

        return jnp.prod(plaq[:i])

        phi = phi.reshape(self.shape)
        return jnp.prod(jnp.exp(1j*phi))

    def action2(self, phi):
        s = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                s += 2.*(phi[i, j, 0] * phi[(i+1) % self.shape[0], j, 1] /
                         phi[i, (j+1) % self.shape[1], 0]/phi[i, j, 1]).real
        return -self.beta*s

    def observe2(self, phi, i):
        s = 1
        for j in range(i):
            x = j // self.shape[0]
            y = j % self.shape[0]
            s *= phi[x, y, 0] * phi[(x+1) % self.shape[0], y, 1] / \
                phi[x, (y+1) % self.shape[1], 0]/phi[x, y, 1]
        return s
        plaq = self.lattice.plaquettes()

        return jnp.prod(plaq[:i])

        phi = phi.reshape(self.shape)
        return jnp.prod(jnp.exp(1j*phi))


@dataclass
class SU2_2D_OBC:
    geom: Tuple[int]
    g: float

    def __post_init__(self):
        self.dof = np.prod(self.geom, dtype=int)
        self.shape = self.geom

        self.periodic = True

    def action(self, phi):
        phi = phi.reshape([self.dof, 2**2-1])

        return jnp.sum(-4./(self.g**2)*jnp.sin(phi[:, 0])*jnp.cos(phi[:, 1])-jnp.log(jnp.sin(2*phi[:, 0])))

    def observe(self, phi):
        phi = phi.reshape([self.dof, 2**2-1])
        plaq = jnp.array([[jnp.sin(phi[:, 0])*jnp.exp(1j*phi[:, 1]),
                         jnp.cos(phi[:, 0])*jnp.exp(1j*phi[:, 2])], [-jnp.cos(phi[:, 0])*jnp.exp(-1j*phi[:, 2]), jnp.sin(phi[:, 0])*jnp.exp(-1j*phi[:, 1])]]).transpose(2, 0, 1)
        return jnp.trace(reduce(jnp.matmul, plaq))
