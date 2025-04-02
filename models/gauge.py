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
        self.V = self.dof//self.shape[-1]

    def idx(self, *args):
        n = len(args)
        return jnp.ravel_multi_index(args, self.shape[-n:], mode='wrap')

    def plaquettes_2d(self):
        index = []
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                index.append([[x, y, 0], [(x+1) % self.shape[0], y, 1],
                             [x, (y+1) % self.shape[1], 0], [x, y, 1]])
        return jnp.array(index)

    def plaquettes_3d(self):
        index = []
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                for z in range(self.shape[2]):
                    index.append([[x, y, z, 0], [(x+1) % self.shape[0], y, z, 1],
                                  [x, (y+1) % self.shape[1], z, 0], [x, y, z, 1]])
                    index.append([[x, y, z, 1], [x, (y+1) % self.shape[1], z, 2],
                                  [x, y, (z+1) % self.shape[2], 1], [x, y, z, 2]])
                    index.append([[x, y, z, 2], [x, y, (z+1) % self.shape[2], 0],
                                  [(x+1) % self.shape[0], y, z, 2], [x, y, z, 0]])

        # Order: xy plaq, yz plaq, zx plaq for each lattice point
        return jnp.array(index).reshape([self.V, 3, 4, 4]).transpose([1, 0, 2, 3]).reshape([self.dof, 4, 4])


@dataclass
class U1_2D_OBC:
    geom: Tuple[int]
    beta: float

    def __post_init__(self):
        self.dof = np.prod(self.geom, dtype=int)
        self.shape = (self.geom[0], self.geom[1], 1)

    def action(self, phi):
        return -self.beta*jnp.cos(phi).sum()

    def observe(self, phi, i):
        # phi = phi.reshape(self.shape)
        # return jnp.array([jnp.prod(jnp.exp(1j*phi[:k, :k])) for k in range(1, self.shape[0]+1)])
        # return jnp.array([jnp.mean(jnp.array([jnp.prod(jnp.exp(1j*(jnp.roll(phi, (i, j), axis=(0, 1))[:k, :k])))
        #                                     for i in range(self.shape[0]) for j in range(self.shape[1])])) for k in range(1, self.shape[0]+1)])
        return jnp.prod(jnp.exp(1j*phi[:i]))
        # Move the area and take average, full area
        # obs = jnp.array([jnp.mean(jnp.array([jnp.prod(jnp.exp(1j*(jnp.roll(phi, (i, j), axis=(0, 1))[:k, :k])))
        #                                     for i in range(self.shape[0]) for j in range(self.shape[1])])) for k in range(self.shape[0])])
        return obs


@dataclass
class U1_2D_PBC:
    geom: Tuple[int]
    beta: float

    def __post_init__(self):
        self.shape = (self.geom[0], self.geom[1], 2)

        self.lattice = Lattice(self.shape)
        self.dof = self.lattice.dof
        self.V = self.lattice.V

        self.plaq = self.lattice.plaquettes_2d()

    def plaquette(self, phi):
        phi = jnp.exp(1j*phi)
        phi = phi.reshape(self.shape)
        return jax.vmap(lambda y: y[0]*y[1]/y[2]/y[3])(jax.vmap(jax.vmap(lambda y: phi[*y]))(self.plaq))

    def actionl(self, phi):
        return self.beta*jnp.sum(1-self.plaquette(phi)).real

    def observel(self, phi, i):
        x = self.plaquette(phi)
        return jnp.prod(x[:i])

    def action(self, phi):
        return self.beta*jnp.sum(1-jnp.cos(phi))

    def observe(self, phi, i):
        x = jnp.exp(1j*phi)
        return jnp.prod(x[:i])


@dataclass
class SU2_2D_OBC_Bronzan:
    geom: Tuple[int]
    g: float

    def __post_init__(self):
        self.dof = np.prod(self.geom, dtype=int)
        self.shape = self.geom

    def action(self, phi):
        phi = phi.reshape([self.dof, 2**2-1])

        return jnp.sum(-4./(self.g**2)*jnp.sin(phi[:, 0])*jnp.cos(phi[:, 1])-jnp.log(jnp.sin(2*phi[:, 0])))

    def observe(self, phi, i):
        phi = phi.reshape([self.dof, 2**2-1])
        plaq = jnp.array([[jnp.sin(phi[:, 0])*jnp.exp(1j*phi[:, 1]),
                         jnp.cos(phi[:, 0])*jnp.exp(1j*phi[:, 2])], [-jnp.cos(phi[:, 0])*jnp.exp(-1j*phi[:, 2]), jnp.sin(phi[:, 0])*jnp.exp(-1j*phi[:, 1])]]).transpose(2, 0, 1)
        return 0.5*jnp.trace(reduce(jnp.matmul, plaq[:i]))
        return reduce(jnp.matmul, plaq[:i])[0, 0]


@dataclass
class SU2_2D_OBC_Euler:
    geom: Tuple[int]
    g: float

    def __post_init__(self):
        self.dof = np.prod(self.geom, dtype=int)
        self.shape = self.geom

    def action(self, phi):
        phi = phi.reshape([self.dof, 2**2-1])

        return jnp.sum(-4./(self.g**2)*jnp.cos(phi[:, 0]/2) - jnp.log(jnp.sin(phi[:, 0]/2)**2 * jnp.sin(phi[:, 1])))

    def observe(self, phi, i):
        phi = phi.reshape([self.dof, 2**2-1])

        plaq = jnp.array([[jnp.cos(phi[:, 0]/2) + 1j*jnp.sin(phi[:, 0]/2) * jnp.cos(phi[:, 1]), jnp.sin(phi[:, 0]/2) * jnp.sin(phi[:, 1]) * (1j * jnp.cos(phi[:, 2]) + jnp.sin(phi[:, 2]))], [jnp.sin(phi[:, 0]/2) * jnp.sin(phi[:, 1]) * (1j * jnp.cos(
            phi[:, 2]) - jnp.sin(phi[:, 2])), jnp.cos(phi[:, 0]/2) - 1j*jnp.sin(phi[:, 1]) * jnp.cos(phi[:, 1])]]).transpose(2, 0, 1)
        return 0.5*jnp.trace(reduce(jnp.matmul, plaq[:i]))
        return reduce(jnp.matmul, plaq[:i])[0, 0]


@dataclass
class U1_3D_PBC:
    geom: Tuple[int]
    beta: float

    def __post_init__(self):
        self.shape = (self.geom[0], self.geom[1], self.geom[2], 3)

        self.lattice = Lattice(self.shape)
        self.dof = self.lattice.dof
        self.V = self.lattice.V

        self.plaq = self.lattice.plaquettes_3d()

    def plaquette(self, phi):
        phi = jnp.exp(1j*phi)
        phi = phi.reshape(self.shape)
        return jax.vmap(lambda y: y[0]*y[1]/y[2]/y[3])(jax.vmap(jax.vmap(lambda y: phi[*y]))(self.plaq))

    def action(self, phi):
        return self.beta*jnp.sum(1-self.plaquette(phi)).real

    def wilsonloop12(self, phi, i):
        # first 3: direction of plaquettes xy, yz, zx # lattice increases with z -> y -> x
        x = self.plaquette(phi).reshape([3, self.V])
        return jnp.prod(x[1, :i])

    def wilsonloop01(self, phi, i):
        # first 3: direction of plaquettes xy, yz, zx # lattice increases with z -> y -> x
        x = self.plaquette(phi).reshape([3, self.V])[0]
        x = x.reshape(self.shape[:-1]).transpose([2, 0, 1]).reshape(self.V)
        return jnp.prod(x[:i])

    def wilsonloop20(self, phi, i):
        # first 3: direction of plaquettes xy, yz, zx # lattice increases with z -> y -> x
        x = self.plaquette(phi).reshape([3, self.V])[2]
        x = x.reshape(self.shape[:-1]).transpose([1, 2, 0]).reshape(self.V)
        return jnp.prod(x[:i])

    # z direction is the time direction in this convention
    def correlation(self, phi, i, av):
        pl = self.plaquette(phi).reshape(self.shape[-1:]+self.shape[:-1])
        o = jnp.mean(pl[0], axis=(0, 1))  # plaquettes on xy-plane
        return jnp.sum(jnp.roll(o-av, -i) * (o-av))

    def plaq_av(self, phi):
        pl = self.plaquette(phi).reshape(self.shape[-1:]+self.shape[:-1])
        return jnp.mean(pl[0], axis=(0, 1))  # plaquettes on xy-plane
