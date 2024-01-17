from dataclasses import dataclass
from typing import Tuple
from functools import reduce

import jax.numpy as jnp
import numpy as np


@dataclass
class U1_2D_OBC:
    geom: Tuple[int]
    beta: float

    def __post_init__(self):
        self.dof = np.prod(self.geom, dtype=int)
        self.shape = self.geom

        self.periodic = True

    def action(self, phi):
        return -self.beta*jnp.cos(phi).sum()

    def observe(self, phi):
        phi = phi.reshape(self.shape)
        return jnp.prod(jnp.exp(1j*phi[:self.shape[0], :self.shape[1]]))
        # Move the area and take average, full area
        # obs = jnp.array([jnp.mean(jnp.array([jnp.prod(jnp.exp(1j*(jnp.roll(phi, (i, j), axis=(0, 1))[:k, :k])))
        #                                     for i in range(self.shape[0]) for j in range(self.shape[1])])) for k in range(self.shape[0])])
        return obs


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