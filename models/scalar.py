from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
import numpy as np


def sk_contour(nbeta, nt, shape='S'):
    if shape == 'L':
        return jnp.concatenate([1j*jnp.ones(nt), -1j*jnp.ones(nt), jnp.ones(nbeta)])
    elif shape == 'S':
        nbeta1 = int(nbeta/2)
        nbeta2 = nbeta-nbeta1
        return jnp.concatenate([
            1j*jnp.ones(nt),
            jnp.ones(nbeta1),
            -1j*jnp.ones(nt),
            jnp.ones(nbeta2),
        ])
    elif shape == 'F':
        return jnp.concatenate([1j*jnp.ones(nt), -1j*jnp.ones(nt), 1j*jnp.ones(nt), -1j*jnp.ones(nt), jnp.ones(nbeta)])
    else:
        raise "TODO"


@dataclass
class Model:
    geom: Tuple[int]
    nbeta: int
    nt: int
    m2: float
    lamda: float
    dt: float = 1.
    sk_shape: str = 'S'

    def __post_init__(self):
        self.D = len(self.geom)
        self.NT = self.nbeta + 2*self.nt
        self.dof = np.prod(self.geom, dtype=int)*self.NT
        self.shape = (self.NT,)+self.geom
        self.contour = sk_contour(self.nbeta, self.nt, shape=self.sk_shape)
        self.contour_site = jnp.array(
            [(self.contour[i]+self.contour[(i-1) % self.NT])/2 for i in range(self.NT)])
        self.dt_link = self.dt * \
            jnp.tile(self.contour.reshape(
                (self.NT,)+(1,)*self.D), (1,)+self.geom)
        self.dt_site = self.dt * \
            jnp.tile(self.contour_site.reshape(
                (self.NT,)+(1,)*self.D), (1,)+self.geom)
        self.L = self.geom[0]

        # Backwards compatibility
        self.periodic = False

    def action(self, phi):
        m2 = self.m2
        lamda = self.lamda
        phi = phi.reshape(self.shape)
        pot = jnp.sum((self.dt_site)*(m2/2*phi**2 + lamda*phi**4/24))
        kin_s = [jnp.sum((self.dt_site)*(jnp.roll(phi, -1, axis=d)-phi)**2)/2
                 for d in range(1, self.D+1)]
        kin_t = jnp.sum((jnp.roll(phi, -1, axis=0) - phi)**2/(2*self.dt_link))
        return pot + (jnp.sum(jnp.array(kin_s)) + kin_t)

    def action_local(self, phi, n):  # only for 1+1D
        phi_n = phi[n]
        phi = phi.reshape(self.shape)

        t = n // self.nbeta
        x = (n - t * self.nbeta) % self.L

        # Rolling is slow
        idx_mt = (t - 1) % self.nbeta, x
        idx_mx = t, (x - 1) % self.L
        idx_pt = (t + 1) % self.nbeta, x
        idx_px = t, (x + 1) % self.L

        pot = self.m2/2. * phi_n**2 + self.lamda/24. * phi_n**4
        kint = ((phi[idx_pt]-phi_n)**2 + (phi[idx_mt]-phi_n)**2) / 2.
        kinx = ((phi[idx_px]-phi_n)**2 + (phi[idx_mx]-phi_n)**2) / 2.

        return pot+kint+kinx

    def observe(self, phi, i):
        # return jnp.array([phi[0]*phi[i] for i in range(self.NT)] + [self.action(phi)])
        phi_re = phi.reshape(self.shape)
        # return jnp.array([jnp.mean(phi_re * jnp.roll(phi_re, -i, axis=1)) for i in range(int(self.dof/self.NT))] + [self.action(phi)]) # only for 1D
        phi_av = jnp.mean(phi_re, axis=1)  # only for 1D
        return jnp.mean(phi_av * jnp.roll(phi_av, -i))
        return jnp.array([jnp.mean(phi_av * jnp.roll(phi_av, -i)) for i in range(self.NT)])
        # return jnp.array([jnp.mean(phi_av * jnp.roll(phi_av, -i)) for i in range(self.NT)] + [phi_av[i] for i in range(self.NT)] + [self.action(phi)])
        return jnp.mean(phi_av * jnp.roll(phi_av, self.NT//2))
        return phi[0] * phi[self.dof//2]
