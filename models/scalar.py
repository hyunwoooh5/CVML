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
    # sk_shape: str = 'S'

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

        # Backwards compatibility
        self.periodic_contour = False

    def _action_quadratic(self, phi):
        m2 = self.m2
        phi = phi.reshape(self.shape)
        pot = jnp.sum((self.dt_site)*(m2/2*phi**2)).real
        kin_s = [jnp.sum((self.dt_site.real)*(jnp.roll(phi, -1, axis=d)-phi)**2)/2
                 for d in range(1, self.D+1)]
        kin_t = jnp.sum((jnp.roll(phi, -1, axis=0) - phi)**2/(2*self.dt_link))
        return pot + jnp.sum(jnp.array(kin_s)) + kin_t

    def _action_quartic(self, phi):
        lamda = self.lamda
        phi = phi.reshape(self.shape)
        pot = jnp.sum((self.dt_site.real)*(lamda*phi**4))
        return pot

    def _action(self, phi, t=1.):
        m2 = self.m2
        lamda = self.lamda
        phi = phi.reshape(self.shape)
        pot = jnp.sum((self.dt_site.real)*(t*m2/2*phi**2 + lamda*phi**4))
        kin_s = [jnp.sum((self.dt_site.real)*(jnp.roll(phi, -1, axis=d)-phi)**2)/2
                 for d in range(1, self.D+1)]
        kin_t = jnp.sum((jnp.roll(phi, -1, axis=0) - phi)**2/(2*self.dt_link))
        return pot + t*(jnp.sum(jnp.array(kin_s)) + kin_t)

    def _observe(self, phi):
        # return jnp.array([phi[0]*phi[i] for i in range(self.NT)] + [self._action(phi)])
        # phi_re = phi.reshape(self.shape)
        # return jnp.array([jnp.mean(phi_re * jnp.roll(phi_re, -i, axis=1)) for i in range(int(self.dof/self.NT))] + [self._action(phi)]) # only for 1D
        # phi_av = jnp.mean(phi_re, axis=1) # only for 1D
        # return jnp.array([jnp.mean(phi_av * jnp.roll(phi_av, -i)) for i in range(self.NT)] + [self._action(phi)])

        return phi[0] * phi[self.dof//2]

    def _phi(self, z):
        if self.lamda == 0:
            # TODO this is not right. Need to invert the whole matrix.
            raise "Free theory not yet supported"
            coefs = self._action_quadratic(z*0+1)
            phases = coefs / jnp.abs(coefs)
            return z / (phases**(1/2))

        if self.sk_shape == 'S':
            coefs = (self.dt_site)*self.lamda
            phases = coefs / jnp.abs(coefs)
            return z / (phases**(1/4)).reshape(z.shape)
        else:
            raise "Non-S shapes not supported"

    def action(self, z, t=1.):
        return self._action(z, t=t)

    def observe(self, z):
        return self._observe(z)
