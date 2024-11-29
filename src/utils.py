"""
Credit goes to: https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/04_PartialDifferentialEquations/04_03_Diffusion_Explicit.html
"""

from typing import Callable

import numpy as np


class Stepper:
    def __init__(
        self,
        func_rhs: Callable,
        dt: float,
    ) -> None:
        """ Calculates the RHS solution for the next the time step (t+1).
        
        Args:
            func_rhs (Callable): right hand side function
            dt (float): time step
        """
        self.func = func_rhs
        self.dt = dt

    def euler_step(self, T_t: np.ndarray, *args) -> np.ndarray:
        """ Euler Method.

        Args:
            temperature_at_t (np.ndarray): array of current values
            args: -

        Returns:
            (np.ndarray): Current values + dt values
        """
        dxdt_plus_t = T_t + self.dt * self.func(T_t, *args)
        return dxdt_plus_t

    def rk4_step(self, T_t: np.ndarray, *args) -> np.ndarray:
        """ 4th Order Runge-Kutta Method.
        Assumes that f is time independent.
        T_t is the values at the current time step, ergo temperature values (alternatively: t0)

        Args:
            temperature_at_t (np.ndarray): array of current values
            args: -

        Returns:
            (np.ndarray): Current values + dt values
        """
        k1 = self.func(T_t, *args)
        k2 = self.func(T_t + self.dt / 2 * k1, *args)
        k3 = self.func(T_t + self.dt / 2 * k2, *args)
        k4 = self.func(T_t + k3 * self.dt, *args)
        
        dxdt_plus_t = T_t + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        return dxdt_plus_t


def analytical_1d_heat_eq(
    x_domain,
    time,
    alpha,
    heat_source,
):
    # With transient
    _ = np.exp(-(np.pi * alpha) ** 2 * time) * np.sin(np.pi * x_domain) \
        + heat_source / (np.pi * alpha) ** 2 \
        * (1 - np.exp(-(np.pi * alpha) ** 2 * time)) \
        * np.sin(np.pi * x_domain)
    
    # Without transient
    solution = heat_source / (np.pi * alpha) ** 2 \
               * (1 - np.exp(-(np.pi * alpha) ** 2 * time)) \
               * np.sin(np.pi * x_domain)
    return solution


def rhs_centered(T, dx, alpha, source):
    """Returns the right-hand side of the 1D heat
    equation based on centered finite differences
    
    Parameters
    ----------
    T : array of floats
        solution at the current time-step.
    dx : float
        grid spacing
    alpha : float
        heat conductivity
    source : array of floats
        source term for the heat equation
    
    Returns
    -------
    f : array of floats
        right-hand side of the heat equation with
        Dirichlet boundary conditions implemented
    """
    nx = T.shape[0]
    f = np.empty(nx)

    f[1:-1] = alpha / dx ** 2 * (T[:-2] - 2 * T[1:-1] + T[2:]) + source[1:-1]
    f[0] = 0
    f[-1] = 0
    
    return f


def rhs_centered_neumann(T, dx, alpha, source, heat_transfer_coefficient):
    """Returns the right-hand side of the 1D heat
    equation based on centered finite differences
    
    Parameters
    ----------
    T : array of floats
        solution at the current time-step.
    dx : float
        grid spacing
    alpha : float
        heat conductivity
    source : array of floats
        source term for the heat equation
    
    Returns
    -------
    f : array of floats
        right-hand side of the heat equation with
        Dirichlet boundary conditions implemented
    """
    nx = T.shape[0]
    f = np.empty(nx)

    # Neumann
    f[0] = f[0] + alpha / (dx ** 2) * (2 * f[1] - 2 * f[0] - 2 * dx * (f[0] - 0)) * heat_transfer_coefficient
    f[-1] = f[-1] + alpha / (dx ** 2) * (2 * f[-2] - 2 * f[-1] - 2 * dx * (f[-1] - 0)) * heat_transfer_coefficient
    
    f[1:-1] = alpha / dx ** 2 * (T[:-2] - 2 * T[1:-1] + T[2:]) + source[1:-1]
    
    return f
