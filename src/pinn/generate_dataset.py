"""
Reference:
https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/04_PartialDifferentialEquations/04_03_Diffusion_Explicit.html (Free University of Brussels)
"""
import os

from typing import List

import numpy as np
import plotly.express as px

from pyDOE import lhs
from src.utils import analytical_1d_heat_eq
from src.constants import X_DOMAIN, ALPHA, HEAT_SOURCE_INTENSITY

# Physical parameters
alpha = 0.1                             # Heat transfer coefficient
lx = 1.                                 # Size of computational domain

# Grid parameters
nx = 21                                 # number of grid points 
X_DOMAIN = np.linspace(0., lx, nx)      # coordinates of grid points

u_list = []
TIME_DOMAIN = np.linspace(0, 5, 100)


def generate_dataset(path: str = "src/pinn/data",):
    u_list = []
    for time in TIME_DOMAIN:
        u = analytical_1d_heat_eq(
            x_domain=X_DOMAIN,
            time=time,
            heat_source=HEAT_SOURCE_INTENSITY,
            alpha=ALPHA,
        )
        u_list.append(u)

    y_train = np.vstack(u_list)

    # * Initial and boundary conditions
    X, T = np.meshgrid(X_DOMAIN, TIME_DOMAIN)  # (100, 21), (100, 21)

    X_star = np.hstack((
        X.flatten().reshape(-1, 1),
        T.flatten().reshape(-1, 1),
    ))

    lower_boundary = X_star.min(axis=0)
    upper_boundary = X_star.max(axis=0)

    # initial conditions
    x_train_IC = np.hstack((X[0:1, :].T, T[0:1, :].T))
    y_train_IC = y_train[0, :].reshape(-1, 1)
    #y_train_IC = np.sin(X_DOMAIN).reshape(-1, 1)  # sinIC

    # Upsample IC
    x_train_IC = np.tile(x_train_IC, 3).reshape(-1, 2)
    y_train_IC = np.tile(y_train_IC, 3).reshape(-1, 1)

    # boundary conditions
    x_train_BC_lb = np.hstack((X[:, 0:1], T[:, 0:1]))
    y_train_BC_lb = y_train[:, 0:1]
    x_train_BC_ub = np.hstack((X[:, -1:], T[:, -1:]))
    y_train_BC_ub = y_train[:, -1:]

    x_train_BC = np.vstack([x_train_BC_lb, x_train_BC_ub])
    y_train_BC = np.vstack([y_train_BC_lb, y_train_BC_ub])

    # * Sample collocation
    x_train = lower_boundary + (upper_boundary - lower_boundary) * lhs(2, 1000)
    x_train = np.vstack((x_train, x_train_BC))

    # * Final data
    idx = np.random.choice(x_train_BC.shape[0], 100, replace=False)
    x_train_BC = x_train_BC[idx, :]
    y_train_BC = y_train_BC[idx, :]

    idx = np.random.choice(x_train_IC.shape[0], 100, replace=True)
    x_train_IC = x_train_IC[idx, :] 
    y_train_IC = y_train_IC[idx, :]

    # * Save
    np.save(os.path.join(path, "x_train_BC"), x_train_BC)
    np.save(os.path.join(path, "y_train_BC"), y_train_BC)
    np.save(os.path.join(path, "x_train_IC"), x_train_IC)
    np.save(os.path.join(path, "y_train_IC"), y_train_IC)
    np.save(os.path.join(path, "x_train"), x_train)
    np.save(os.path.join(path, "y_train"), y_train)
    np.save(os.path.join(path, "X_star"), X_star)


if __name__ == "__main__":
    generate_dataset()
