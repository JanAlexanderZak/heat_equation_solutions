""" Ordinary finite-differences approach for 1D heat equation.
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import Stepper, rhs_centered
from constants import (
    dx, X_DOMAIN, NUM_OF_SPATIAL_STEPS,
    dt, TIME_STEPS, NUM_OF_TIME_STEPS,
    ALPHA, HEAT_SOURCE_INTENSITY,
)


if __name__ == "__main__":

    # * Initial condition
    T0 = np.zeros_like(X_DOMAIN)
    heat_source = HEAT_SOURCE_INTENSITY * np.sin(np.pi * X_DOMAIN)

    # Temperature matrix
    T = np.empty((NUM_OF_TIME_STEPS + 1, NUM_OF_SPATIAL_STEPS))
    T[0] = T0.copy()

    stepper = Stepper(rhs_centered, dt)
    for time_step in TIME_STEPS:
        T[time_step + 1] = stepper.rk4_step(T[time_step], dx, ALPHA, heat_source)

        if time_step % 200 == 0:
            plt.plot(X_DOMAIN, T[time_step], label=f"t={time_step * dt:.2f}")

    plt.title("Finite-differnces solution.")
    plt.xlabel("Position, x [-]")
    plt.ylabel("Temperature, u [-]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("src/plots/FDM_solution.png", dpi=200)
    #plt.show()
    plt.close()
