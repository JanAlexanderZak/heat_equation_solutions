""" Non-dimensionalized numerical approach for 1D heat equation.
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import Stepper, rhs_centered_neumann
from constants import (
    dx, X_DOMAIN, NUM_OF_SPATIAL_STEPS,
    dt, TIME_STEPS, NUM_OF_TIME_STEPS,
    ALPHA, THERMAL_DIFFUSIVITY,
)

if __name__ == "__main__":
    # Real physical parameter
    print(THERMAL_DIFFUSIVITY)
    
    # Redefine space
    reference_length = 1
    NUM_OF_SPATIAL_STEPS = 40
    dx = reference_length / (NUM_OF_SPATIAL_STEPS - 1)
    X_DOMAIN = np.linspace(0, reference_length, NUM_OF_SPATIAL_STEPS)

    # Redefine time, depends now on thermal_diffusivity
    reference_time = 10
    dt = 0.49 * dx ** 2 / THERMAL_DIFFUSIVITY
    NUM_OF_TIME_STEPS = int(reference_time / dt)
    TIME_STEPS = np.arange(0, NUM_OF_TIME_STEPS)

    # Characteristic values
    char_temp = max(X_DOMAIN) * np.pi ** 2 / THERMAL_DIFFUSIVITY
    char_time = max(X_DOMAIN) / THERMAL_DIFFUSIVITY

    # * Initial condition
    T0 = np.zeros_like(X_DOMAIN)
    heat_source = 1 * np.sin(np.pi * X_DOMAIN / max(X_DOMAIN))          # intensity = 1

    # Temperature matrix
    T = np.empty((NUM_OF_TIME_STEPS + 1, NUM_OF_SPATIAL_STEPS))
    T[0] = T0.copy()

    stepper = Stepper(rhs_centered_neumann, dt / char_time)
    heat_transfer_coefficient = 0.01
    
    # Critical: solve ordinary here and just alter time and char temp
    for time_step in TIME_STEPS:
        T[time_step + 1] = stepper.rk4_step(T[time_step], dx, ALPHA, heat_source, heat_transfer_coefficient)

        # Revert non-dimensionalization for solution
        if time_step % 200 == 0:
            plt.plot(X_DOMAIN, T[time_step] * char_temp, label=f"t={time_step * dt:.2f}")

    plt.title("Non-dimensionalized numerical solution.")
    plt.xlabel("Position x, [-]")
    plt.ylabel("Temperature u, [°C]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("src/plots/non-dimensionalized_FDM_solution_neumann_ht=0.1.png", dpi=200)
    #plt.show()
    plt.close()
