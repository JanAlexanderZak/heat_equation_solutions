import numpy as np
import matplotlib.pyplot as plt

from utils import analytical_1d_heat_eq
from constants import (
    X_DOMAIN,
    TIME_DOMAIN,
    HEAT_SOURCE_INTENSITY,
    THERMAL_DIFFUSIVITY,
    NUM_OF_TIME_STEPS,
)


if __name__ == "__main__":
    #
    # * Introductory example, factor out kappa and heat_source
    #
    kappa = 1

    char_temp = HEAT_SOURCE_INTENSITY / kappa
    char_time = 1 / kappa

    for idx, time in enumerate(TIME_DOMAIN):
        # alpha and heat source are kept as with ordinary analytical solution
        temperature = analytical_1d_heat_eq(
            x_domain=X_DOMAIN / max(X_DOMAIN),
            time=time / char_time,
            heat_source=1,
            alpha=1,
        )
        if idx % 200 == 0:
            plt.plot(X_DOMAIN, temperature * char_temp, label=f"t={time:.2f}")

    plt.title("Non-dimensionalized analytical solution, alpha=1")
    plt.xlabel("Position x, [-]")
    plt.ylabel("Temperature u, [-]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("src/plots/non-dimensionalized_analytical_solution_alpha=1.png", dpi=200)
    #plt.show()
    plt.close()
    
    #
    # * Change physical parameters
    #
    print(THERMAL_DIFFUSIVITY)      # approx 0.1 hence 10 time higher
    char_temp = max(X_DOMAIN) * HEAT_SOURCE_INTENSITY / THERMAL_DIFFUSIVITY
    char_time = max(X_DOMAIN) / THERMAL_DIFFUSIVITY

    for idx, time in enumerate(np.linspace(0, 10, NUM_OF_TIME_STEPS)):
        temperature = analytical_1d_heat_eq(
            x_domain=X_DOMAIN / max(X_DOMAIN),
            time=time / char_time,
            heat_source=1,
            alpha=1,
        )
        if idx % 200 == 0:
            plt.plot(X_DOMAIN, temperature * char_temp, label=f"t={time:.2f}")

    plt.title("Non-dimensionalized analytical solution, alpha=0.1")
    plt.xlabel("Position x, [-]")
    plt.ylabel("Temperature u, [°C]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("src/plots/non-dimensionalized_analytical_solution_alpha=0.1.png", dpi=200)
    #plt.show()
    plt.close()

    #
    # * Change spatial domain
    #
    print(THERMAL_DIFFUSIVITY)          # Approx 0.1 hence 10 time higher
    X_DOMAIN = np.linspace(0, 2, 100)   # Changed to twice the length
    char_temp = max(X_DOMAIN) ** 2 * HEAT_SOURCE_INTENSITY / THERMAL_DIFFUSIVITY
    char_time = max(X_DOMAIN) ** 2 / THERMAL_DIFFUSIVITY

    for idx, time in enumerate(np.linspace(0, 30, NUM_OF_TIME_STEPS)):
        temperature = analytical_1d_heat_eq(
            x_domain=X_DOMAIN / max(X_DOMAIN),
            time=time / char_time,
            heat_source=1,
            alpha=1,
        )
        if idx % 200 == 0:
            plt.plot(X_DOMAIN / max(X_DOMAIN), temperature * char_temp, label=f"t={time:.2f}")

    plt.title("Non-dimensionalized analytical solution, L=2, alpha=0.1")
    plt.xlabel("Position, x [-]")
    plt.ylabel("Temperature, u [°C]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("src/plots/non-dimensionalized_analytical_solution_L=2_alpha=0.1.png", dpi=200)
    #plt.show()
    plt.close()
