import numpy as np
import matplotlib.pyplot as plt

from utils import analytical_1d_heat_eq
from constants import X_DOMAIN, ALPHA, HEAT_SOURCE_INTENSITY, TIME_DOMAIN


if __name__ == "__main__":
    for idx, time in enumerate(TIME_DOMAIN):
        temperature = analytical_1d_heat_eq(
            x_domain=X_DOMAIN,
            time=time,
            heat_source=HEAT_SOURCE_INTENSITY,
            alpha=ALPHA,
        )
        if idx % 200 == 0:
            plt.plot(X_DOMAIN, temperature, label=f"t={time:.2f}")

    plt.title("Analytical solution.")
    plt.xlabel("Position, x [-]")
    plt.ylabel("Temperature, u [-]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("src/plots/analytical_solution.png", dpi=200)
    #plt.show()
    plt.close()
