import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

from src.constants import X_DOMAIN, TIME_DOMAIN


def main():
    X, T = np.meshgrid(X_DOMAIN, TIME_DOMAIN)

    X_star = np.hstack((
        X.flatten().reshape(-1, 1),
        T.flatten().reshape(-1, 1),
    ))

    u_pred = torch.load(f"./src/pinn/data/predictions/predictions.pkl")
    u_pred = torch.tensor(u_pred).reshape(-1, 1)
    u_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")

    for time_step in range(u_pred.shape[0]):
        if time_step % 10 == 0:
            plt.plot(X_DOMAIN, u_pred[time_step], label=f"t={time_step * 1:.2f}")

    plt.title("PINN solution.")
    plt.xlabel("Position x, [-]")
    plt.ylabel("Temperature u, [-]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("./src/plots/pinn_solution.png", dpi=200)
    #plt.show()
    plt.close()


if __name__ == "__main__":
    main()
