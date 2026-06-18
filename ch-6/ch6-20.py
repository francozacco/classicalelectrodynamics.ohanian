import matplotlib.pyplot as plt
import numpy as np



def fn_tau_1(V0):
    return 2 * V0 * np.sqrt(1 - (V0**2))

def fn_tau_2(V0):
    return V0 * np.sqrt(1 - (4 * (V0**2))) + (np.arcsin(2 * V0) / 2)

if __name__ == "__main__":
    V0 = np.arange(0, 1, 0.001)
    tau_1 = fn_tau_1(V0)
    tau_2 = fn_tau_2(V0)

    plt.plot(V0, tau_1, label="tau1")
    plt.plot(V0, tau_2, label="tau2")
    plt.ylabel("tau")
    plt.xlabel("V0")
    plt.legend()
    plt.grid()
    plt.show()