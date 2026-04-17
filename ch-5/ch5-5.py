import matplotlib.pyplot as plt
import numpy as np


def E_fn(x):
    fact = e /((a**2) * (x**2))
    return fact * (1 - ((np.e**(-x)) * ( 1 + (x) + ((x**2)/2) ) ) )


if __name__ == "__main__":
    e = 4.803e-10
    a = 0.23e-13
    r = np.arange(0, 5, 0.001)
    E = E_fn(r)

    plt.plot(r, E)
    plt.xticks([i for i in range(5)] + [1.451])
    plt.ylabel("E(x)")
    plt.xlabel("x = r/a")
    plt.grid()
    plt.show()