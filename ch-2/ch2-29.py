import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from math import floor, ceil
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def phi(x, y):
    arr = np.array(
        [
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],# 5 -> -6
            [0.99, 0.98, 0.96, 0.92, 0.85, 0.71, 0.44, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],# 4 -> -5
            [1.99, 1.97, 1.94, 1.88, 1.76, 1.53, 1.07, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],# 3 -> -4
            [2.99, 2.97, 2.94, 2.88, 2.78, 2.60, 2.29, 1.88, 1.74, 1.69, 1.68, 1.67, 1.67, 1.67, 1.67],# 2 -> -3
            [3.99, 3.98, 3.96, 3.93, 3.87, 3.78, 3.64, 3.48, 3.39, 3.36, 3.34, 3.34, 3.33, 3.33, 3.33],# 1 -> -2
            [5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00] # 0 -> -1
        ]
    )
    return arr[-(y + 1), x]

def regphi(xn, yn):
    if yn <= 0:
        return 5
    if is_bound(xn, yn):
        return 0

    x1 = floor(xn) if xn != 0 else 0
    x2 = ceil(xn) if xn != 0 else 1
    if x1 == x2:
        x2 += 1
    y1 = floor(yn) if yn != 0 else 0
    y2 = ceil(yn) if yn != 0 else 1 
    if y1 == y2:
        y2 += 1
        
    Q11 = phi(x1, y1)
    Q21 = phi(x2, y1)
    Q12 = phi(x1, y2)
    Q22 = phi(x2, y2)
    val = (
        (Q11 * (x2 - xn) * (y2 - yn)) +
        (Q21 * (xn - x1) * (y2 - yn)) +
        (Q12 * (x2 - xn) * (yn - y1)) +
        (Q22 * (xn - x1) * (yn - y1))
    ) / ((x2 - x1) * (y2 - y1))

    return val


def is_bound(x, y):
    if x < 6 and y >= 5: return True
    if x >= 6 and y >= 3: return True
    return False


def Ex(x, y):
    h = 0.01
    return (regphi(x + h, y) - regphi(x - h, y)) / (2 * h)


def Ey(x, y):
    h = 0.01
    return (regphi(x, y + h) - regphi(x, y - h)) / (2 * h)


def main():
    step = 0.01
    mult = 4
    for i in range(1, 14 * mult):
        x_points = [i / mult]
        y_points = [0]
        cond = True
        while cond:
            Ex_val = -Ex(x_points[-1], y_points[-1])
            Ey_val = -Ey(x_points[-1], y_points[-1])
            x_points.append(x_points[-1] + (Ex_val * step))
            y_points.append(y_points[-1] + (Ey_val * step))
            if is_bound(x_points[-1], y_points[-1]):
                x_points.pop(-1)
                y_points.pop(-1)                
                cond = False
        plt.plot(x_points, y_points, color='blue')

    plt.plot([i for i in range(7)], [5 for i in range(7)], color='black')
    plt.plot([6 for i in range(3, 6)], [i for i in range(3, 6)], color='black')
    plt.plot([i for i in range(6, 15)], [3 for i in range(6, 15)], color='black')
    plt.plot([i for i in range(15)], [0 for i in range(15)], color='black')


if __name__ == "__main__":
    main()
    plt.show()