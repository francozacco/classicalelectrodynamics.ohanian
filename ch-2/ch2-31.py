import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil
from scipy.optimize import root_scalar, root
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


GRID = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0, 3.0, 6.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0, 4.0, 4.0, 8.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0, 5.0, 5.0, 8.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        [6.0, 6.0, 6.0, 6.0, 6.0, 8.0, 6.0, 6.0, 6.0, 6.0, 6.0],
        [7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 7.0, 7.0, 7.0, 7.0, 7.0],
        [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
    ]
)


def phi(x, y):
    return GRID[-(y + 1), x]


def is_bound(i, j):
    if i <= 0 or i >= 8 or j >= 10:
        return True
    if i >= 4 and j == 5:
        return True
    return False


def item_relaxation(i, j):
    return (GRID[i-1, j] + GRID[i+1, j] + GRID[i, j-1] + GRID[i, j+1]) / 4


def get_errors_matrix():
    errors = np.zeros_like(GRID)
    errors[0, :] = 1
    errors[-1, :] = 1    
    errors[:, 0] = 1
    errors[:, -1] = 1
    return errors

def relaxation():
    errors = get_errors_matrix()
    prev_grid = np.copy(GRID)
    for i in range(1, 8):
        for j in range(1, 10):
            if is_bound(i, j):
                errors[i, j] = 1
                continue
            GRID[i, j] = round(item_relaxation(i, j), 3)
            error = abs(GRID[i, j] - prev_grid[i, j])
            if error < 0.01:
                errors[i, j] = 1
    return errors

def compute_relaxation():
    while True:
        errors_ok = relaxation()
        if np.all(errors_ok):
            break

# Equipotential lines #########################################################

def regphi(xn, yn):
    if yn >= 8:
        return 0
    if yn <= 0:
        return 8
    if yn <= 4 and np.isclose(xn, 5):
        return 8
    if xn >= 10:
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

def help_func(y, j, i):
    reg = regphi(j, y)
    res = reg - i
    return res

def compute_equipotentials():
    lines = np.arange(0.25, 8, 0.5)
    for i in lines:
        x = np.arange(0, 10, 0.1)
        y_list = []
        for j in x:
            sol = root_scalar(help_func, bracket=[0,8], args=(j, i))
            y_list.append(sol.root)
        label = "Equipotential lines" if i == lines[-1] else None
        plt.plot(x, np.array(y_list), color='blue', label=label)
        
# Electric field lines ########################################################

def Ex(x, y, h):
    val = (regphi(x + h, y) - regphi(x - h, y)) / (2 * h)
    return val


def Ey(x, y, h):
    val = (regphi(x, y + h) - regphi(x, y - h)) / (2 * h)
    return val


def compute_electric_field():
    step = 0.01
    x = np.arange(0, 10, 0.25)
    for j in x:
        x_points = [j]
        y_points = [8]
        cond = True
        while cond:
            Ex_val = Ex(x_points[-1], y_points[-1], step)
            Ey_val = Ey(x_points[-1], y_points[-1], step)
            x_points.append(x_points[-1] + (Ex_val * step))
            y_points.append(y_points[-1] + (Ey_val * step))
            x_last, y_last = x_points[-1], y_points[-1]
            if y_last <= 0:
                y_points.pop(-1)
                y_points.append(0)
                cond = False
            if y_last > 8:
                y_points.pop(-1)
                y_points.append(8)
                cond = False
            if y_last <= 4 and np.isclose(x_last, 5):
                x_points.pop(-1)
                x_points.append(5)
                cond = False
            if x_last >= 10:
                x_points.pop(-1)
                x_points.append(10)
                cond = False
        label = "Electric field lines" if j == x[-1] else None
        plt.plot(x_points, np.array(y_points), color='red', label=label)


def main():
    compute_relaxation()
    x = np.arange(0, 11, 1)
    borders_width = 3
    plt.plot(x, np.zeros(11), color='black', linewidth=borders_width)
    plt.plot(x, np.ones(11) * 8, color='black', linewidth=borders_width)
    plt.plot([5, 5], [0, 4], color='black', linewidth=borders_width)
    compute_equipotentials()
    compute_electric_field()
    plt.legend(loc='upper right')


if __name__ == "__main__":
    main()
    plt.show()
