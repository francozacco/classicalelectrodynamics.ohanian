import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil
from scipy.optimize import root_scalar, root
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.set_printoptions(suppress=True)



GRID = np.array(
    [
        #0     1     2     3     4     5     6     7     8     9     10    11    12    13    14
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 5 -> -6
        [0.99, 0.98, 0.96, 0.92, 0.85, 0.71, 0.44, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 4 -> -5
        [1.99, 1.97, 1.94, 1.88, 1.76, 1.53, 1.07, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 3 -> -4
        [2.99, 2.97, 2.94, 2.88, 2.78, 2.60, 2.29, 1.88, 1.74, 1.69, 1.68, 1.67, 1.67, 1.67, 1.67], # 2 -> -3
        [3.99, 3.98, 3.96, 3.93, 3.87, 3.78, 3.64, 3.48, 3.39, 3.36, 3.34, 3.34, 3.33, 3.33, 3.33], # 1 -> -2
        [5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00]  # 0 -> -1
    ]
)
FINE_GRID = np.array(
    [   
        #0     1     2     3     4     5     6     7     8     9     10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 10 -> -11
        [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 9 -> -10
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 8 -> -9
        [1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 7 -> -8
        [2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 6 -> -7
        [2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 5 -> -6
        [3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84], # 4 -> -5
        [3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67], # 3 -> -4
        [4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50], # 2 -> -3
        [4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 3.33, 3.33, 3.33, 3.33, 3.33, 3.33, 3.33, 3.33, 3.33, 3.33, 3.33, 3.33, 3.33, 3.33, 3.33], # 1 -> -2
        [5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00]  # 0 -> -1
    ]
)


def phi(x, y, grid = GRID):
    return grid[-(y + 1), x]


def is_bound(x, y):
    if y <= 0: return True
    if x >= 28: return True
    if y >= 10: return True
    if x >= 14 and y <= 5: return True

    return False



# Relaxation ##################################################################

def item_relaxation(i, j, grid):
    return (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1]) / 4


def get_errors_matrix(grid):
    errors = np.zeros_like(grid)
    errors[0, :] = 1
    errors[-1, :] = 1
    errors[:, 0] = 1
    errors[:, -1] = 1
    for i in range(1, 6):
        errors[i, 14:] = 1
    return errors

def relaxation(grid):
    errors = get_errors_matrix(grid)
    prev_grid = np.copy(grid)
    for i in range(1, 10):
        for j in range(1, 29):
            if is_bound(j, i):
                errors[i, j] = 1
                continue
            grid[i, j] = round(item_relaxation(i, j, grid), 3)
            error = abs(grid[i, j] - prev_grid[i, j])
            if error < 0.01:
                errors[i, j] = 1
    return grid, errors

def compute_relaxation(grid):
    while True:
        grid, errors_ok = relaxation(grid)
        if np.all(errors_ok):
            break

# Equipotential lines #########################################################

def regphi(xn, yn, grid, fine):
    if fine:
        if yn <= 0: return 5
        if xn >= 28: return 0
        if yn >= 10: return 0
        if xn >= 14 and yn >= 5: return 0
    else:
        if yn <= 0: return 5
        if xn >= 14: return 0
        if yn >= 5: return 0
        if xn >= 7 and yn >= 3: return 0

    x1 = floor(xn) if xn != 0 else 0
    x2 = ceil(xn) if xn != 0 else 1
    if x1 == x2:
        x2 += 1
    y1 = floor(yn) if yn != 0 else 0
    y2 = ceil(yn) if yn != 0 else 1 
    if y1 == y2:
        y2 += 1

    Q11 = phi(x1, y1, grid)
    Q21 = phi(x2, y1, grid)
    Q12 = phi(x1, y2, grid)
    Q22 = phi(x2, y2, grid)
    val = (
        (Q11 * (x2 - xn) * (y2 - yn)) +
        (Q21 * (xn - x1) * (y2 - yn)) +
        (Q12 * (x2 - xn) * (yn - y1)) +
        (Q22 * (xn - x1) * (yn - y1))
    ) / ((x2 - x1) * (y2 - y1))

    return val

def help_func(y, j, i, grid, fine):
    reg = regphi(j, y, grid, fine)
    res = reg - i
    return res

def compute_equipotentials(grid, fine):
    y_max = 10 if fine else 5
    lines = np.arange(0.25, 5, 0.25)
    for i in lines:
        top_x = 28 if fine else 13
        x = np.arange(0, top_x, 0.1)
        y_list = []
        for j in x:
            sol = root_scalar(help_func, bracket=[0, y_max], args=(j, i, grid, fine))
            y_list.append(sol.root)
        label = "Equipotential lines" if i == lines[-1] else None
        plt.plot(x, np.array(y_list), color='blue', label=label)


def main():
    # plt.plot([i for i in range(8)], [5 for i in range(8)], color='black')
    # plt.plot([7 for i in range(3, 6)], [i for i in range(3, 6)], color='black')
    # plt.plot([i for i in range(7, 14)], [3 for i in range(7, 14)], color='black')
    # plt.plot([i for i in range(14)], [0 for i in range(14)], color='black')
    # compute_equipotentials(GRID, False)

    plt.plot([i for i in range(15)], [10 for i in range(15)], color='black')
    plt.plot([14 for i in range(5, 11)], [i for i in range(5, 11)], color='black')
    plt.plot([i for i in range(14, 29)], [5 for i in range(14, 29)], color='black')
    plt.plot([i for i in range(29)], [0 for i in range(29)], color='black')
    compute_relaxation(FINE_GRID)
    compute_equipotentials(FINE_GRID, True)

    plt.legend(loc='upper right')


if __name__ == "__main__":
    main()
    plt.show()
