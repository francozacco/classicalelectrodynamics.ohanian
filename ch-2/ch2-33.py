import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil
from scipy.optimize import root_scalar, root
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.set_printoptions(suppress=True)



GRID = np.array(
    [
        #0     1     2     3     4     5     6     7     8     9     10    11    12    13    14    15    16    17    18
        [4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00], # 4 -> -5
        [3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 2.66, 2.00, 2.66, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00], # 3 -> -4
        [2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 1.33, 0.00, 1.33, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00], # 2 -> -3
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.00, 0.00, 0.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00], # 1 -> -2
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 0 -> -1
    ]
)


def phi(x, y, grid = GRID):
    return grid[-(y + 1), x]


def is_bound(i, j):
    if i <= 0: return True
    if i >= 4: return True
    if i == 2 and j == 9: return True
    if i == 3 and 8 <= j <= 10: return True

    return False

# Relaxation ##################################################################

def item_relaxation(i, j, grid):
    if j == 0:
        return (grid[i-1, j] + grid[i+1, j] + grid[i, j+1]) / 3
    if j == 18:
        return (grid[i-1, j] + grid[i+1, j] + grid[i, j-1]) / 3
    return (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1]) / 4


def get_errors_matrix(grid):
    errors = np.zeros_like(grid)
    errors[0, :] = 1
    errors[-1, :] = 1
    errors[-2, 8:11] = 1
    errors[-3, 9] = 1
    return errors


def relaxation(grid):
    errors = get_errors_matrix(grid)
    prev_grid = np.copy(grid)
    for i in range(1, 4):
        for j in range(0, 19):
            if is_bound(i, j):
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

def is_inside_triangle(xn, yn):
    total = triangle_area((7, 0), (9, 2), (11, 0))
    A1 = triangle_area((xn, yn), (7, 0), (9, 2)) 
    A2 = triangle_area((xn, yn), (9, 2), (11, 0))
    A3 = triangle_area((xn, yn), (11, 0), (7, 0))
    if total == (A1 + A2 + A3):
        return True
    return False


def triangle_area(point_a, point_b, point_c):
    x1, y1 = point_a
    x2, y2 = point_b
    x3, y3 = point_c
    return 0.5 * abs((x1*(y2 - y3)) + (x2*(y3 - y1)) + (x3*(y1 - y2)))


def regphi(xn, yn, grid):
    if yn == 0: return 0
    if yn == 4: return 4
    if is_inside_triangle(xn, yn):
        return 0

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


def help_func(y, j, i, grid):
    reg = regphi(j, y, grid)
    res = reg - i
    return res


def compute_equipotentials(grid):
    lines = np.arange(0.25, 4, 0.25)
    for i in lines:
        x = np.arange(0, 18, 0.01)
        y_list = []
        for j in x:
            sol = root_scalar(help_func, bracket=[0, 4], args=(j, i, grid))
            y_list.append(sol.root)
        label = "Equipotential lines" if i == lines[-1] else None
        plt.plot(x, np.array(y_list), color='blue', label=label)


def main():
    plt.plot([i for i in range(8)], [0 for i in range(8)], color='black')
    plt.plot([i for i in range(7, 10)], [0, 1, 2], color='black')
    plt.plot([i for i in range(9, 12)], [2, 1, 0], color='black')
    plt.plot([i for i in range(11, 19)], [0 for i in range(11, 19)], color='black')
    plt.plot([i for i in range(19)], [4 for i in range(19)], color='black')

    compute_relaxation(GRID)
    print(GRID)
    compute_equipotentials(GRID)

    plt.legend(loc='upper right')


if __name__ == "__main__":
    main()
    plt.show()
