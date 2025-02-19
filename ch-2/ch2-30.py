import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


GRID = np.array(
    [
        [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
        [9.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 9.0],
        [9.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 9.0],
        [9.0, 6.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 9.0],
        [9.0, 6.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 9.0],
        [9.0, 6.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 9.0],
        [9.0, 6.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 9.0],
        [9.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 9.0],
        [9.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 9.0],
        [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
    ]
)


def is_bound(i, j):
    if GRID[i,j] == 0:
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
    for i in range(1, 9):
        for j in range(1, 10):
            if is_bound(i, j):
                errors[i, j] = 1
                continue
            GRID[i, j] = round(item_relaxation(i, j), 3)
            error = abs(GRID[i, j] - prev_grid[i, j])
            if error < 0.001:
                errors[i, j] = 1
    return errors


def main():
    print(GRID)
    while True:
        errors = relaxation()
        if np.all(errors):
            break
    print(GRID)


if __name__ == "__main__":
    main()
