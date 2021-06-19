#! python3

import numpy as np
from itertools import combinations
import logging
from pprint import pprint, pformat
from tabulate import tabulate

#logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

def production_planning():
    def f(x):
        return - x[1] - 2*x[2]

    A = np.array([1,0,1,0,0,0,2,0,1,0,1,1,0,0,1]).reshape((3, -1))
    b = np.array([100, 200, 150]).reshape((3, -1))

    return f, A, b

def example_1():
    def f(x):
        return - 3*x[1] + x[2]

    A = np.array([1,2,1,0,-1,1,0,1]).reshape((2, -1))
    b = np.array([4, 1]).reshape((2, -1))

    return f, A, b

def hw4():
    def f(x):
        return x[1] + 4*x[2] + x[3]

    A = np.array([2,2,1,1,0,1,0,-1,0,-1]).reshape((2, -1))
    b = np.array([4, 1]).reshape((2, -1))

    return f, A, b

def cycling_example():
    def f(x):
        return -2*x[1] - 3*x[2] + x[3] + 12*x[4]

    A = np.array([[  -2, -9,    1,  9, 1, 0],
                 [  1/3,  1, -1/3, -2, 0, 1]])
    b = np.array([0, 0]).reshape((2, -1))

    return f, A, b

def get_soln(A, b):
    m, n = A.shape
    
    header = ['Indices', 'Solution', 'Objective', 'X_B', 'X_N']
    table = []
    
    for columns in combinations(range(n), m):
        xb = np.array([A[:, column] for column in columns]).transpose()
        try:
            xB = np.linalg.inv(xb).dot(b)
        except np.linalg.LinAlgError:
            continue
        B = set(col+1 for col in columns)
        N = set(range(1, n+1)) - B
        x_B = {column: float(value) for column, value in zip(B, xB)}
        x_N = {column: 0 for column in N}
        X = {**x_B, **x_N}
        
        for xn in X.values():
            if xn < 0:
                fx = 'Infeasible'
                break
        else:
            fx = f(X)
            yield (X, B, N)

        line = [B, list(X.values()), fx, x_B, x_N]
        table.append(line)

    table = tabulate(table, header, tablefmt="fancy_grid")
    logging.debug(table)
        
f, A, b = cycling_example()
for x, B, N in get_soln(A, b):
    pass
    #print(f'x = {pformat(x)}\nB = {B}\nN = {N}\n')
