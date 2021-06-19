#! python3

import numpy as np
from tabulate import tabulate
from pprint import pprint, pformat
from numba import jit
import logging

#logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


def simplex(A, b, c, B, two_phase=None):
    """Simplex Algorithm: Returns Optimal Solution Vector, Basis, Non-basis"""

    if two_phase == None:
        logging.debug("Simplex Method:\n" + '-'*8 + '\n')
    elif two_phase == 1:
        logging.debug('\n' + " Phase 1 ".center(50, '_') + '\n')
    elif two_phase == 2:
        logging.debug('\n' + " Phase 2 ".center(50, '_') + '\n')
    
    m, n = A.shape
    B = np.array(list(B))
    indices = np.array(list(range(1, n+1)))
    N = np.setdiff1d(indices, B)

    iterations = 1

    while True:
        logging.debug(f'ITERATION {iterations}:\n')
        
        # Step 1:
        B, N = np.array(sorted(B)), np.array(sorted(N))
        A_B = np.array([A[:, i-1] for i in B]).transpose()
        A_B_inv = np.linalg.inv(A_B)
        
        logging.debug(f'STEP 1:\nB = {B}, \nN = {N}, \nA_B = \n{A_B}')#, \nA_B^-1 = \n{A_B_inv}')

        # Step 2:
        x_B = A_B_inv.dot(b)
        x_N = np.zeros((1, len(N)))
        x = np.zeros((n))
        for idx, i in enumerate(B):
            x[i-1] = x_B.flatten()[idx]
        for idx, i in enumerate(N):
            x[i-1] = x_N.flatten()[idx]
        c_B = np.array([c[i-1] for i in B])
        z = c_B.dot(x_B)[0]    # Objective value

        logging.debug(f'\nSTEP 2:\nc_B = {c_B}\nx_B = {x_B.flatten()}^T\nx = {x}\nz = {z}')

        # Step 3:
        c_bar = np.array([c[i-1] - c_B.dot(A_B_inv.dot(A[:, i-1].T)) for i in N])

        logging.debug(f'\nSTEP 3:\nc_bar = {c_bar}')

        enter_indices = np.where(c_bar<0)[0]
        # This Nj index will enter Basis
        if enter_indices.size:
            Nj = N[enter_indices[0]]
            logging.debug(f'Nj = {Nj}')
        else:
            logging.debug(f'BFS = {B}\n'+'_'*50+'\n')
            if two_phase != 1:
                print(f"Solution x = {x}\nOptimum value = {c.dot(x)}\nBasis = {B}\nNon-basis = {N}")
            break
        
        # Step 4:
        d = -A_B_inv.dot(A[:, Nj-1].transpose())

        logging.debug(f'\nSTEP 4:\nd = {d}')

        if not any(d<0):
            print('Unbounded Problem.')
            return None, None, None

        # This Bi index will enter Basis
        theta = {i+1: -x/d for i, (x, d) in enumerate(zip(x_B, d)) if d<0}
        Bi = B[min(zip(theta.values(), theta.keys()))[1]-1]
        logging.debug(f'Bi = {Bi}')

        # Swap
        N[np.where(N==Nj)[0]], B[np.where(B==Bi)[0]] = Bi, Nj

        logging.debug('_'*50+'\n')
        iterations += 1
        
    return x, B, N


def two_phase_simplex(A, b, c):
    """Two Phase Simplex Algorithm: Returns Optimal Solution Vector, Basis,
    Non-basis"""
    
    logging.debug("Two Phase Simplex Method:")
    m, n = A.shape

    signs_b = [-1 if bi < 0 else 1 for bi in b]
    
    A = np.array([sign * a for a, sign in zip(A, signs_b)])
    b = np.array([sign * bb for bb, sign in zip(b, signs_b)])
    
    # Phase 1
    A2 = np.array([list(a) + list(i) for a, i in zip(A, np.identity(m))])
    c2 = np.array([0,]*len(c) + [1,]*m)

    nn = n+1
    for _ in range(n):
        try:
            x, B, N = simplex(A2, b, c2,  set(range(nn, nn+m)), 1)
            break
        except np.linalg.LinAlgError:
            nn = n - 1
            continue

    initial_basis, non_basis = list(B), list(N)

    # Optimum value from phase 1 must be zero else, original problem is infeasible
    if c2.dot(x) != 0:
        print('Infeasible Problem.')
        return None, None, None
    
    while m - len(initial_basis) > 0:
        n = non_basis.pop()
        initial_basis.append(n)

    # Phase 2    
    x, B, N = simplex(A, b, c, initial_basis, 2)
    
    return x, B, N
        
if __name__ == '__main__':
    def problem():
        c = np.array([-500, -250, -600, 0, 0, 0])
        A = np.array([
                        [2, 1, 1, 1, 0, 0],
                        [3, 1, 2, 0, 1, 0],
                        [1, 2, 4, 0, 0, 1]
                    ])
        b = np.array([[240],
                      [150],
                      [180]])
        return c, A, b

    def problem_2():
        c = np.array([-2, -3, 1, 12, 0, 0])
        A = np.array([[-2, -9,    1,  9, 1, 0],
                      [1/3, 1, -1/3, -2, 0, 1]])
        b = np.array([[0],
                      [0]])
        return c, A, b

    def production_planning():
        c = np.array([-1, -2, 0, 0, 0])
        A = np.array([1,0,1,0,0,0,2,0,1,0,1,1,0,0,1]).reshape((3, -1))
        b = np.array([100, 200, 150]).reshape((3, -1))

        return c, A, b

    def toy():
        c = np.array([-3, 1, 0, 0])
        A = np.array([1, 2, 1, 0, -1, 1, 0, 1]).reshape((2, -1))
        b = np.array([4, 1]).reshape((2, -1))

        return c, A, b

    def toy():
        c = np.array([-3, 1, 0, 0])
        A = np.array([1, 2, 1, 0, -1, 1, 0, 1]).reshape((2, -1))
        b = np.array([4, 1]).reshape((2, -1))

        return c, A, b

    def ppdd():
        c = np.array([13, 10, 6])
        A = np.array([5, 1, 3, 3, 1, 0]).reshape((2, -1))
        b = np.array([8, 3]).reshape((2, -1))

        return c, A, b

    def mid():
        c = np.array([4, 2, 2])
        A = np.array([2, 6, 1,
                      1, 1, 1]).reshape((2, -1))
        b = np.array([4, 3]).reshape((2, -1))

        return c, A, b

    def mid1():
        c = np.array([2, 3, 4, 3, 0, 0])
        A = np.array([1, 3, 0, 4, 1, 0, 0,
                      2, 1, 0, 0, 0, 1, 0,
                      0, 1, 4, 1, 0, 0, 1]).reshape((3, -1))
        b = np.array([5, 3, 4]).reshape((3, -1))

        return c, A, b

    c, A, b = mid1()
    x, B, N = two_phase_simplex(A, b, c)
        
