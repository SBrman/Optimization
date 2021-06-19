from Simplex_method import *

def primal():
    """Minimization problem"""
    c = np.array([13, 10, 6])
    A = np.array([5,1,3,3,1,0]).reshape((2, -1))
    b = np.array([8,3]).reshape((2, -1))

    return c, A, b

def dual():
    """Maximization problem so return mnegative c"""
    c = np.array([8, 3, 0, 0, 0])
    A = np.array([[5, 3, 1, 0, 0],
                  [1, 1, 0, 1, 0],
                  [3, 0, 0, 0, 1]])
    b = np.array([13, 10, 6]).reshape((3, -1))

    return -c, A, b

c, A, b = dual()
x, B, N = two_phase_simplex(A, b, c)
