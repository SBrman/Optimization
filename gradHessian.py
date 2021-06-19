#! python3

import numpy as np
from sympy import diff, Symbol, log, exp, pprint, sympify, Matrix, N
from inspect import getfullargspec, getsource, isgeneratorfunction
from tabulate import tabulate
import re

class Function:
    def __init__(self, f):
        self.f = f
        args = getfullargspec(f).args
        self.args = []
        for arg in args:
            self.args.append(Symbol(arg))
        if f.__name__ == "<lambda>":
            self.text = re.split(':', getsource(self.f))
            self.name = re.split(' = lambda ', self.text[0].strip())[0]
        elif isgeneratorfunction(f):
            raise TypeError("Function can't be a generator.")
        else:
            self.text = re.split('return', getsource(self.f))
            self.name = f.__name__
        self.func_str = self.text[-1].strip()

    def __repr__(self):
        args = ', '.join([str(arg) for arg in self.args])
        return f'{self.name}({args}) = {self.func_str}'
    
    @property
    def gradient(self):
        self.G = np.transpose(np.array([diff(self.func_str, arg) for arg in self.args]))
        return self.G.reshape(-1,1)
    
    @property
    def hessian(self):
        self.H = np.array([[diff(str(func), arg) for func in self.G] for arg in self.args])
        return self.H

    def print_gradient(self):
        print("Gradient, ∇f(x) =")
        print(tabulate(self.gradient, tablefmt='fancy_grid'), '\n')

    def print_hessian(self):
        print("Hessian, Hf(x) =")
        print(tabulate(self.hessian, tablefmt='fancy_grid'), '\n')

    def eigen_values(self, mat):
        A = Matrix(mat)
        vals = [N(i, 4).as_real_imag()[0] for i in A.eigenvals()]
        return vals

    def is_convex(self, verbose=0):
        eigen_values = self.eigen_values(self.hessian)
        
        if verbose:
            self.print_gradient()
            self.print_hessian()
            print('Eigenvalues:', eigen_values, '\n')

        try:
            if all(1 if e >= 0 else 0 for e in eigen_values):
                if verbose:
                    print(f'The function, {self} is a convex function.')
                return True
            else:
                return False
        except TypeError:
            print('\n\nCan not determine if the function is convex or not.')

if __name__ == "__main__":
    #func = lambda x1, x2, x3: x1**2 + x1*x2 + x1*(exp(x3)) + x2*log(x3)
    def f(x1, x2, x3):
        return x1**2 + x1*x2 + x1*(exp(x3)) + x2*log(x3)

    def f1(x, y, z):
        return 2*x**2 + 2*y**2 + z**2 - 2*x*y - 2*x*z - 6*y + 7

    def f2(x, y):
        return 1 - (1 / (1 + x**2 + y**2))

    def ff(x1, x2):
        return 2*x1**2 + 2*x1*x2 + x2**2 - 10*x1 -10*x2

    def fn(x, y, z):
        return z**2 - x*y

    def fv(r, h):
        π = np.pi
        return π * r**2 * h

    def fff(x, y, z):
        return 2*x*y + 2*y*z + 2*z*x

    def L(x, y, z):
        return 2*x*y + 2*y*z + 2*z*x - (x*y*z-64)*λ - α*x - β*y - θ*z

    def f12(x, y):
        return 2*x + 3*y
    
    f = Function(f12)
    print(f, '\n')
    f.print_gradient()
    f.print_hessian()
    print('Eigenvalues:', f.eigen_values(f.hessian))
    print('\nConvex function:', end=' ')
    print(f.is_convex())
