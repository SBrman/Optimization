#! python3

__author__ = 'Simanta Barman'

import re
import numpy as np
import logging
import matplotlib.pyplot as plt
from sympy import diff, lambdify, Symbol, log, exp, pprint, sympify, Matrix, N
from tabulate import tabulate
from inspect import getfullargspec, getsource, isgeneratorfunction
from pylatex import Document, Section, Subsection, Math, VectorName, Command
from pylatex import Matrix as latexMatrix

logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


class Function:
    def __init__(self, f, problem=None):
        if not problem:
            self.problem = 'min'
        else:
            self.problem = problem
            
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
        self.G = ''
        self.H = ''

    def __repr__(self):
        args = ', '.join([str(arg) for arg in self.args])
        return f'{self.name}({args}) = {self.func_str}'

    @property
    def gradient(self):
        if type(self.G) != str:
            return self.G.reshape(-1, 1)
        self.G = np.array([diff(self.func_str, arg) for arg in self.args])
        return self.G.reshape(-1, 1)
    
    @property
    def hessian(self):
        if type(self.H) != str:
            return self.H
        self.H = np.array([[diff(str(func), arg) for func in self.G] for arg in self.args])
        return self.H

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

    def __get_arg_dict(self, argvals):
        if type(argvals) is not np.ndarray:
            argvals = np.array(argvals)
        return dict(zip(self.args, argvals.reshape(1, -1)[0]))
        
    def get_numerical_gradient(self, argvals):
        arg_dict = self.__get_arg_dict(argvals)
        return np.array([[i[0].subs(arg_dict)] for i in self.gradient], dtype=float)

    def get_numerical_hessian(self, argvals):
        arg_dict = self.__get_arg_dict(argvals)
        return np.array([[elem.subs(arg_dict) for elem in row] for row in self.hessian], dtype=float)

    def differentiate(self, wrt):
        """
        Generator that will return each derivative from wrt if wrt is a list
        otherwise this will act like a function and return derivative W.R.T. wrt
        """
        if type(wrt) in {tuple, set, list, np.array}:
            for arg in wrt:
                yield diff(self.func_str, arg)
        else:
            yield diff(self.func_str, wrt)

    def __get_initial_guess(self, initial_guess, epsilon):

        wrt = self.args[0]
        if len(self.args) > 1:
            raise TypeError("Can't work with a f(X) where X ∈ ℝⁿ")
        g = lambdify(wrt, list(self.differentiate(wrt)).pop(), 'numpy')
        if initial_guess == None:
            initial_guess = []
            for x0 in range(1000, -1000, -100):
                if g(x0) < 0:
                    initial_guess.append(x0)
                    break
            for x1 in range(-1000, 1000, 100):
                if g(x1) > 0:
                    initial_guess.append(x1)
                    break
        elif len(initial_guess) == 2:
            pass
        else:
            raise TypeError("Initial guess must be a array of two numbers.")
        
        return g, initial_guess

    def bisection(self, initial_guess=None, epsilon=0.0001):

        g, (x0, x1) = self.__get_initial_guess(initial_guess, epsilon)
        report = [["x_left", "x_right", "x_mid", "g(x_mid)", "g(x_mid) > 0"]]
        while abs(x0 - x1) > epsilon:
            x_mid = (x0 + x1) / 2
            report.append([x0, x1, x_mid, g(x_mid), bool(g(x_mid) > 0)])
            if g(x_mid) > 0:
                x1 = x_mid
            elif g(x_mid) < 0:
                x0 = x_mid
            else:
                break
        logging.debug(tabulate(report, tablefmt='fancy_grid'))
        logging.debug(f"Solution = {x_mid}")
        return x_mid

    def golden_search(self, initial_guess=None, epsilon=0.0001):
        _, (x0, x1) = self.__get_initial_guess(initial_guess, epsilon)
        phi = (3 - 5**0.5) / 2
        report = [["x_left", "x_right", "f(x0_new)", "f(x1_new)", "f(x0_new) < f(x1_new)"]]
        while abs(x0 - x1) > epsilon:
            x0_new = phi*x1 + (1 - phi)*x0
            x1_new = phi*x0 + (1 - phi)*x1
            report.append([x0_new, x1_new, self.f(x0_new), self.f(x1_new),
                           bool(self.f(x0_new) < self.f(x1_new))])
            if self.f(x0_new) < self.f(x1_new):
                x1 = x1_new
            elif self.f(x0_new) > self.f(x1_new):
                x0 = x0_new
            else:
                break
        logging.debug(tabulate(report, tablefmt='fancy_grid'))
        return (x0_new + x1_new) / 2

    def __f(self, x_vector):
        """
        Return vectorized funtion
        """
        return np.array([self.f(*args) for args in x_vector.reshape(1, -1)])
        
    def __backtracking_line_search(self, x_k, alpha=None, beta=None):
        """
        Returns alpha_k
        """
        # Step 1
        alpha = 0.1 #np.random.uniform(1e-3, 0.5) if alpha == None else alpha    # alpha \in (0, 0.5)
        beta = 0.1 #np.random.uniform(1e-3, 1) if beta == None else beta         # beta \in (0, 1)

        # Step 2
        t = 1

        # Step 3
        grad_f = self.get_numerical_gradient(x_k)
        d_k = - grad_f

        logging.debug(self.f(*(x_k + t*d_k).flatten()), self.f(*x_k.flatten()), alpha*t*(grad_f.transpose().dot(d_k))[0][0])
        
        while True:
            if self.f(*(x_k + t*d_k).flatten()) <= self.f(*x_k.flatten()) + alpha*t*(grad_f.transpose().dot(d_k))[0][0]:
                return t
            else:
                t = beta * t

    def __newton_direction(self, x_k, grad_f_xk):
        """
        Return d_k for Newton's method.
        """
        hessian_inv = np.linalg.inv(self.get_numerical_hessian(x_k)) 
        return - hessian_inv.dot(grad_f_xk)

    def __optimizer(self, x_0, epsilon, method='GD'):
        x_k = x_0
        k = 0
        while True:
            grad_f_xk = self.get_numerical_gradient(x_k)
            
            # Step 1
            if np.linalg.norm(grad_f_xk) <= epsilon:
                return x_k

            # Step 2
            if method == 'GD':
                d_k = - grad_f_xk
            elif method == 'NR':
                d_k = self.__newton_direction(x_k, grad_f_xk) 
            else:
                raise TypeError(f'No method named "{method}" is implemented.')
            
            # Step 3
            alpha_k = self.__backtracking_line_search(x_k)

            # Step 4
            x_k = x_k + alpha_k * d_k
            k += 1

    def gradient_descent(self, x, epsilon=0.01):
        return self.__optimizer(x, epsilon, method='GD')

    def newton_raphson(self, x, epsilon=0.01):
        return self.__optimizer(x, epsilon, method='NR')


    def print_gradient(self):
        print("Gradient, ∇f(x) =")
        print(tabulate(self.gradient, tablefmt='fancy_grid'), '\n')

    def print_hessian(self):
        print("Hessian, Hf(x) =")
        print(tabulate(self.hessian, tablefmt='fancy_grid'), '\n')

    def plot(self, soln=None, wrt=None):
        if wrt==None:
            wrt = self.args[0]
        sign = 1 if self.problem == 'min' else -1
        x = np.linspace(0, 5)
        f = np.vectorize(self.f)
        g = lambdify(wrt, list(self.differentiate(wrt)).pop(), 'numpy')

        plt.figure(1)
        plt.subplot(211)
        plt.plot(x, sign*f(x), soln, sign*f(soln), 'o')
        plt.ylabel(r'$f(x)$')
        plt.subplot(212)
        plt.plot(x, sign*g(x), soln, sign*g(soln), 'o')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$g(x)$')
        plt.show()

    def generate_latex_file(self):
        doc = Document()
        sections = {'Function': str(self),
                    'Gradient': np.matrix(self.gradient),
                    'Hessian': np.matrix(self.hessian),
                    'Eigenvalues': np.matrix(f.eigen_values(f.hessian)),
                    'Convex': f.is_convex()}

        for sec, mat in sections.items():
            section = Section(sec, numbering=False)
            if type(mat) not in [bool, str]:
                mat = latexMatrix(mat, mtype='b')
            data = Math(data=[mat])
            section.append(data)
            doc.append(section)
            
        # doc.generate_pdf('mat', clean_tex=False)
        return doc.dumps()

    def report(self):
        
        print(f, '\n')
        f.print_gradient()
        f.print_hessian()
        print('Eigenvalues:', f.eigen_values(f.hessian))
        print('\nConvex function:', end=' ')
        print(f.is_convex())

        # soln = f.bisection(initial_guess=[0, 5], epsilon=1e-6)
        # soln = f.golden_search([0, 5])
        # print(f'Optimizer = {soln}')
        # f.plot(soln)
        
        xx = np.random.randint(0, 1, size=(len(self.args), 1))
        gd = f.gradient_descent(xx)
        gds = tabulate(gd, tablefmt='fancy_grid')
        print(f'\nGradient Descent Solution vector = \n{gds}')

        xx = np.random.randint(0, 10, size=(len(self.args), 1))
        nr = f.newton_raphson(xx)
        nrs = tabulate(nr, tablefmt='fancy_grid')
        print(f'\nNewton Raphson Solution vector = \n{nrs}')


if __name__ == "__main__":             
    def f1(x):
        return x**2 + 2*x*2 + 4

    def f2(x, y):
        return x**2 + x*y + y**2

    f3 = lambda x: -(x * exp(-x) / (1 + exp(-x)))

    def f4(x1, x2):
        return exp(x1 + x2) + x1**2 + 3*x2**2 - x1*x2

    f = Function(f4, 'min')
##    f.report()
##    text = f.generate_latex_file()

