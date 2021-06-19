#! python3

"""Chapter 4: Bisection Method for finding optimal solution (Minimization)"""

from sympy import *
import logging

#logging.disable(logging.CRITICAL)
logging.basicConfig(level= logging.INFO, format='%(message)s')

class Minimize:

    def __init__(self, function, a, b, precision=0.1):

        self.function = function
        self.lower = a
        self.upper = b
        self.precision = precision

    def bisect(self):
        """Returns the inerval where the optimal solution lies"""
        x = Symbol('x')
        print('Global Minimum using Bisection Method:')
        logging.info('\n'+ 'k'.rjust(4) + 'a'.rjust(16) + 'b'.rjust(16) + 
                    'midpoint'.rjust(16) + 'dk'.rjust(16))

        # Start of Bisection Algorithm
        # Step 0: Initializing, k, a0, b0
        k, a, b = 1, self.lower, self.upper

        while True:
            # Step 1: Find derivative at midpoint, dk = f'(a+b/2)
            midpoint = (a+b) / 2 
            dk = diff(eval(self.function), x).subs({x: midpoint})

            # Printing the row of the operations preformed
            logging.info(f'{str(k).rjust(4)}{a:>16n}{b:>16n}'
                            f'{midpoint:>16n}{float(dk):>16n}')

            # Step 2:
            a, b = (a, midpoint) if dk > 0 else (midpoint, b)
            
            # Step 3: Terminating condition check and terminate or return to 1
            k += 1
            if b - a < self.precision:
                midpoint = (a+b)/2
                print()
                break

        return midpoint


M = Minimize('(x-1)**2', 0, 3, 0.0001)
print('Global Minimum: ', M.bisect())
