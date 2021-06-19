#! python3

"""Chapter 4: Methods for finding optimal solution (Minimization)"""

import numpy as np
import logging
import matplotlib.pyplot as plt
from sympy import *

#logging.disable(logging.CRITICAL)
logging.basicConfig(level= logging.INFO, format='%(message)s')

FUNCTION_LABEL = 'f(x) = x^2 + sin(5x) + cos(20x) + 5'
def function(x):
    """Returns f(x)"""
    return x**2 + sin(5*x) + cos(20*x) + 5

def plt_plot(xb, current_x, g_or_T, iteration, method):
    """Shows a dynamic plot of the processes SA or GA"""
    
    f = function

    if method == 'SA':
        title = f'Simulated Annealing (T = {g_or_T}, n = {iteration})'
    elif method == 'GA':
        title = f'Genetic  Algorithm (Generation = {g_or_T}, Child = {iteration})'

    step = np.linspace(-4, 4, 1000)
    xx = [x for x in step]
    yy = [f(x) for x in step]

    fig = plt.figure(num=1, figsize=(16,9), clear=True)
    ax1 = fig.add_subplot(1,1,1)

    ax1.plot(xx, yy, 'b-', label=FUNCTION_LABEL)
    ax1.plot([xb, xb], [0, f(xb)], 'g-', label=f'x* = {(round(xb, 5), round(f(xb), 5))}')
    ax1.plot([current_x, current_x], [0, f(current_x)], 'r--', \
        label=f'Child or x\' = {(round(current_x, 5), round(f(current_x), 5))}')
    
    plt.legend(loc="upper left")
    plt.xlabel('x') 
    plt.ylabel('y = f(x)')  
    plt.title(title)
    
    plt.axis([-4, 4, 0, 40])
    plt.ion()
    plt.show()
    plt.pause(1e-7)

class Minimize:

    def __init__(self, function):
        self.function = function

    def bisection(self, a, b, precision=0.1):
        """Returns the inerval where the optimal solution lies"""

        print('\n' + '-'*70 + '\n' + 'Global Minimum using Bisection Method'.center(70)
                + '\n' + '-'*70)
        logging.info('\n'+ 'k'.rjust(4) + 'a'.rjust(16) + 'b'.rjust(16) + 
                    'midpoint'.rjust(16) + 'dk'.rjust(16))

        # Start of Bisection Algorithm
        # Step 0: Initializing, k, a0, b0
        k, a, b = 1, a, b

        while True:
            # Step 1: Find derivative at midpoint, dk = f'(a+b/2)
            midpoint = (a+b) / 2
            x = Symbol('x')
            f = self.function(x)
            dk = diff(f, x).subs({x: midpoint})

            # Printing the row of the operations preformed
            logging.info(f'{str(k).rjust(4)}{a:>16n}{b:>16n}'
                            f'{midpoint:>16n}{float(dk):>16n}')

            # Step 2:
            a, b = (a, midpoint) if dk > 0 else (midpoint, b)
            
            # Step 3: Terminating condition check and terminate or return to 1
            k += 1
            if b - a < precision:
                midpoint = (a+b)/2
                print()
                break

        return midpoint

    def simulated_annealing(self, n, k, To, Tf):
        """Returns the approximate global minima for an optimization problem (Minimization)
        using Simulated Annealing."""

        print('\n' + '-'*70 + '\n' + 'Global Minimum using Simulated Annealing'.center(70) 
                + '\n' + '-'*70)

        def __is_feasible(x):
            """Returns boolean based on Constraints"""
            result = True if -4 <= x <= 4 else False
            return result

        def __generate_xprime(x, neighbour=0.5):
            """Returns generated x\'"""
            while True:
                x = np.random.uniform(x - neighbour, x + neighbour)
                if __is_feasible(x):
                    return x
        
        # Starting Simulated Annealing
        f = self.function

        # Step 1: Choose an initial feasible solution x from X
        x = np.random.uniform(-4, 4)
        
        # Step 2: x* <-- x
        x_best = x

        # Step 3: T <-- To          where, To = initial highest temp
        T = To

        while True:
            
            # Step 4: Repeat the following steps with same temperature
            for iteration in range(n):

                # Step 4.a: Generate x' in feasible region
                x_prime = __generate_xprime(x)
                
                # Step 4.b:
                x_best = x_prime if f(x_prime) <= f(x_best) else x_best
                
                # Step 4.c:
                if f(x_prime) < f(x):
                    x = x_prime
                    
                # Step 4.d:
                else:
                    probability = exp(( (f(x) - f(x_prime)) / T))
                    if probability >= np.random.uniform(0, 1):
                        x = x_prime
                        
                # Plotting (OPTIONAL)
                plt_plot(x_best, x_prime, T, iteration+1, 'SA')
                
            # Step 5: Lower the Temperature
            if T > Tf:              # Where, Tf = final lowest temp
                T *= k              #         k = multiplier (0 < k <= 1)
                
            # Step 6:
            else:
                return x_best

    def genetic_algorithm(self, N, mutation_probability):
        """Returns the approximate global minima for an optimization problem (Minimization)
        using Genetic Algorithm."""

        print('\n' + '-'*70 + '\n' + 'Global Minimum using Genetic Algorithm'.center(70) 
                + '\n' + '-'*70)

        def __validate_child(child):
            """Returns a valid solution float Ensuring the child has only one decimal point
            in it."""

            valid_child = ''
            found = 0
            for i in child:
                if found and i=='.':
                    continue
                elif i=='.':
                    found = 1
                valid_child += i
            
            return float(valid_child)

        def __mutate(child, probability=0.05):
            """Returns a mutated child"""
            
            # if there is a negative sign then ignoring the index of the sign for swapping
            start = 1 if child >= 0 else 0

            # Taking all the elements of child in a list
            child = list(str(child))

            # swap_range = list of indexes from where 2 indeces will be swapped randomly
            if '.' in child:
                # only taking the indeces of the digits after the decimal for swapping
                swap_range = range(child.index('.')+1, len(child))
            else:
                # Ignoring negative signs index if exists
                swap_range = range(start, len(child))

            # if a random probability <= probability of mutating the, make the swap
            if np.random.uniform() <= probability:
                i, j = np.random.choice(swap_range, 2)

                # Temporary variable is not required in python for swap
                child[i], child[j] = child[j], child[i]

            m_child = ''.join(child)
            return float(m_child)

        # Starting Genetic Algorithm
        f = self.function

        # Step 1: Generate the Initial population and set generation counter g <-- 0
        population = list(np.random.uniform(-4, 4, size=(N)).astype(np.float32))

        for g in range(10):
            children = []

            for n in range(N):
                # Step 2.a: Choose two parent solutions from the population
                fx = [(f(solution), solution) for solution in population]
                fx.sort()
                parents = [str(fx[0][1]), str(fx[1][1])]
                #logging.info(f'Parents = {parents}')

                # Step 2.b: Combine the Parent solutions and get the child solution
                child = ''
                for p1_digit, p2_digit in zip(parents[0], parents[1]):
                    child += np.random.choice([p1_digit, p2_digit])          # Randomly getting genes from parents
                child = __validate_child(child)
                #logging.info(f'child = {child}')
                
                # Step 3: With probability of p, mutate the new solution
                mutated_child = __mutate(child, mutation_probability)
                children.append(mutated_child)
                #logging.info(f'Mutated child = {mutated_child}')
                
                # Save x*
                if g == 0:
                    x_star = mutated_child
                if f(mutated_child) < f(x_star):
                    x_star = mutated_child

                #logging.info(f'Generation: {g} --> child = {child}, x* = {x_star}')
                plt_plot(x_star, mutated_child, g, n, 'GA')

            population = children

        return x_star
        

M = Minimize(function)
print('Approximate Global Minimum (Bisection): ', M.bisection(a=-5, b=5, precision=0.01))
print('Approximate Global Minimum (SA): ', M.simulated_annealing(n=30, k=0.75, To=1000, Tf=1))
print('Approximate Global Minimum (GA): ', M.genetic_algorithm(N=50, mutation_probability=0.02))
