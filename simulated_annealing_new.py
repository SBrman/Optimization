#! python3

"""Chapter 4: Simulated Annealing for finding optimal solution (Minimization)"""

import numpy as np
from math import *
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')

def f(x):
    """Returns f(x)"""
    return x**2 + sin(5*x) + cos(20*x) + 5              # here, f(x)>0

def is_feasible(x):
    """Returns boolean based on Constraints"""
    result = True if -4 <= x <= 4 else False
    return result

def generate_xprime(x, neighbour=0.5):
    while True:
        x = np.random.uniform(x - neighbour, x + neighbour)
        if is_feasible(x):
            return x

def plt_plot(xb, xp, T, n):
    step = np.linspace(-4, 4, 1000)
    xx = [x for x in step]
    yy = [f(x) for x in step]

    fig = plt.figure(num=1, figsize=(16,9), clear=True)
    ax1 = fig.add_subplot(1,1,1)

    ax1.plot(xx, yy, 'b-', label='f(x) = x^2 + sin(5x) + cos(20x) + 5')
    ax1.plot([xb, xb], [0, f(xb)], 'g-', label=f'x* = {(round(xb, 5), round(f(xb), 5))}')
    ax1.plot([xp, xp], [0, f(xp)], 'r--', label=f'x\' = {(round(xp, 5), round(f(xp), 5))}')
    ax1.plot(label=f'T = {T}, n = {n}')
    
    plt.legend(loc="upper left")
    
    plt.xlabel('x') 
    plt.ylabel('y = f(x)')  
    plt.title(f'Simulated Annealing (T = {T}, n = {n})')
    
    plt.axis([-4, 4, 0, 40])
    plt.ion()
    plt.show()
    plt.pause(0.000000001)

def sim_anneal(n, k, To, Tf):
    """Performs simulated Annealing on an optimization problem (Minimization)
and returns the global minima."""
            
    # Step 1: Choose an initial feasible solution x from X
    x = np.random.uniform(-4, 4)
    logging.info(f'Initial x = {x}')
    y = f(x)
    
    # Step 2: x* <-- x
    x_best = x

    # Step 3: T <-- To          where, To = initial highest temp
    T = To

    while True:
        
        # Step 4: Repeat the following steps with same temperature
        for iteration in range(n):

            # Step 4.a: Generate x' in feasible region
            x_prime = generate_xprime(x)
            
            # Step 4.b:
            x_best = x_prime if f(x_prime) <= f(x_best) else x_best
            
            # Step 4.c:
            if f(x_prime) < f(x):
                x = x_prime
                
            # Step 4.d:
            else:
                probability = e**( (f(x) - f(x_prime)) / T)
                if probability >= np.random.uniform(0, 1):
                    x = x_prime
                    
            # Plotting (OPTIONAL)
            plt_plot(x_best, x_prime, T, iteration+1)
            
        # Step 5: Lower the Temperature
        if T > Tf:              # Where, Tf = final lowest temp
            T *= k              #         k = multiplier (0 < k <= 1)
            
        # Step 6:
        else:
            return x_best


for _ in range(10):
    print(sim_anneal(n=30, k=0.75, To=1000, Tf=1))
