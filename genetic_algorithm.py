#! python3

"""Chapter 4: Genetic Algorithm for finding optimal solution (Minimization)"""

import numpy as np
from math import *
import logging
import matplotlib.pyplot as plt

logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def f(x):
    """Returns f(x)"""
    return x**2 + sin(5*x) + cos(20*x) + 5              # here, f(x)>0

def plt_plot(xb, m_child, generation, nth_child):
    step = np.linspace(-4, 4, 1000)
    xx = [x for x in step]
    yy = [f(x) for x in step]

    fig = plt.figure(num=1, figsize=(16,9), clear=True)
    ax1 = fig.add_subplot(1,1,1)

    ax1.plot(xx, yy, 'b-', label='f(x) = x^2 + sin(5x) + cos(20x) + 5')
    ax1.plot([xb, xb], [0, f(xb)], 'g-', label=f'x* = {(round(xb, 5), round(f(xb), 5))}')
    ax1.plot([m_child, m_child], [0, f(m_child)], 'r--', label=f'Child or x\' = {(round(m_child, 5), round(f(m_child), 5))}')
    
    plt.legend(loc="upper left")
    
    plt.xlabel('x') 
    plt.ylabel('y = f(x)')  
    plt.title(f'Genetic  Algorithm (Generation = {generation}, Child = {nth_child})')
    
    plt.axis([-4, 4, 0, 40])
    plt.ion()
    plt.show()
    plt.pause(0.000000001)

def validate_child(child):
    """Returns a valid solution float Ensuring the child has only one decimal point in it."""

    valid_child = ''
    found = 0
    for i in child:
        if found and i=='.':
            continue
        elif i=='.':
            found = 1
        valid_child += i
	
    return float(valid_child)

def mutate(child, probability=0.05):
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
    
def genetic_algorithm(N, mutation_probability):
    """Returns the optimal solution (Approximate value of Global Minima for
    Minimization Problems)"""

    # Step 1: Generate the Initial population and set generation counter g <-- 0
    population = list(np.random.uniform(-4, 4, size=(N)).astype(np.float32))

    for g in range(10):
        children = []
        
        for n in range(N):
            # Step 2.a: Choose two parent solutions from the population
            fx = [(f(solution), solution) for solution in population]
            fx.sort()
            parents = [str(fx[0][1]), str(fx[1][1])]
            logging.info(f'Parents = {parents}')

            # Step 2.b: Combine the Parent solutions and get the child solution
            child = ''
            for p1_digit, p2_digit in zip(parents[0], parents[1]):
                child += np.random.choice([p1_digit, p2_digit])          # Randomly getting genes from parents
            child = validate_child(child)
            logging.info(f'child = {child}')
            
            # Step 3: With probability of p, mutate the new solution
            mutated_child = mutate(child, mutation_probability)
            children.append(mutated_child)
            logging.info(f'Mutated child = {mutated_child}')
            
            # Save x*
            if g == 0:
                x_star = mutated_child
            if f(mutated_child) < f(x_star):
                x_star = mutated_child

            logging.info(f'Generation: {g} --> child = {child}, x* = {x_star}')
            #plt_plot(x_star, mutated_child, g, n)

        population = children

    return x_star

        
for _ in range(10):
    xx = genetic_algorithm(N=100, mutation_probability=0.05)
    print(xx, f(xx))
