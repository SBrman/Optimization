#! python3

"""Chapter 4: Simulated Annealing for finding optimal solution (Minimization)"""

import numpy as np
from networkx.algorithms.shortest_paths.generic import shortest_path
##class Problem:
##    def __init__(self, obj_func, constraints):
##        self.obj_func = obj_func
##        self.constraints = constraints
##
##
##class Cooling_schedule:
##    def __init__(self, To, Tf, n, k):
##        self.To = To
##        self.Tf = Tf
##        self.n = n
##        self.k = k


def sim_anneal():#Problem, Cooling_schedule):
    
    """Performs simulated Annealing on an optimization problem (Minimization)
and returns the global minima."""
        
    cost = np.array([  [ 8.7, 13.5, 11. , 10.2,  6.3,  6.4, 12.4, 13.5,  9.7,  6.8],
                       [ 5.8, 12.4,  8.8, 15.1, 14.7, 13.9,  9. ,  5.2,  8.7, 12.7],
                       [11.8,  9.8, 12.4, 12.6,  6.3, 13.3,  7.7, 10.2, 13.6, 13.5],
                       [ 5.4,  6.9, 11.1,  8.4, 14.2, 10.9, 10.7, 11.7,  8.1,  8.9],
                       [ 4.9, 12.5, 10.8,  6.6, 12. ,  6.8, 11.9,  9.2,  9.5, 10. ],
                       [14.4,  9.7,  8.6, 11.6,  8. ,  6.7, 12.7,  5.9,  7.6, 14.1],
                       [ 7.7,  9.6,  6.4,  9.9,  9.2, 13.3, 12.3, 14.7, 15. ,  5.1],
                       [ 7.6, 14.7,  7.1,  5.5,  5.5,  9.7,  9.7, 14.4,  7.9, 15.1],
                       [10.9, 12.5,  9.2, 12.1, 14.8,  6.4,  6.1, 10.5, 12.5, 10.8],
                       [12. ,  5.9, 13.1, 10. , 11.9, 10. ,  8.6,  8.4, 10.7,  5.3]])

    customers = np.array([ [0., 1., 1., 0., 0., 0., 0., 1., 0., 0.],
                           [0., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
                           [0., 0., 1., 0., 2., 0., 0., 0., 0., 0.],
                           [2., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 1., 1., 0., 1., 0.],
                           [2., 1., 0., 0., 1., 1., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 2., 0., 0.],
                           [0., 2., 0., 0., 0., 0., 0., 2., 0., 0.],
                           [0., 0., 0., 1., 0., 0., 0., 0., 2., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    G = nx.grid_2d_graph(10, 10)
    
    def get_cost_customer(_from, to):
        paths = shortest_path(G, source=_from, target=to)
        c = []
        for path in paths:
            for node in path:
                cost_node = 0
                if _from == path:
                    continue
                cost_node += cost[node[0]][node[1]]
            
    
    # Step 1: Choose an initial feasible solution x from X
    x = [(i,j) for i, j in zip(random)]
