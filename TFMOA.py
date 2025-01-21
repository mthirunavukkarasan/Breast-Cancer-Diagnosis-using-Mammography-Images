import time
from math import inf
import numpy as np
def TFMOA(agents,objective_function,lb, ub,max_iter):
    num_agents,num_dimensions = agents.shape[0],agents.shape[1]
    # Initialize global best position and fitness
    global_best_position = []
    global_best_fitness = inf
    # Main loop
    ct = time.time()
    conv = np.zeros((max_iter))
    for iter in np.arange(1, max_iter + 1).reshape(-1):
        # Evaluate fitness for each agent
        fitness = np.zeros((num_agents, 1))
        for i in np.arange(1, num_agents + 1).reshape(-1):
            fitness[i] = objective_function(agents[i,:])
            # Update global best position and fitness
        min_fitness, min_index = np.amin(fitness)
        if min_fitness < global_best_fitness:
            global_best_fitness = min_fitness
            global_best_position = agents[min_index,:]
         # Update agents positions
        for i in np.arange(1, num_agents + 1).reshape(-1):
            r1 = np.random.rand(1, num_dimensions)
            r2 = np.random.rand(1, num_dimensions)
            agents[i, :] = agents[i,:] + np.multiply(r1, (global_best_position - agents[i,:])) + np.multiply(r2,(np.mean(agents) - agents[i,:]))
            agents[i, :] = np.amax(agents[i,:], lb)
            agents[i, :] = np.amin(agents[i,:], ub)
        conv[iter] = global_best_fitness
    best_position = global_best_position
    best_fitness = global_best_fitness
    ct = time.time()-ct
    return best_position, conv,best_fitness,ct
