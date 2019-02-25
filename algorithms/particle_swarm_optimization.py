'''
    Implementation of Particle Swarm Optimization
'''

import numpy as np
from algorithms.alg import ALG
from time import time


class PSO(ALG):
    def __init__(self, args, function_name):
        # Call __init__ function of a parent class
        super(PSO, self).__init__(args) 
        # Get function's statistics
        self.get_function_statistics(function_name)

    def run(self):    
        # Initialize the number of function called
        self.reset()

        # Random initialization of agents between lower bound and upper bound of input parameters
        self.agents = np.random.uniform(self.lower_bound, self.upper_bound, (self.args.num_agents, self.dimension))
    
        # Add to position history
        self.position_history.append(np.copy(self.agents))

        # Particle's best position
        particle_best = self.agents
        # Global best position
        global_best = particle_best[np.array([self.f(x) for x in particle_best]).argmin()]
    
        # Velocity zero initialization
        velocity = np.zeros((self.args.num_agents, self.dimension))

        for t in range(self.args.iterations): 
            # r1 and r2 are random values [0,1] regenrated every velocity update
            r1 = np.random.random((self.args.num_agents, self.dimension))
            r2 = np.random.random((self.args.num_agents, self.dimension))
    
            '''
            Update each particle i's position given velocity equation:
                v_i(t+1) = w*v_i(t)+c1*r1(i's best solution-i's current position)+c2*r2[global solution-i's current position]
            '''
            velocity = self.args.w * velocity + self.args.c1*r1*(particle_best - self.agents) + self.args.c2*r2*(global_best - self.agents)
            self.agents += velocity
            # Clipping with bound
            self.agents = np.clip(self.agents, self.lower_bound, self.upper_bound)
    
            # Update particle's best position
            for i in range(self.args.num_agents):
                if self.f(self.agents[i]) < self.f(particle_best[i]):
                    particle_best[i] = self.agents[i]

            # Update global best position
            global_best = particle_best[np.array([self.f(x) for x in particle_best]).argmin()]   
            global_best_eval = self.f(global_best)
           
            self.print_result(t, global_best, global_best_eval) 
            # Stop optimization when minimum value is near to real miminum value
            if abs(self.min_value - global_best_eval) <= self.args.tolerance:
                self.correct_flag = True
                self.duration = time() - self.start_time
                break
            # Add to position history
            self.position_history.append(np.copy(self.agents))
        
        # Setting global best position
        self.set_global_best(global_best) 
   
