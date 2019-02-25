'''
    Implementation of Cuckoo Search
'''
import math
import numpy as np
from algorithms.alg import ALG
from time import time


class CS(ALG):
    def __init__(self, args, function_name):
        # Call __init__ function of a parent class
        super(CS, self).__init__(args)
        # Get function's statistics
        self.get_function_statistics(function_name)

    def run(self):
        # Initialize the number of function called
        self.reset()

        # Random initial nests
        self.nests = np.random.uniform(self.lower_bound, self.upper_bound, (self.args.num_agents, self.dimension))

        # Random initial cuckoo
        cuckoo = np.random.uniform(self.lower_bound, self.upper_bound, (self.args.num_cuckoo, self.dimension))

        self.position_history.append(np.copy(self.nests))

        # Global best position
        global_best = self.nests[np.array([self.f(x) for x in self.nests]).argmin()]

        # Number of worst nests to be replaced
        num_worst_nests = int(self.args.pa * self.args.num_agents)

        levy_step = self.levy_flight_step(self.args.LAMBDA)


        for t in range(self.args.iterations):
            # Choose a nest randomly and replace by the new solution based on fitness score.
            for c in cuckoo:
                random_val = np.random.randint(0, self.args.num_agents)
                if self.f(c) < self.f(self.nests[random_val]):
                    self.nests[random_val] = c

            # function evaluation of agents
            func_nests = [(self.f(nest), i) for i, nest in enumerate(self.nests)]
            func_cuckoo = [(self.f(c), i) for i, c, in enumerate(cuckoo)]

            # Sort based on fitness score
            func_nests.sort()
            func_cuckoo.sort(reverse=True)

            # Get worst nest index
            worst_nests = [func_nests[-i-1][1] for i in range(num_worst_nests)]

            # Randomly subsitute worst nests with uniform probability distribution
            for i in worst_nests:
                self.nests[i] = np.random.uniform(self.lower_bound, self.upper_bound, (1, self.dimension))

            # Clipping with bound
            # Add new objective, if not, all of 'self.nests' will be updated to latest one.
            self.nests = np.clip(self.nests, self.lower_bound, self.upper_bound)

            # Update cuckoo's position 
            # cuckoo's next generation
            for i in range(min(self.args.num_agents, self.args.num_cuckoo)):
                if func_nests[i][0] < func_cuckoo[i][0]:
                    cuckoo[func_cuckoo[i][1]] = self.nests[func_nests[i][1]]

            # Levy fly for cuckoo
            for i in range(self.args.num_agents):
                stepsize = self.args.step_size * levy_step #* (global_best - cuckoo[i]) 
                cuckoo[i] += stepsize * np.random.normal(0, self.args.sigma, size=self.dimension)

            # Clipping with bound
            cuckoo = np.clip(cuckoo, self.lower_bound, self.upper_bound)
            
            current_best = self.nests[np.array([self.f(x) for x in self.nests]).argmin()]
            global_best_eval = self.f(global_best)
            current_best_eval = self.f(current_best)

            if global_best_eval > current_best_eval:
                global_best = current_best


            self.print_result(t, global_best, global_best_eval)
            # Stop optimization when minimum value is near to real minimum value
            if abs(self.min_value - global_best_eval) <= self.args.tolerance:
                self.correct_flag = True
                self.duration = time() - self.start_time
                break
           
            # Add to position history
            self.position_history.append(np.copy(self.nests))

        self.set_global_best(global_best)


    
    def levy_flight_step(self, LAMBDA):
        '''
        A random walk in which the step-lengths have a heavy-tailed probability distribution. 
        Generate step from levy distribution.
        '''
        sigma1 = np.power((math.gamma(1+LAMBDA) * np.sin(np.pi*LAMBDA/2)) \
                / (math.gamma((1+LAMBDA)/2)*np.power(2,(LAMBDA-1)/2)), 1/LAMBDA)
        sigma2 = 1

        u = np.random.normal(0, sigma1, size=self.dimension)
        v = np.random.normal(0, sigma2, size=self.dimension)
        step = u / np.power(np.abs(v), 1/LAMBDA)

        return step
