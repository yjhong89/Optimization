'''
    Implementation of Conformation Space Annealing
'''

import math
import numpy as np
from algorithms.alg import ALG
from scipy import optimize
from time import time as measure_time


class CSA(ALG): 
    def __init__(self, args, function_name):
        # Call __init__ function of a parent class
        super(CSA, self).__init__(args)
        # Get function's statistics
        self.get_function_statistics(function_name)      

        self.num_seeds = int(self.args.num_agents*self.args.seed_ratio)


    def run(self):
        # Initialize the number of function called
        self.reset()

        self.first_bank, self.bank = self.create_banks()

        # Global best position
        global_best = self.bank[np.array([self.f(x) for x in self.bank]).argmin()]

        self.position_history.append(np.copy(self.bank))

        for stage in range(self.args.iterations):
            print('Stage %d' % stage)
            # At the beginning of each stage d_cut is set as d_avg/2
            d_avg = self.get_average_distance()
            print('Average distance among the first bank: %.4f' % d_avg)

            d_cut = d_avg / self.args.d_cut_initial

            self.num_solutions = self.bank.shape[0]

            for round in range(self.args.max_rounds):
                print('\tRound %d' % round)
                # Round is completed, all bank solutions are mareked as unused
                self.bank_status = [False for _ in range(self.num_solutions)]
                time = 0
                while True:
                    print('\t\tTime %d' % time)               
 
                    # Seeds are selected from unused solutions
                    unselected_solution_idxs = np.where(np.asarray(self.bank_status) == False)[0].tolist()
                    seed_idxs = self.get_seeds(round, unselected_solution_idxs, d_avg)
    
                    # Daughter solutions are generated bank
                    seeds = self.bank[seed_idxs]
        
                    # For each seed, 3 types of crossover and mutation methods are used 10 times, generating 30 daughter solution
                    daughter_solutions = list()
                    for seed, seed_idx in zip(seeds, seed_idxs):
                        # First type crossover: Partner solution comes from a bank
                        daughter_solution = self.crossover(seed, seed_idx, round, partner_src='b', max_crossover_ratio=self.args.max_crossover_ratio_type1)
                        # Subsequently energy minimized
                        daughter_solution = self.local_minimize(daughter_solution)
                        daughter_solutions.append(daughter_solution)
        
                        # Second type crossover: Partner solution comes from a first bank
                        daughter_solution = self.crossover(seed, seed_idx, round, partner_src='f', max_crossover_ratio=self.args.max_crossover_ratio_type2)
                        daughter_solution = self.local_minimize(daughter_solution)
                        daughter_solutions.append(daughter_solution) 
            
                        # Mutation   
                        daughter_solution = self.mutation(seed, self.args.mutation_range, self.args.num_mutations)
                        daughter_solution = self.local_minimize(daughter_solution)
                        daughter_solutions.append(daughter_solution)

 
                    # All daughter solution update bank one by one
                    for daughter in daughter_solutions:
                        self.bank_update(daughter, d_cut)
#                    print('Bank status: %s' % self.bank_status)

                    # Update global best position
                    current_best = self.bank[np.array([self.f(x) for x in self.bank]).argmin()]
                    global_best_eval = self.f(global_best)
                    
                    if global_best_eval > self.f(current_best):
                        global_best = current_best

                    # Stop optimization when minimum value is near to real minimum value

                    # Add to position history
                    self.position_history.append(self.bank)
    
                    # Decrease d_cut
                    d_cut = max(d_cut*self.args.d_cut_reduce, d_avg/self.args.d_cut_low)

                    # Cound unused solutions
                    num_unused = self.count_unused_bank(seed_idxs)
#                    print('# of unused solutions: %d\n' % num_unused)
                    
                    time += 1
                    if (num_unused < self.args.num_unused) or (time % 100 == 0):
                        break 


            self.print_result(stage, global_best, global_best_eval)
            if abs(self.min_value - global_best_eval) <= self.args.tolerance:
                self.correct_flag = True
                self.duration = measure_time() - self.start_time
                break


            # At the beginning of a new stage, add randomly-generated and subsequently-minimized solutions to both banks
            new_first_bank, new_bank = self.create_banks()

            # Concatenate to previous banks
            self.first_bank = np.concatenate((self.first_bank, new_first_bank), axis=0)
            self.bank = np.concatenate((self.bank, new_bank), axis=0)


        self.set_global_best(global_best)


    def create_banks(self):
        # First bank contains solutions that are randomly generated and each solutions are locally minimized
        first_bank = self.make_new(self.args.num_agents)
        for i in range(first_bank.shape[0]):
            first_bank[i] = self.local_minimize(first_bank[i])
        # The bank holds the evolving population and initially identical to the first bank
        bank = np.copy(first_bank)

        return first_bank, bank


    def get_seeds(self, round, unselected_idx, d_avg):
        if round == 0:
            seed_idxs = np.random.choice([i for i in range(self.num_solutions - self.args.num_agents, self.num_solutions)], size=self.num_seeds, replace=False).tolist()

        else:
            num_unused = len(unselected_idx)
            if num_unused >= self.num_seeds:
                seed_idxs = list()
                # One is selected randomly, 
                seed_idx = np.random.choice(unselected_idx, size=1, replace=False)[0]
                seed_idxs.append(seed_idx)
    #            print('%d added to seeds' % seed_idx) 
        
                for _ in range(self.num_seeds-1):
                    over_avg_solutions = list()
                    under_avg_solutions = list()
                    unselected_idx.remove(seed_idx)
                    # Distances are measured between the selected seed and non-selected unused solutions.       
                    for i in unselected_idx:
                        distance = self.get_distance(self.bank[seed_idx], self.bank[i])
    #                    print('For %d, distance: %.4f' % (i, distance))
                        if self.get_distance(self.bank[seed_idx], self.bank[i]) > d_avg:
                            over_avg_solutions.append((self.f(self.bank[i]), i))
                        else:
                            under_avg_solutions.append((self.f(self.bank[i]), i))
    
                    over_avg_solutions.sort()
                    if len(over_avg_solutions) == 0:
                        under_avg_solutions.sort()
                        seed_idx = under_avg_solutions[0][1]
                    else:
                        seed_idx = over_avg_solutions[0][1]
    #                print('%d added to seeds' % seed_idx)
                    seed_idxs.append(seed_idx)
    
            else:
                seed_idxs = unselected_idx
                for _ in range(self.num_seeds - len(seed_idxs)):
                    solutions = list()
                    for i in range(self.bank.shape[0]):
                        if i in seed_idxs:
                            continue
                        solutions.append((self.f(self.bank[i]), i))
                    solutions.sort()
                    seed_idx = solutions[0][1]
    #                print('%d added to seeds' % seed_idx)
                    seed_idxs.append(seed_idx)                   

        # Mark selected seeds as used
        for seed_idx in seed_idxs:
            self.bank_status[seed_idx] = True

        return seed_idxs


    def crossover(self, seed, seed_index, round, partner_src='b', max_crossover_ratio=0.5):
        '''
            Crossover between a seed and a bank
            Seed's partner is selected from bank in a random fashion exluding the seed itself
            partner_src: 'b'=bank, 'f'=first bank
        '''

        # First, we copy the seed as a daughter
        daughter = np.copy(seed)
        # Crossover size is set in a random fashion between 1 and half of the total number of variables
        num_crossover = np.random.randint(1, max(2, int(self.dimension*max_crossover_ratio)))

        # Pick the indexes for the crossover
        # Use random sampling with np.random.choice
        crossover_idx = np.random.choice(self.dimension, size=num_crossover, replace=False)

        # At the first round, partner conformations for crossover are selected only among the newly added.
        if round == 0:
            while True:
                partner_idx = np.random.randint(self.num_solutions-self.args.num_agents, self.num_solutions)
                if partner_idx != seed_index:
                    break
        else:
            partner_idx = np.random.randint(0, self.num_solutions)

        if partner_src == 'b':
            daughter[crossover_idx] = self.bank[partner_idx][crossover_idx]
        elif partner_src == 'f':
            daughter[crossover_idx] = self.first_bank[partner_idx][crossover_idx]
        else:
            raise Exception('Not proper partner solution type')
      

        return daughter

    def mutation(self, seed, mutate_range=0.2, num_mutations=1):
        mutation_index = np.random.choice([i for i in np.arange(self.dimension)], size=num_mutations, replace=False)
        daughter = np.copy(seed)

        origin_param = daughter[mutation_index] 

        for i in range(len(origin_param)):
            mutate_param = np.random.uniform((1-mutate_range)*origin_param[i], (1+mutate_range)*origin_param[i])
            daughter[mutation_index[i]] = mutate_param

        return daughter


    def count_unused_bank(self, seed_mask):
        count = 0

        for i in range(len(self.bank_status)):
            if self.bank_status[i] == False:
                count += 1

        return count                        

   
    def bank_update(self, daughter_solution, d_cut):
        '''
            For each daughter solution, dsitance to all the bank solutions are measured to identify its closest solution
        '''
#        print('Bank before update')
#        print(self.bank)

        distances = list()
        worst_bank_evals = self.f(self.bank[0])
        worst_bank_index = 0
        for i in range(self.num_solutions):
            distances.append(self.get_distance(daughter_solution, self.bank[i]))
            bank_evals = self.f(self.bank[i])
            if (worst_bank_evals < bank_evals) and (i > 0):
                worst_bank_evals = bank_evals
                worst_bank_index = i

        # Get closest distance for daughter solution
        closest_distance, closest_idx = distances[np.asarray(distances).argmin()], np.asarray(distances).argmin()
        daughter_eval = self.f(daughter_solution)         

        # If the closest distance is less than or equal to d_cut and daughter solution is more optimzal than closest solution, replace
        if (closest_distance <= d_cut) and (daughter_eval < self.f(self.bank[closest_idx])):
            self.bank[closest_idx] = daughter_solution
            self.bank_status[closest_idx] = False

        # If the closest distance is more than d_cut and daughter solution is more optimal than worst solution, replace
        if (closest_distance > d_cut) and (daughter_eval < worst_bank_evals):
            self.bank[worst_bank_index] = daughter_solution
            self.bank_status[worst_bank_index] = False

        self.bank = self.clip_bound(self.bank)

#        print('Bank after update')
#        print(self.bank)

        # Otherwise, daughter solution is discarded

 
    # Average distance among the first bank solutions
    def get_average_distance(self):
        distance = 0
        count = 0
        for i in range(self.first_bank.shape[0]):
            for j in range(i, self.first_bank.shape[0]):
                count += 1
                distance += self.get_distance(self.first_bank[i], self.first_bank[j])

        return distance / count


    # Euclidian distance of the two solutions
    def get_distance(self, param1, param2):
        return np.sqrt(np.sum((param1-param2)**2))


    def local_minimize(self, param):
        # param range: [(min1, max1), (min2, max2)..]
        param_range = [(self.limits[i][0], self.limits[i][1]) for i in range(self.dimension)]

        # OptimizeResult object
        res = optimize.minimize(self.f, param, method='L-BFGS-B', bounds=param_range, options={'maxiter':self.args.min_maxiter})

        # Local minimum function value
        score = res.fun
        # If local minimization is failed, replace score with very big number
        if np.isnan(score):
            score = 1.e10
        
        # Minimized conformation
        min_param = res.x

        return min_param
                   

