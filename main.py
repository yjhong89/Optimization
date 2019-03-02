import numpy as np
import tensorflow as tf
import argparse
import logging
import importlib

from algorithms.test_func import TEST_FUNC as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', nargs='+')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--num_agents', type=int)
    parser.add_argument('--dimension', type=int)
    parser.add_argument('--func', type=str)
    parser.add_argument('--tolerance', type=float)
    parser.add_argument('--num_tests', type=int)
    
    # Particle swarm optimization
    parser.add_argument('--w', type=float, default=0.5)
    parser.add_argument('--c1', type=float, default=1)
    parser.add_argument('--c2', type=float, default=1)

    # Cuckoo Search(including Improved version)
    parser.add_argument('--LAMBDA', type=float, default=1.5)
    parser.add_argument('--num_cuckoo', type=int, default=1000)
    parser.add_argument('--pa', type=float, default=0.25)
    parser.add_argument('--step_size', type=float, default=0.02)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--mutation_prob', type=float, default=0.1)
    parser.add_argument('--cuckoo_mutation_range', type=float, default=0.2)
    parser.add_argument('--max_crossover_ratio', type=float, default=0.5)
    parser.add_argument('--prob_genetic_replace', type=float, default=0.5)

    # Conformation Space Annealing
    parser.add_argument('--seed_ratio', type=float, default=0.4)
    parser.add_argument('--max_rounds', type=int, default=3)
    parser.add_argument('--max_crossover_ratio_type1', type=float, default=0.5)
    parser.add_argument('--max_crossover_ratio_type2', type=float, default=0.2)
    parser.add_argument('--mutation_range', type=float, default=0.2)
    parser.add_argument('--num_mutations', type=int, default=1)
    parser.add_argument('--num_unused', type=int, default=10)
    parser.add_argument('--d_cut_reduce', type=float, default=0.983912)
    parser.add_argument('--d_cut_initial', type=float, default=2.0)
    parser.add_argument('--d_cut_low', type=float, default=5.0)
    parser.add_argument('--min_maxiter', type=int, default=1000)

    parser.set_defaults(algorithm=['improved_cuckoo_search', 'particle_swarm_optimization', 'cuckoo_search', 'conformation_space_annealing'], 
                        iterations=100000, 
                        num_agents=16,
                        dimension=5,
                        func='eggholder',
                        tolerance=1e-2,
                        num_tests=100,
                        test_result='test_result_dim5.txt')

    args = parser.parse_args()

    # Set logging format
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s][%(filename)s:%(lineno)d]: %(message)s]')

    with open(args.test_result, 'a') as f:
    
        for alg in args.algorithm:      
            f.write('['+alg+']\n')            
            f.write('\tFor %d tests' % args.num_tests)
            logging.info('For %d tests' % args.num_tests)
            logging.info('%s algorithm' % alg)
            module_name = 'algorithms.' + alg
            module = importlib.import_module(module_name)
            alg_name = ''.join([word[0] for word in alg.split('_')]).upper()
            algorithm = getattr(module, alg_name)
            logging.info('Import %s module' % algorithm)

            for func in tf.funcs:
                num_func_evals = 0
                num_correct = 0
                total_duration = 0
                function_name = func + '_function'
                f.write('\n\t['+function_name+']\n')
                opt = algorithm(args, function_name=function_name)    
    
                for test in range(args.num_tests):
                    opt.run()
                    if opt.correct_flag:
                        num_func_evals += opt.get_func_count()
                        num_correct += 1
                        total_duration += opt.duration

                if num_correct == 0:
                    f.write('\tOptimization fail\n')
                else:
                    num_func_evals /= num_correct
                    total_duration /= num_correct
                    accuracy = num_correct * 100 / args.num_tests
                    f.write('\tAccuracy %.4f within tolerance %.4f\n' % (accuracy, args.tolerance))
                    f.write('\tAverage duration for optimization %.4f secs\n' % total_duration)
                    f.write('\tAverage number of function evaluation: %.4f\n' % num_func_evals)
                    logging.info('\tAccuracy %.4f within tolerance %.4f' % (accuracy, args.tolerance)) 
                    logging.info('\tAverage duration for optimization %.4f secs' % total_duration)
                    logging.info('\tAverage number of function evaluation: %.4f' % num_func_evals)

#                logging.info('Global best position: %s'% opt.g_best)
#    opt.animation(alg_name, opt.position_history, opt.f, opt.lower_bound, opt.upper_bound, save=False)  

