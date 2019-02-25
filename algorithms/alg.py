import numpy as np
from algorithms.test_func import TEST_FUNC as tf
from matplotlib import pyplot as plt
import matplotlib.animation
from time import time

class ALG(object):
    def __init__(self, args):
        self.args = args
        self.position_history = list()
        self.correct_flag = False
        self.duration = 0
                    
        # Function name: function, lb, ub, {dim:(min_value, coordinate}
        self.test_func = {
            'eggholder_function':(tf.eggholder_function, -512, 512, \
                        {2:(-959.6407, (512.0, 404.2318)), 
                        5:(-3719.72, (485.590, 436.124, 451.083, 466.431, 421.959)), 
                        10:(-8291.24, (480.85, 431.37, 444.91, 457.55, 471.96, 427.50, 442.09, 455.12, 469.43, 424.94)),
                        20:(-17455.9, (481.0, 431.5, 445.1, 457.9, 472.4, 428.0, 443.0, 456.5, 471.7, 427.3, 442.5, 456.3, 471.6, 427.1, 442.5, 456.3, 471.7, 427.3, 442.8, 456.9))}),
            'ackley_function':(tf.ackley_function, -35, 35, \
                        (lambda x: 0, [0 for _ in range(self.args.dimension)])),
            'bukin_function':(tf.bukin_function, -15, 15, \
                        {2:(0, (-10, 1))}),
            'cross_in_tray_function':(tf.cross_in_tray_function, -10, 10, \
                        {2:(-2.06261, ((1.3491, -1.3491), (1.3491, 1.3491), (-1.3491, 1.3491), (-1.3491, -1.3491)))}),
            'sphere_function':(tf.sphere_function, -5, 5, \
                        (lambda x: 0, [0 for _ in range(self.args.dimension)])),
            'bohachevsky_function':(tf.bohachevsky_function, -10, 10, \
                        {2:(0, (0,0))}),
            'sum_squares_function':(tf.sum_squares_function, -5.12, 5.12, \
                        (lambda x: 0, [0 for _ in range(self.args.dimension)])),
            'sum_of_different_powers_function':(tf.sum_of_different_powers_function, -1, 1, \
                        (lambda x: 0, [0 for _ in range(self.args.dimension)])),
            'booth_function':(tf.booth_function, -10, 10, \
                        {2:(0, (1,3))}),
            'matyas_function':(tf.matyas_function, -10, 10, \
                        {2:(0, (0,0))}),
            'mccormick_function':(tf.mccormick_function, -4, 4, \
                        {2:(-1.9133, (-0.54719, -1.54719))}),
            'dixon_price_function':(tf.dixon_price_function, -10, 10, \
                        (lambda x: 0, [np.power(2, -((np.power(2, x)-2)/np.power(2, x))) for x in range(1, self.args.dimension)])),
            'six_hump_camel_function':(tf.six_hump_camel_function, -3, 3, \
                        {2:(-1.0316, ((0.0898, -0.7126), (-0.0898, 0.7126)))}),
            'three_hump_camel_function':(tf.three_hump_camel_function, -5, 5, \
                        {2:(0, (0,0))}),
            'easom_function':(tf.easom_function, -20, 20, {2:(-1, (np.pi, np.pi))}),
            'michalewicz_function':(tf.michalewicz_function, 0, np.pi,\
                        {2:(-1.8013, (2,20, 1.57)),
                        5:(-4.687658, (None)),
                        10:(-9.66015, (None))}),
            'beale_function':(tf.beale_function, -4.5, 4.5, \
                        {2:(0, (3, 0.5))}),
            'drop_wave_function':(tf.drop_wave_function, -5.12, 5.12, \
                        {2:(-1, (0,0))}),
            'griewank_function':(tf.griewank_function, -600, 600, \
                        (lambda x: 0, [0 for _ in range(self.args.dimension)]))
        }

    def reset(self):
        self.correct_flag = False
        self.start_time = time()
        self.duration = 0
        tf.reset_func_count()

    def get_func_count(self):
        return tf.get_func_count()
        
    def get_agents(self):
        return self.position_history

    def set_global_best(self, global_best):
        self.g_best = global_best

    def get_function_statistics(self, function_name):
        function_info = self.test_func[function_name]
        self.f = function_info[0]
        self.lower_bound = function_info[1]
        self.upper_bound = function_info[2]
        self.dimension = self.args.dimension

        if isinstance(function_info[3], dict):
            if self.dimension not in function_info[3].keys():
                raise KeyError
            else:
                self.min_value = function_info[3][self.dimension][0]
                self.real_coord = function_info[3][self.dimension][1]
        else:
            self.min_value = function_info[3][0](self.dimension)
            self.real_coord = function_info[3][1]

    
    def print_result(self, it, global_best, global_best_eval):
        print('\nAt iteration %d' % it)
        print('Optimization result with tolerance %.5f' % self.args.tolerance)
        print('\tReal global minimum value: %.5f at ' % (self.min_value), end="")
        print(*self.real_coord, sep=',')
        print('\tOptimized global minimum value: %.5f at ' % (global_best_eval), end="")
        print(*global_best, sep=',')
        print('\tNumber of function evaluation: %d' % self.get_func_count())


    def animation(self, alg_name, position_history, function, lb, ub, save=False):
        # 3d plot for 2 input parameters
        grid = np.linspace(lb, ub, int(ub-lb)*10)
        '''    
            Generate grid point by np.meshgrid(x,y)
            ex) X, Y = np.meshgrid(x,y) for x = array([0,1]), y = array([0,1,2]) => Make 3*2 grid
                X = array([[0,1],       Y = array([[0,0],
                           [0,1],                  [1,1],
                           [0,1]])                 [2,2]])            
        '''
   
        print('Make x-y grid') 
        X, Y = np.meshgrid(grid, grid)
        Z = np.array([function([x,y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = np.reshape(Z, X.shape) 
    
        fig = plt.figure()
    
        plt.axes(xlim=(lb, ub), ylim=(lb, ub))
        # Color-mapping Z values in 2d plot
        plt.pcolormesh(X, Y, Z, shading='gouraud')
    
        # Initial agent point
        x = np.array([j[0] for j in position_history[0]])
        y = np.array([j[1] for j in position_history[0]])
        
        sc = plt.scatter(x, y, color='black')
    
        plt.title(function.__name__, loc='left')
        
        def plot(i):
            x = np.array([j[0] for j in position_history[i]])
            y = np.array([j[1] for j in position_history[i]])
            # Scatter update (x,y) through animation
            sc.set_offsets(list(zip(x,y)))
            plt.title('Iteration: {}'.format(i), loc='right')
   
        ani = matplotlib.animation.FuncAnimation(fig, plot, frames=len(position_history)-1)
    
        if save:
            ani.save('%s.mp4' % function.__name__)
    
        plt.show()
    
        
