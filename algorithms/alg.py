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
                    
        # Function name: function, {dim:(min_value, coordinate, limits)}
        self.test_func = {
            'eggholder_function':(tf.eggholder_function, \
                        {2:(-959.6407, (512.0, 404.2318), [(-512, 512) for _ in range(2)]), 
                        5:(-3719.72, (485.590, 436.124, 451.083, 466.431, 421.959), [(-512, 512) for _ in range(5)]), 
                        10:(-8291.24, (480.85, 431.37, 444.91, 457.55, 471.96, 427.50, 442.09, 455.12, 469.43, 424.94), [(-512, 512) for _ in range(10)]),
                        20:(-17455.9, (481.0, 431.5, 445.1, 457.9, 472.4, 428.0, 443.0, 456.5, 471.7, 427.3, 442.5, 456.3, 471.6, 427.1, 442.5, 456.3, 471.7, 427.3, 442.8, 456.9), [(-512, 512) for _ in range(20)])}),
            'ackley_function':(tf.ackley_function, \
                        lambda x: (0, [0 for _ in range(x)], [(-32.768, 32.768) for _ in range(x)])),
            'bukin_function':(tf.bukin_function, \
                        {2:(0, (-10, 1), [(-15, -5), (-3, 3)])}),
            'cross_in_tray_function':(tf.cross_in_tray_function, \
                        {2:(-2.06261, ((1.3491, -1.3491), (1.3491, 1.3491), (-1.3491, 1.3491), (-1.3491, -1.3491)), [(-10, 10) for _ in range(2)])}),
            'sphere_function':(tf.sphere_function, \
                        lambda x: (0, [0 for _ in range(x)], [(-5, 5) for _ in range(x)])),
            'bohachevsky_function':(tf.bohachevsky_function, \
                        {2:(0, (0,0), [(-10, 10) for _ in range(2)])}),
            'sum_squares_function':(tf.sum_squares_function, \
                        lambda x: (0, [0 for _ in range(x)], [(-10, 10) for _ in range(x)])),
            'sum_of_different_powers_function':(tf.sum_of_different_powers_function, \
                        lambda x: (0, [0 for _ in range(x)], [(-1, 1) for _ in range(x)])),
            'booth_function':(tf.booth_function, \
                        {2:(0, (1,3), [(-10, 10) for _ in range(2)])}),
            'matyas_function':(tf.matyas_function, \
                        {2:(0, (0,0), [(-10, 10) for _ in range(2)])}),
            'mccormick_function':(tf.mccormick_function, \
                        {2:(-1.9133, (-0.54719, -1.54719), [(-1.5, 4), (-3, 4)])}),
            'dixon_price_function':(tf.dixon_price_function, \
                        lambda x: (0, [np.power(2, -((np.power(2, i)-2)/np.power(2, i))) for i in range(1, x+1)], [(-10, 10) for _ in range(x)])),
            'six_hump_camel_function':(tf.six_hump_camel_function, \
                        {2:(-1.0316, ((0.0898, -0.7126), (-0.0898, 0.7126)), [(-3, 3), (-2, 2)])}),
            'three_hump_camel_function':(tf.three_hump_camel_function, \
                        {2:(0, (0,0), [(-5, 5) for _ in range(2)])}),
            'easom_function':(tf.easom_function, {2:(-1, (np.pi, np.pi), [(-100, 100) for _ in range(2)])}),
            'michalewicz_function':(tf.michalewicz_function, \
                        {2:(-1.8013, (2,20, 1.57), [(0, np.pi) for _ in range(2)]),
                        5:(-4.687658, (None), [(0, np.pi) for _ in range(5)]),
                        10:(-9.66015, (None), [(0, np.pi) for _ in range(10)])}),
            'beale_function':(tf.beale_function, \
                        {2:(0, (3, 0.5), [(-4.5, 4.5) for _ in range(2)])}),
            'drop_wave_function':(tf.drop_wave_function, \
                        {2:(-1, (0,0), [(-5.12, 5.12) for _ in range(2)])}),
            'griewank_function':(tf.griewank_function, \
                        lambda x: (0, [0 for _ in range(x)], [(-600, 600) for _ in range(x)]))
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
        self.dimension = self.args.dimension

        if isinstance(function_info[1], dict):
            if self.dimension not in function_info[1].keys():
                raise KeyError
            else:
                f_info = function_info[1][self.dimension]
        else:
            f_info = function_info[1](self.dimension)

        self.min_value = f_info[0]
        self.real_coord = f_info[1]
        self.limits = f_info[2]

    def make_new(self, num_new):
        target = list()
        for i in range(self.dimension):
            target.append(np.random.uniform(self.limits[i][0], self.limits[i][1], (num_new,)))
        # Make (self.args.num_agents, self.dimension)
        target = np.stack(target, axis=1)
        
        return target

    def clip_bound(self, before_clip):
        after_clip = list()
        for i in range(self.dimension):
            after_clip.append(np.clip(before_clip[:,i], self.limits[i][0], self.limits[i][1]))
        after_clip = np.stack(after_clip, axis=1)

        return after_clip        


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
    
        
