'''
    https://www.sfu.ca/~ssurjano/index.html
'''

from math import *
import numpy as np

class TEST_FUNC(object):
    num_func_called = 0
    funcs = ['eggholder',
            'ackley',
#            'bukin',
#            'cross_in_tray',
            'sphere',
#            'bohachevsky',
            'sum_squares',
            'sum_of_different_powers',
#            'booth',
#            'matyas',
#            'mccormick',
            'dixon_price',
#            'six_hump_camel',
#            'three_hump_camel',
#            'easom',
#            'michalewicz',
#            'beale',
#            'drop_wave',
            'griewank'] 

    @classmethod
    def eggholder_function(cls, x):
        cls.num_func_called += 1
        x0 = x[:-1]
        x1 = x[1:]
        return np.sum([-(x_next+47)*np.sin(np.sqrt(np.absolute(0.5*x_prev+x_next+47)))-x_prev*np.sin(np.sqrt(np.absolute(x_prev-x_next-47))) for (x_prev, x_next) in zip(x0,x1)])
    
    @classmethod
    def ackley_function(cls, x):
        cls.num_func_called += 1
        return -20*exp(-0.02*sqrt(0.5*sum([i**2 for i in x]))) - \
               exp(0.5*sum([cos(i) for i in x])) + 20 + exp(1)
    
    @classmethod
    def bukin_function(cls, x):    
        cls.num_func_called += 1
        return 100*sqrt(abs(x[1]-0.01*x[0]**2)) + 0.01*abs(x[0] + 10)
    
    @classmethod
    def cross_in_tray_function(cls, x):
        cls.num_func_called += 1
        return round(-0.0001*(abs(sin(x[0])*sin(x[1])*exp(abs(100 -
                                sqrt(sum([i**2 for i in x]))/pi))) + 1)**0.1, 7)
    
    @classmethod
    def sphere_function(cls, x):
        cls.num_func_called += 1
        return sum([i**2 for i in x])
    
    @classmethod
    def bohachevsky_function(cls, x):
        cls.num_func_called += 1
        return x[0]**2 + 2*x[1]**2 - 0.3*cos(3*pi*x[0]) - 0.4*cos(4*pi*x[1]) + 0.7
    
    
    @classmethod
    def sum_squares_function(cls, x):
        cls.num_func_called += 1
        return sum([(i+1)*x[i]**2 for i in range(len(x))])
    
    @classmethod
    def sum_of_different_powers_function(cls, x):
        cls.num_func_called += 1
        return sum([abs(x[i])**(i+2) for i in range(len(x))])
    
    @classmethod
    def booth_function(cls, x):
        cls.num_func_called += 1
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
    
    @classmethod
    def matyas_function(cls, x):
        cls.num_func_called += 1
        return 0.26*sum([i**2 for i in x]) - 0.48*x[0]*x[1]
    
    @classmethod
    def mccormick_function(cls, x):
        cls.num_func_called += 1
        return sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1
    
    @classmethod
    def dixon_price_function(cls, x):
        cls.num_func_called += 1
        return (x[0] - 1)**2 + sum([(i+1)*(2*x[i]**2 - x[i-1])**2
                                    for i in range(1, len(x))])
    
    @classmethod
    def six_hump_camel_function(cls, x):
        cls.num_func_called += 1
        return (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1]\
               + (-4 + 4*x[1]**2)*x[1]**2
    
    @classmethod
    def three_hump_camel_function(cls, x):
        cls.num_func_called += 1
        return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2
    
    @classmethod
    def easom_function(cls, x):
        cls.num_func_called += 1
        return -cos(x[0])*cos(x[1])*exp(-(x[0] - pi)**2 - (x[1] - pi)**2)
    
    @classmethod
    def michalewicz_function(cls, x):
        cls.num_func_called += 1
        return -sum([sin(x[i])*sin((i+1)*x[i]**2/pi)**20 for i in range(len(x))])
    
    @classmethod    
    def beale_function(cls, x):
        cls.num_func_called += 1
        return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + \
               (2.625 - x[0] + x[0]*x[1]**3)**2
    
    @classmethod
    def drop_wave_function(cls, x):
        cls.num_func_called += 1
        return -(1 + cos(12*sqrt(sum([i**2 for i in x]))))/(sum([i**2 for i in x]) + 2)

    @classmethod
    def griewank_function(cls, x):
        cls.num_func_called += 1
        return np.sum([np.power(x[i], 2)/4000 for i in range(len(x))]) - np.prod([np.cos(x[i]/np.sqrt(i+1)) for i in range(len(x))]) + 1
   

    @classmethod
    def get_func_count(cls):
        return cls.num_func_called

    @classmethod
    def reset_func_count(cls):        
        cls.num_func_called = 0
        print('# of function called is initialized to 0')
