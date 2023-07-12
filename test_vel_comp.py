import numpy as np

def gnerate_rate_with_time_step(input_list, time_step):
    rate = np.zeros( np.shape( input_list ) )
    rate[1 : -2] = (input_list[2 : -1] -  input_list[0 : -3]) / time_step /2
    return rate

a = np.linspace(start = 0, stop = 100, num = 100)

a_v = gnerate_rate_with_time_step(a, 1)

print('a:', a)
print('a_v:', a_v)