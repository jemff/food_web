depth = 20 #Previously 5 has worked well.
layers = 200 #5 works well.
log_size_range = 12 # 9.5 #8 did NOT work well.
size_classes = 2
t_end = 1
time_step = 0.00001
lam = 1
res_max = 1
simulate = True
verbose = False

from size_based_ecosystem import *

obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!


