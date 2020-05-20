import numpy as np
import scipy.optimize as optm
import matlab.engine
import scipy as scp
import itertools as itertools
import matplotlib.pyplot as plt
from size_based_ecosystem import *





depth = 10 #Previously 5 has worked well.
layers = 10 #5 works well.
log_size_range = 10 #8 did NOT work well.
size_classes = 5

def constraint_builder(M, classes):
    lower_bound = np.zeros(M.shape[0]*classes)
    upper_bound = np.array([np.inf]*M.shape[0]*classes)
    identity_part = np.identity(M.shape[0]*classes)

    matrix_vec = M.sum(axis = 0)
    print(matrix_vec)
    matrix = np.zeros((classes, M.shape[0]*classes))
    for i in range(classes):
        matrix[i, i*M.shape[0]: (i+1)*M.shape[0]] = matrix_vec

    one_bound = np.repeat(1, classes)

#    total_matrix = np.vstack((identity_part, matrix))
#    total_lower = np.concatenate((lower_bound, one_bound), axis = None)
#    total_upper = np.concatenate((upper_bound, one_bound), axis = None)
    bounds = optm.Bounds(lower_bound, upper_bound)

    return matrix, one_bound, one_bound, bounds

obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!

#print(constraint_builder(obj.M, size_classes))

A, lb, ub, bounds = constraint_builder(obj.M, size_classes)

constr = optm.LinearConstraint(A, lb, ub)


mass_vector = np.exp(np.linspace(0,log_size_range,size_classes)) # np.array([1, 300, 300**2, 10*300**2])
print(mass_vector)
res_start = 2/(1+np.exp(0.3*obj.x)) #Previously used the constant 1. Now\re fully loaded!


water_start = water_column(obj, res_start, layers = layers)

params = ecosystem_parameters(mass_vector, obj)

eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, loss = 'l2')


#print(params.layered_foraging.shape)
#print(eco.loss_function(eco.strategy_matrix.flatten()).shape, lb.shape, A.shape)

#print(bounds)
#np.dot(A, eco.strategy_matrix.flatten())
#x_res0 = optm.minimize(eco.loss_function, x0=eco.strategy_matrix.flatten(),method = 'trust-constr',  constraints = constr, bounds = bounds)

x_res = optm.least_squares(eco.loss_function, x0=eco.strategy_matrix.flatten(), bounds = (0, np.inf)).x #optm.minimize(eco.loss_function, x0=eco.strategy_matrix.flatten(),method = 'trust-constr',  constraints = [constr])
#optm.least_squares(eco.loss_function, x0=eco.strategy_matrix.flatten(), bounds = (0, np.inf)).x
new_size = 40 #Off-by-one
new_spectral = spectral_method(depth, new_size - 1)
res_start2 = 2/(1+np.exp(0.3*new_spectral.x))
water_start2 = water_column(new_spectral, res_start2, layers = new_size)
params2 = ecosystem_parameters(mass_vector, new_spectral)

print(x_res)
seed_vec2 = interpolater(x_res, layers, new_size, size_classes, obj, new_spectral)
print(seed_vec2)
eco2 = ecosystem_optimization(mass_vector, new_size, params2, new_spectral, water_start2)
print(eco2.layers, "Eco2shape")

eco2.strategy_setter(seed_vec2)
print(np.dot(np.repeat(1, new_size),np.dot(new_spectral.M,np.repeat(1, new_size))), "What, why no 5")

x_res2 = optm.least_squares(eco2.loss_function, x0=eco2.strategy_matrix.flatten(), tr_solver='lsmr', bounds = (-0.000000001, np.inf)).x

for k in range(size_classes):
    print(np.dot(np.repeat(1, new_size),np.dot(new_spectral.M, x_res2[new_size*k:new_size*(k+1)])))

