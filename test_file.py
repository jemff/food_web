depth = 10 #Previously 5 has worked well.
layers = 40 #5 works well.
log_size_range = 6 #8 did NOT work well.
size_classes = 2


from size_based_ecosystem import *

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

    bounds = optm.Bounds(lower_bound, upper_bound)

    return matrix, one_bound, bounds

obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!


A, one, bounds = constraint_builder(obj.M, size_classes)


constr = ({'type': 'eq', 'fun': lambda x: np.dot(A,x) - 1})
#print(constraint_builder(obj.M, size_classes))

mass_vector =np.exp(np.linspace(0,log_size_range,size_classes)) # np.array([1, 300, 300**2, 10*300**2])
print(mass_vector)
res_start = res_start = 2/(1+np.exp(0.3*obj.x)) #Previously used the constant 1. Now\re fully loaded!
water_start = water_column(obj, res_start, layers = layers)

params = ecosystem_parameters(mass_vector, obj)

print(params.who_eats_who)
print(params.forager_or_not)

print(params.attack_matrix)


eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, loss = 'constr')
strat_temp = eco.strategy_matrix.flatten()
strat_temp[0:layers] = 0
strat_temp[1] = 1
strat_temp[0:layers] = strat_temp[0:layers]/np.dot(eco.ones, np.dot(obj.M, strat_temp[0:layers]))
eco.strategy_setter(strat_temp)
eco.population_setter(np.array([1, 10**(-6)]))
print(np.linalg.norm(eco.loss_function(eco.strategy_matrix.flatten())), "Start loss")

print(eco.one_actor_growth(strat_temp, 0))

opt_obj = optm.minimize(lambda x: (np.linalg.norm(eco.loss_function(x)))**2, x0=eco.strategy_matrix.flatten(),method = 'SLSQP',  constraints = constr, bounds = bounds)
print(opt_obj)

print(eco.loss_function(opt_obj.x))
x_res = opt_obj.x


for k in range(size_classes):
    print(np.dot(np.repeat(1, layers),np.dot(obj.M,x_res[layers*k:layers*(k+1)])))
