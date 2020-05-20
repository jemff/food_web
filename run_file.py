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
strat_temp[0:-1] = 0
strat_temp[1] = 1
strat_temp[0:layers] = strat_temp[0:layers]/np.dot(eco.ones, np.dot(obj.M, strat_temp[0:layers]))
strat_temp[layers+1] = strat_temp[1]
eco.strategy_setter(strat_temp)
eco.population_setter(np.array([1, 10**(-3)]))
print(np.linalg.norm(eco.loss_function(eco.strategy_matrix.flatten())), "Start loss")

print(eco.one_actor_growth(strat_temp, 0))

opt_obj = optm.minimize(lambda x: (np.linalg.norm(eco.loss_function(x)))**2, x0=eco.strategy_matrix.flatten(),method = 'SLSQP',  constraints = constr, bounds = bounds)
print(opt_obj)

print(eco.loss_function(opt_obj.x))
x_res = opt_obj.x


#print(x_res)
for k in range(size_classes):
    print(np.dot(np.repeat(1, layers),np.dot(obj.M,x_res[layers*k:layers*(k+1)])))



plt.figure()

for i in range(size_classes):
    plt.plot(obj.x, x_res[i*layers:(i+1)*layers], label = 'Creature ' + str(i))
    plt.legend(loc = 'upper right')
plt.show()

#new_strat = new_strat/np.dot(np.repeat(1, 10),np.dot(new_spectral.M, new_strat))




eco.strategy_setter(x_res)
print(eco.total_growth(x_res))

t_end = 0.1
time_step = 0.001

simulate = True
if simulate is True:
    time_span = np.linspace(0, t_end, int(t_end/time_step))

    for t in time_span:
        eco.population_setter(eco.total_growth()*time_step + eco.populations)
        eco.consume_resources(time_step)
        eco.water.update_resources()
        x_res = optm.minimize(lambda x: (np.linalg.norm(eco.loss_function(x))) ** 2, x0=eco.strategy_matrix.flatten(),
                                method='SLSQP', constraints=constr, bounds=bounds).x
        eco.strategy_setter(x_res)

        print(t)

    plt.figure()
    for i in range(size_classes):
        plt.plot(obj.x, x_res[i * layers:(i + 1) * layers])

    plt.plot(obj.x, eco.water.res_counts)
    plt.show()
    print(eco.populations, eco.total_growth(x_res))