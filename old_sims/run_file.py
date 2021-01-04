depth = 20 #Previously 5 has worked well.
layers = 80 #5 works well.
log_size_range = 12 # 9.5 #8 did NOT work well.
size_classes = 5
t_end = 1
time_step = 0.0001
lam = 3
res_max = 15
simulate = False
verbose = False

from size_based_ecosystem import *



obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!


A, one, bounds = constraint_builder(obj.M, size_classes)
constr1 = ({'type': 'eq', 'fun': lambda x: np.dot(A[0,0:layers], x) - 1})
bounds1 = optm.Bounds(np.array([0]*layers), np.array([np.inf]*layers))
constr = ({'type': 'eq', 'fun': lambda x: np.dot(A, x) - 1}) #({'type': 'eq', 'fun': lambda x: np.dot(A,np.dot(obj.vandermonde, x.reshape(layers, size_classes)).flatten()) - 1})

mass_vector = np.array([1, 20, 20**2, 20**3, 20**4]) #np.exp(np.linspace(0,log_size_range,size_classes)) #
from scipy import stats as stats
logn = stats.lognorm.pdf(obj.x, 1, 0)
print(logn, "Logn")


res_start = logn #res_start = scp.stats.lognorm.pdf()#2/(1+np.exp(0.3*obj.x)) #Previously used the constant 1. Now\re fully loaded!

water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, time_step = time_step, replacement = lam, advection = 0.1, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj)
eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, loss = 'constr')
eco.population_setter(np.array([1,0.5,0.25,0.125,0.0625]))

if verbose is True:
    plt.figure()
    plt.plot(obj.x, res_start)
    plt.show()
    print(mass_vector)

    print(params.who_eats_who)
    print(params.forager_or_not)

    print(params.attack_matrix)
    print(eco.strategy_matrix, obj.x)


#print(eco.one_actor_hessian(eco.strategy_matrix.flatten(), 1), "Start hessian")
#print(scp.linalg.eigvalsh(eco.one_actor_hessian(eco.strategy_matrix.flatten(), 1)), "Start eigenvalues")
#strat_temp = eco.strategy_matrix.flatten()
#strat_temp[layers:] = 0
#strat_temp[layers] = 1
#strat_temp[layers:] = strat_temp[layers:]/np.dot(eco.ones, np.dot(obj.M, strat_temp[layers:]))
#strat_temp[layers+1] = strat_temp[1]
#eco.strategy_setter(strat_temp)
#eco.population_setter(np.array([1, 10**(-1)]))
#print(np.linalg.norm(eco.loss_function(eco.strategy_matrix.flatten())), "Start loss")

#print(eco.one_actor_growth(strat_temp, 0))

#opt_obj = optm.minimize(lambda x: loss_func(x, size_classes= size_classes, layers = layers, spec = obj, eco = eco), \
#                        x0=np.dot(obj.vandermonde_inv, eco.strategy_matrix.T).flatten(), \
#                        method = 'SLSQP', constraints = constr, bounds = bounds)

#opt_obj = optm.minimize(lambda x: loss_func(eco.loss_function(x), size_classes= size_classes, layers = layers, spec = obj), x0 = eco.strategy_matrix.flatten(), method = 'SLSQP',  constraints = constr, bounds = bounds)

if verbose is True:
    print(opt_obj)

    print(eco.loss_function(opt_obj.x), eco.one_actor_growth(opt_obj.x, 1))
#x_res = np.dot(obj.vandermonde, opt_obj.x.reshape(layers, size_classes)).flatten() #opt_obj.x

x_res = eco.strategy_matrix.flatten() #opt_obj.x
eco.strategy_setter(x_res)
x_temp = np.copy(x_res) #np.zeros(size_classes*layers)
x_temp2 = np.copy(x_res)
error = 1
while error > 10**(-8):
    print("Wut")
    for k in range(size_classes):
        opt_obj = optm.minimize(lambda x: -eco.one_actor_growth(eco.strategy_replacer(x, k, eco.strategy_matrix.flatten()), k),
                      x0=x_temp[layers * k:layers * (k + 1)], method='SLSQP', constraints=constr1, bounds=bounds1)
#        print(opt_obj.message, opt_obj, k)
        x_temp2[k*layers:(k+1)*layers] = opt_obj.x

    print(x_temp - x_temp2, "The error is here")
    error = np.max(np.abs(x_temp - x_temp2))
    x_temp = np.copy(x_temp2)
plt.figure()
for i in range(size_classes):
    plt.plot(obj.x, np.log(x_temp[i*layers:(i+1)*layers]+0.00001))
plt.show()


plt.figure()
if verbose is True:
#print
    print(opt_obj.jac)
    for k in range(size_classes):
        print(np.dot(np.repeat(1, layers),np.dot(obj.M,x_res[layers*k:layers*(k+1)])))
        print(np.diagonal(eco.one_actor_hessian(eco.strategy_matrix.flatten(), k)))

        eigs = scp.linalg.eigvalsh(eco.one_actor_hessian(eco.strategy_matrix.flatten(), k) - np.diag(np.dot(np.linalg.matrix_power(obj.D,2), x_res[layers*k:layers*(k+1)]) ))
        plt.scatter(np.arange(0, layers, 1), eigs, label = str(k))
        plt.legend(loc = 'upper right')
    plt.show()


    plt.figure()

    for i in range(size_classes):
        plt.plot(obj.x, x_res[i*layers:(i+1)*layers], label = 'Creature ' + str(i))
        plt.legend(loc = 'upper right')
    plt.show()

#new_strat = new_strat/np.dot(np.repeat(1, 10),np.dot(new_spectral.M, new_strat))




#print(eco.total_growth(x_res))
#print(eco.water.res_counts)
#eco.water.update_resources()
#print(eco.water.res_counts)



if simulate is True:
    time_span = np.linspace(0, t_end, int(t_end/time_step))

    for t in time_span:
        eco.population_setter(eco.total_growth()*time_step + eco.populations)
#        print(eco.water.res_counts, "Before eating")

        eco.consume_resources(time_step)
 #       print(eco.water.res_counts, "After eating")
        eco.water.update_resources()
#        opt_obj = optm.minimize(
#            lambda x: loss_func(eco.loss_function(x), size_classes=size_classes, layers=layers, spec=obj),
#            x0=eco.strategy_matrix.flatten(), method='SLSQP', constraints=constr, bounds=bounds)

        x_res = sequential_nash(eco)
        eco.strategy_setter(x_res)
        print(eco.populations, min(eco.water.res_counts), max(eco.water.res_counts), eco.total_growth(x_res))

        print(t)

    plt.figure()
    for i in range(size_classes):
        plt.plot(obj.x, x_res[i * layers:(i + 1) * layers])
    plt.plot(obj.x, eco.water.res_counts)
    plt.show()
    print(eco.populations, eco.total_growth(x_res))