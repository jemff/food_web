from size_based_ecosystem import *

depth = 20 #Previously 5 has worked well.
layers = 100 #5 works well.
log_size_range = 12 # 9.5 #8 did NOT work well.
size_classes = 5
t_end = 1/2*1/365
lam = 8
time_step = 1/500000 #One minute #0.00001
res_max = 15
simulate = True
verbose = False
daily_cycle = 365*2*np.pi


mass_vector = np.array([1, 20, 400, 8000, 16000]) #np.array([1, 30, 300, 400, 800, 16000])
from scipy import stats
obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
logn = stats.lognorm.pdf(obj.x, 1, 0)
print(logn, "Logn")

res_start = 5*logn


water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, time_step = time_step, replacement = lam, advection = 1, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj)
eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, loss = 'constr')
eco.population_setter(np.array([1, 1, 1, 1, 0.1]) )#, 1, 1, 1, 0.1]))
#eco.strategy_setter(np.sqrt(eco.strategy_matrix.flatten())) THis is for the L2 version... Quantum fish ahoy
print(graph_builder(eco), "Original graph")
seq_nash = sequential_nash(eco, verbose = True)
eco.strategy_setter(seq_nash)
print(graph_builder(eco), "New graph")


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
        eco.parameters.layered_attack = 1/2*(1.00001+np.cos(t*daily_cycle))*params.layered_attack
        x_res = sequential_nash(eco, verbose = verbose)
        eco.strategy_setter(x_res)
        print(eco.populations, min(eco.water.res_counts), max(eco.water.res_counts), eco.total_growth(x_res))

        print(t)

    plt.figure()
    for i in range(size_classes):
        plt.plot(obj.x, x_res[i * layers:(i + 1) * layers])
    plt.plot(obj.x, eco.water.res_counts)
    plt.show()
    print(eco.populations, eco.total_growth(x_res))