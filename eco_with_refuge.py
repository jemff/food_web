from size_based_ecosystem import *

depth = 1000 #Previously 5 has worked well.
layers = 2 #5 works well.
log_size_range = 12 # 9.5 #8 did NOT work well.
size_classes = 4
t_end = 10 #1/12
lam = 1
time_step = 10**(-5)
res_max = 15
simulate = True
verbose = False
daily_cycle = 365*2*np.pi


mass_vector = np.array([1, 20, 400, 6000]) #np.array([1, 30, 300, 400, 800, 16000])
from scipy import stats
obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
obj.M = np.identity(layers) #/depth


res_start = np.array([10, 0])


water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, time_step = time_step, replacement = lam, advection = 0, diffusion = 0)

#water_start.M = np.identity(layers)/depth #*1/2

params = ecosystem_parameters(mass_vector, obj)
#print(params.who_eats_who, np.sum(params.who_eats_who[:,3]) == 0)
eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, loss = 'constr')
eco.population_setter(np.array([840, 1.6, 0.0001, 0.0001]) )#, 1, 1, 1, 0.1]))
eco.strategy_matrix = eco.strategy_matrix * 500
eco.parameters.layered_attack = 1 * eco.parameters.layered_attack
OG_layered_attack = np.copy(eco.parameters.layered_attack)

#eco.strategy_setter(np.sqrt(eco.strategy_matrix.flatten())) THis is for the L2 version... Quantum fish ahoy
print(graph_builder(eco), "Original graph")
seq_nash = nash_refuge(eco) #nash_refuge(eco, verbose=True) #nash_refuge(eco, verbose = True)
eco.strategy_setter(seq_nash)
print(graph_builder(eco), "New graph")


plt.figure()
for i in range(size_classes):
    plt.scatter(obj.x, seq_nash[i * layers:(i + 1) * layers], label='Creature ' + str(i))
    plt.legend(loc='upper right')
    print(np.dot(eco.ones, np.dot(obj.M, eco.strategy_matrix[i])))
    print(eco.one_actor_growth(eco.strategy_matrix.flatten(), i))
plt.show()

print(eco.strategy_matrix)
print(eco.parameters.layered_attack)
if simulate is True:
    time_span = np.linspace(0, t_end, int(t_end/time_step))
    strategies = np.zeros((time_span.shape[0], size_classes))
    iterator = 0
    for t in time_span:
        eco.population_setter(eco.total_growth()*time_step + eco.populations)
#        print(eco.water.res_counts, "Before eating")

        eco.consume_resources(time_step)
 #       print(eco.water.res_counts, "After eating")
        eco.water.update_resources()
        x_res = nash_refuge(eco, verbose = verbose)
        eco.strategy_setter(x_res)
        #print(eco.populations, min(eco.water.res_counts), max(eco.water.res_counts), eco.total_growth(x_res))
        #print(t)
        strategies[iterator] = eco.strategy_matrix.flatten()[::2]
        iterator += 1
        print(eco.strategy_matrix, eco.populations, eco.water.res_counts)
    for i in range(size_classes):
        plt.figure()
        plt.plot(time_span, strategies[:, i])
        plt.show()

    print(eco.populations, eco.total_growth(x_res))
    print(graph_builder(eco))

#    print("Do you want to continue simulating?")