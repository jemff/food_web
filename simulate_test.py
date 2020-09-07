from size_based_ecosystem import *
import matplotlib.animation as animation
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

depth = 20 #Previously 5 has worked well.
layers = 30 #5 works well.
log_size_range = 12 # 9.5 #8 did NOT work well.
size_classes = 3
t_end = 2/730 #30/365
lam = 1
time_step = 10**(-5) #1/500000
res_max = stats.norm.pdf(obj.x, loc = 2)

simulate = False
verbose = False
daily_cycle = 365*2*np.pi
mass_vector = np.array([1, 20, 400]) #np.array([1, 30, 300, 400, 800, 16000])


from scipy import stats
obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
logn = stats.norm.pdf(obj.x, loc = 2)
res_start = 5*logn #0.1*(1-obj.x/depth)


water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, time_step = time_step, replacement = lam, advection = 0.01, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj)
eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, loss = 'constr')
OG_layered_attack = np.copy(eco.parameters.layered_attack)

eco.population_setter(np.array([1, 0.000001, 0.0000001]) )#, 1, 1, 1, 0.1]))
#eco.parameters.handling_times = np.array([0, 0, 0])
#eco.parameters.layered_attack = 0 * OG_layered_attack
#eco.strategy_setter(np.sqrt(eco.strategy_matrix.flatten())) THis is for the L2 version... Quantum fish ahoy



if simulate is True:

    print(graph_builder(eco), "Original graph")
    seq_nash = np.array(sequential_nash(eco, verbose=True))
    print(seq_nash)
    eco.strategy_setter(seq_nash)
    print(graph_builder(eco), "New graph")

    plt.figure()
    for i in range(size_classes):
        plt.plot(obj.x, seq_nash[i] ** 2)
    plt.plot(obj.x, eco.water.res_counts)
    plt.show()

    time_span = np.linspace(0, t_end, int(t_end/time_step))
#    fig, ax = plt.subplots()
    strategy_history = np.zeros((time_span.shape[0], size_classes, layers))
    population_history = np.zeros((time_span.shape[0], size_classes))
    for i, t in enumerate(time_span):
        eco.parameters.layered_attack = 1/2*(1.0000+np.cos(t*daily_cycle))*OG_layered_attack
        x_res = sequential_nash(eco, verbose = verbose)
        eco.strategy_setter(x_res)
        print(eco.total_growth(), t, eco.populations)
        eco.population_setter(eco.total_growth() * time_step + eco.populations)

#        eco.consume_resources(time_step)

#        eco.water.update_resources()

#        print(eco.populations, min(eco.water.res_counts), max(eco.water.res_counts), eco.total_growth(x_res))
        strategy_history[i] = eco.strategy_matrix
        population_history[i] = eco.populations

    np.save('strategies.npy', strategy_history)
    np.save('populations.npy', population_history)
    plt.figure()
    for i in range(size_classes):
        plt.plot(obj.x, x_res[i]**2)
    plt.plot(obj.x, eco.water.res_counts)
    plt.show()

    plt.figure()
    for i in range(size_classes):
        plt.plot(obj.x, np.mean(strategy_history**2, axis = 0)[i])
    plt.plot(obj.x, eco.water.res_counts)
    plt.show()


elif simulate is False:
    strategy_history = np.load('strategies.npy')
    population_history = np.load('populations.npy')
    time_span = np.linspace(0, t_end, int(t_end/time_step))

    for i, t in enumerate(time_span):
        plt.plot(obj.x, strategy_history[i,2]**2, alpha = 0.01, color = tableau20[6])
    plt.show()