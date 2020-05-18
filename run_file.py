
depth = 20 #Previously 5 has worked well.
layers = 10 #5 works well.
log_size_range = 5 #8 NOT worked well.
size_classes = 4

obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
mass_vector = np.array([1, 300, 300**2, 10*300**2]) #np.exp(np.linspace(0,log_size_range,size_classes)) #
print(mass_vector)
res_start = 4*1/(1+np.exp(np.linspace(0, depth, layers))) #Previously used the constant 1. Now\re fully loaded!
water_start = water_column(obj, res_start, layers = layers)

params = ecosystem_parameters(mass_vector, obj)

eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start)

#print(params.layered_foraging.shape)
x_res = optm.least_squares(eco.loss_function, x0=eco.strategy_matrix.flatten(), bounds = (0, np.inf), loss = 'cauchy').x
#print(x_res)
for k in range(size_classes):
    print(np.dot(np.repeat(1, layers),np.dot(obj.M,x_res[layers*k:layers*(k+1)])))


new_size = 40 #Off-by-one
new_spectral = spectral_method(depth, new_size - 1)
res_start2 = 4*1/(1+np.exp(np.linspace(0, depth, new_size)))
water_start2 = water_column(new_spectral, res_start2, layers = new_size)
params2 = ecosystem_parameters(mass_vector, new_spectral)

seed_vec2 = interpolater(x_res, layers, new_size, size_classes, obj, new_spectral)
print(seed_vec2)
eco2 = ecosystem_optimization(mass_vector, new_size, params2, new_spectral, water_start2)
print(eco2.layers, "Eco2shape")

eco2.strategy_setter(seed_vec2)
print(eco2.strategy_matrix.flatten().shape)
print(np.dot(np.repeat(1, new_size),np.dot(new_spectral.M,np.repeat(1, new_size))), "What, why no 5")

x_res2 = optm.least_squares(eco2.loss_function, x0=eco2.strategy_matrix.flatten(), tr_solver='lsmr', bounds = (0, np.inf)).x

for k in range(size_classes):
    print(np.dot(np.repeat(1, new_size),np.dot(new_spectral.M, x_res2[new_size*k:new_size*(k+1)])))

plt.figure()
plt.plot(obj.x, x_res[0:layers])
plt.show()

plt.figure()

for i in range(size_classes):
    plt.plot(new_spectral.x, x_res2[i*new_size:(i+1)*new_size])

plt.show()

#new_strat = new_strat/np.dot(np.repeat(1, 10),np.dot(new_spectral.M, new_strat))




eco.strategy_setter(x_res)
t_end = 2
time_step = 0.001

simulate = False
if simulate is True:
    time_span = np.linspace(0, t_end, int(t_end/time_step))

    for t in time_span:
        eco.population_setter(eco.total_growth()*time_step + eco.populations)
        eco.consume_resources(time_step)
        eco.water.update_resources()
        x_res = optm.newton_krylov(eco.loss_function, x0=np.reshape(eco.strategy_matrix, layers*size_classes)).x #bounds=([0, np.inf]))
        eco.strategy_setter(x_res)


    plt.figure()
    plt.plot(obj.x, x_res[0:layers])
    plt.show()