max_i = self.one_actor_growth(i, x_input).full().flatten()
strat_temporary = np.copy(x_input)
strat_temporary[i] = max_i
max_gain_i = self.one_actor_growth(i, strat_temporary)
regret_i_theta = self.one_actor_growth(i, strat_temporary) - self.one_actor_growth(i, x_input, solve_mode=False)
