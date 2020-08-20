import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pickle
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm


class GittinsIndex:
    def __init__(self, Sigma, n = 1, N = 100, gamma = 0.9, Lambda = 0):
        self.Sigma = Sigma
        self.n = n
        self.N = N
        self.gamma = gamma
        self.Lambda = Lambda
        
        # row: Sigma,..., Sigma + N; col: n,..., n + N
        self.v = [[0] * (self.N + 1) for _ in range(self.N + 1)]
        
    def valueBMAB(self):
        sigma_range = np.linspace(self.Sigma, self.Sigma + self.N, num = self.N + 1)
        number_sigma = len(sigma_range)
        
        value_vec_prev = np.zeros(number_sigma)
        value_vec_curr = np.zeros(number_sigma)
        
        for i in range(number_sigma):
            self.v[-1][i] = sigma_range[i] / (self.n + self.N) - self.Lambda
            value_vec_prev[i] = max(self.v[-1][i], 0)

        for t in reversed(range(0, self.N)):  # from N - 1 to 0
            t_n = t + self.n
            
            for j in reversed(range(number_sigma - (self.N - t))):
                mu = sigma_range[j] / t_n
                risk_reward = mu + (mu * value_vec_prev[j + 1] + (1 - mu) * value_vec_prev[j])
                self.v[t][j] = risk_reward - self.Lambda
                
                if risk_reward > self.Lambda: 
                    value_vec_curr[j] = risk_reward - self.Lambda  # discount factor already computed for every iter
                else:
                    value_vec_curr[j] = 0
                
            value_vec_prev = value_vec_curr.copy()
            
        return self.v



def calibrate(Sigma = 1, n = 2, N = 30, gamma = 1, hi = 1, lo = 0, tol = 1e-4): 
    while hi - lo > tol:
        lambda_hat = (hi + lo)/2
        if GittinsIndex(Sigma = Sigma, n = n, N = N, gamma = gamma, Lambda = lambda_hat).valueBMAB()[0][0] > 0:
            lo = lambda_hat
        else:
            hi = lambda_hat
    return (lo + hi) / 2


    

bernoulli_gi_512 = []
for N in tqdm(range(1, 512 + 1)):  # remaining horizon h = N - n
    results = [[0] * (N + 1) for _ in range(N + 1)]
    for alpha in tqdm(range(1, N + 1)):
        for beta in range(1, N - alpha + 2):
            results[beta - 1][alpha - 1] = calibrate(Sigma = alpha, n = alpha + beta, N = N, gamma = 1)
        
    bernoulli_gi_512.append(results)


with open('bernoulli_gi_512_finite.pickle', 'wb') as f:
    pickle.dump(bernoulli_gi_512, f)


# with open('bernoulli_gi_512_finite.pickle', 'rb') as f:
#     bernoulli_gi_512 = pickle.load(f)



gittins_steps32_runs1000000 = MultiArmedBandits(gittins = bernoulli_gi_512[:32], num_steps = 32, num_runs = 1000000).simulate()
thompson_steps32_runs1000000 = MultiArmedBandits(thompson = True, num_steps = 32, num_runs = 1000000).simulate()
print("Gittins Index mean: " + str(np.mean(gittins_steps32_runs1000000[2])))
print("Gittins Index std: " + str(np.std(gittins_steps32_runs1000000[2])/np.sqrt(1000000)))
print("Thompson Sampling mean: " + str(np.mean(thompson_steps32_runs1000000[2])))
print("Thompson Sampling std: " + str(np.std(thompson_steps32_runs1000000[2])/np.sqrt(1000000)))


