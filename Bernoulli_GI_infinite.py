# This script is for the MPhil Project - Yuxin Chang

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import math
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.stats import norm, beta
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
            value_vec_prev[i] = max(self.v[-1][i], 0) / (1 - self.gamma)
        
        for t in reversed(range(0, self.N)):  # from N - 1 to 0
            t_n = t + self.n
            
            for j in reversed(range(number_sigma - (self.N - t))):
                mu = sigma_range[j] / t_n
                risk_reward = mu + self.gamma * (mu * value_vec_prev[j + 1] + (1 - mu) * value_vec_prev[j])
                self.v[t][j] = risk_reward - self.Lambda
                
                if risk_reward > self.Lambda: 
                    value_vec_curr[j] = risk_reward - self.Lambda  # discount factor already computed for every iter
                else:
                    value_vec_curr[j] = 0
                
            value_vec_prev = value_vec_curr.copy()
            
        return self.v




class MultiArmedBandits:
    def __init__(self, k = 10, epsilon = 0, initial = 0, true_expected_reward = 0, step_size = None, ucb = None, 
                 gradient = None, baseline = None, thompson = None, bayes_ucb = None, gittins = None, 
                 num_steps = 1000, num_runs = 2000):
        self.k = k
        self.epsilon = epsilon
        self.initial = initial
        self.initial_mu = true_expected_reward
        self.step_size = step_size
        self.ucb_c = ucb # degree of exploration
        self.gradient = gradient # step_size for gradient bandit algorithm
        self.baseline = baseline
        self.thompson = thompson
        self.bayes_ucb_c = bayes_ucb 
        self.gi_values = gittins
        self.num_steps = num_steps
        self.num_runs = num_runs
        
        
    def new_bandit(self):
        self.q = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k).astype(int)
        self.mu = np.random.random(size = self.k)  # between 0 and 1
        self.action_optimal = np.argmax(self.mu)
        self.reward_episode = np.zeros(self.num_steps)
        self.action_episode = np.zeros(self.num_steps)
        if self.gradient:
            self.h = np.zeros(self.k)
            self.prev_rewards_bar = 0
            self.rewards_bar = 0
        if self.thompson or self.bayes_ucb_c or self.gi_values != None:
            self.alpha = np.ones(self.k).astype(int)
            self.beta = np.ones(self.k).astype(int)
        
            
        
    # default: 1000 steps make up one episode/run    
    def episode(self):
        for step in range(self.num_steps):
            
            # choose action
            if np.random.random() < self.epsilon: # choose randomly
                action = np.random.randint(1,self.k)
            elif self.ucb_c:
                if min(self.action_count) == 0:
                    action = np.argmin(self.action_count)
                else:
                    ucb_action = self.q + self.ucb_c * np.sqrt(np.divide(np.log(step + 1), self.action_count))
                    action = np.argmax(ucb_action)
            elif self.gradient: # gradient bandit: only consider numerical preference
                h_exp = np.exp(self.h)
                pi = h_exp/np.sum(h_exp)
                action = np.random.choice(np.arange(self.k), p = pi)
            elif self.thompson:
                action = np.argmax(np.random.beta(self.alpha, self.beta))
            elif self.bayes_ucb_c:
                if min(self.action_count) == 0:
                    action = np.argmin(self.action_count)
                else:
                    quantile = beta.ppf(1 - 1 / ((step + 1) * pow(np.log(self.num_steps), self.bayes_ucb_c)), self.alpha, self.beta)
                    action = np.random.choice(np.where(quantile == max(quantile))[0])
            elif self.gi_values != None:
                index = np.zeros(self.k)
                for i in range(self.k):
                    index[i] = self.alpha[i] / (self.alpha[i] + self.beta[i]) + self.gi_values[self.beta[i] - 1][self.alpha[i] - 1]
                action = np.argmax(index)
            else:
                action = np.argmax(self.q)  
            
            
            # update action
            self.action_count[action] += 1
            if action == self.action_optimal:
                self.action_episode[step] = 1

                
            # receive Bernoulli rewards
            if np.random.random() <= self.mu[action]:
                reward = 1
            else:
                reward = 0
            self.reward_episode[step] = reward
            
            
            # update parameters
            if self.step_size:
                self.q[action] += self.step_size * (reward - self.q[action])  # constant step-size
            elif self.gradient:
                h_action = np.zeros(self.k)
                h_action[action] = 1
                if self.baseline: # if no baseline, q is constantly 0
                    self.rewards_bar += (reward - self.rewards_bar) / (step + 1) # average of all rewards
                self.h += self.gradient * (reward - self.rewards_bar) * (h_action - pi) # update h, eq. 2.12 stochastic gradient ascent              
            elif self.thompson or self.bayes_ucb_c or self.gi_values != None:
                if reward == 1:
                    self.alpha[action] += 1
                else:
                    self.beta[action] += 1
            else: 
                self.q[action] += (reward - self.q[action]) / self.action_count[action]  # sample average
        return self.reward_episode, self.action_episode
    
    # default: average of 2000 runs
    def simulate(self):
        average_reward = np.zeros((self.num_runs, self.num_steps))
        action_percentage = np.zeros((self.num_runs, self.num_steps))
        for run in tqdm(range(self.num_runs)):
            self.new_bandit()
            average_reward[run], action_percentage[run] = self.episode()
        return [np.mean(average_reward, axis = 0), np.mean(action_percentage, axis = 0), np.mean(average_reward, axis = 1)]




# horizon n = 32
horizon = 32
gi_32_bernoulli = [[0] * horizon for _ in range(horizon)]
for alpha in tqdm(range(1, horizon + 1)):
    for beta in range(1, horizon - alpha + 2):
        gi_32_bernoulli[beta - 1][alpha - 1] = calibrate(Sigma = alpha, n = alpha + beta, N = 32, gamma = 0.96875)
        


gittins_steps32_runs1000000 = MultiArmedBandits(gittins = gi_32_bernoulli, num_steps = 32, num_runs = 1000000).simulate()
thompson_steps32_runs1000000 = MultiArmedBandits(thompson = True, num_steps = 32, num_runs = 1000000).simulate()
print("Gittins Index mean: " + str(np.mean(gittins_steps32_runs1000000[2])))
print("Gittins Index std: " + str(np.std(gittins_steps32_runs1000000[2])/np.sqrt(1000000)))
print("Thompson Sampling mean: " + str(np.mean(thompson_steps32_runs1000000[2])))
print("Thompson Sampling std: " + str(np.std(thompson_steps32_runs1000000[2])/np.sqrt(1000000)))



