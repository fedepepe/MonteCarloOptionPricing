# -*- coding: utf-8 -*-

import numpy as np
import math

# Models the underling asset assuming geometetric Brownian motion
class StochasticProcess:

    def __init__(self, asset_price, drift, delta_t, asset_volatility):
        self.current_asset_price = asset_price
        self.asset_prices = []
        self.asset_prices.append(asset_price)
        self.drift = drift
        self.delta_t = delta_t
        self.asset_volatility = asset_volatility
        
    def time_step(self):
        # Brownian motion is ~N(0, delta_t), np.random.normal takes mean and standard deviation
        dW = np.random.normal(0, math.sqrt(self.delta_t))
        dS = self.current_asset_price*(self.drift*self.delta_t + self.asset_volatility*dW)
        new_asset_price = self.current_asset_price + dS
        self.asset_prices.append(new_asset_price)
        # The new current asset price is the one just simulated
        self.current_asset_price = new_asset_price

class VanillaOption:

    def __init__(self, strike, otype):
        self.strike = strike
        self.otype = otype
    
    def vanillaPayoff(self, asset_price):
        if self.otype == 'call':
            return max(asset_price[-1] - self.strike, 0)
        elif self.otype == 'put':
            return max(self.strike - asset_price[-1], 0)
        
    def payoff(self, asset_price):
        return self.vanillaPayoff(asset_price)

class EuroOption(VanillaOption):
    pass

class AsianOption(VanillaOption):
    def payoff(self, asset_prices):
        return self.vanillaPayoff(np.mean(asset_prices, keepdims=True)) 

class BarrierOption(VanillaOption):

    def __init__(self, strike, otype, barrier, style):
        self.strike = strike
        self.otype = otype
        self.barrier = barrier
        self.style = style
    
    def isKnockedInOut(self, asset_prices):
        if self.style == 'downout' or self.style == 'downin':
            return any(asset_prices < self.barrier)
        elif self.style == 'upout' or self.style == 'upin':
            return any(asset_prices > self.barrier)
        
    def payoff(self, asset_prices):
        if self.style == 'downout' or self.style == 'upout':
            return self.vanillaPayoff(asset_prices) if not self.isKnockedInOut(asset_prices) else 0
        elif self.style == 'downin' or self.style == 'upin':
            return self.vanillaPayoff(asset_prices) if self.isKnockedInOut(asset_prices) else 0

class BinaryOption(VanillaOption):
    def payoff(self, asset_prices):
        return 1 if self.vanillaPayoff(asset_prices) > 0 else 0
            
class MonteCarloPriceSimulator:

    def __init__(self, Option, n_runs, initial_asset_price, drift, delta_t, asset_volatility, time_to_expiration, risk_free_rate):
        # List of stochastic processes modeling the underlying asset
        stochastic_processes = []
        # Allocate list entries for the n_runs stochastic processes to be simulated
        for i in range(n_runs):
            stochastic_processes.append(StochasticProcess(initial_asset_price, drift, delta_t, asset_volatility)) # Note delta t is annualized

        # Simulate the n_runs stochastic processes
        for stochastic_process in stochastic_processes:
            tte = time_to_expiration
            while((tte-stochastic_process.delta_t) > 0):
                # Account for the passing of time
                tte = tte - stochastic_process.delta_t
                # Simulate a new sample of the stochastic process
                stochastic_process.time_step()

        payoffs = []
        # compute their corresponding final payoff
        for stochastic_process in stochastic_processes:
            payoff = Option.payoff(np.array(stochastic_process.asset_prices))
            payoffs.append(payoff)
        
        # return the discounted values
        self.price = np.average(payoffs)*math.exp(-time_to_expiration*risk_free_rate)
        

# Optional seed
np.random.seed(12345678)
print('Monte Carlo Price: ', MonteCarloPriceSimulator(BinaryOption(100,'put'), 10000, 100, 0, 1/365, .2435, 39/365, .0017).price)