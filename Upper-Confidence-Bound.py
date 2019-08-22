# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:52:16 2019

@author: Walmond
"""

# Importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
#Because the clicks of each ads was zero at first
# N is total round
# math.log (n = 1) because the library begin their index number with 1
import math
N = 10000
d = 10
ads_selected = []
numbers_of_selection = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selection[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selection[ad] = numbers_of_selection[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] =sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
# Visualizing the result
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of Times each time ad is selected')
plt.show()