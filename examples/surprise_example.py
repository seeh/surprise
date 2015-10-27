#!/usr/bin/env python
"""
Uses surprise on a toy example.
"""

from surprise import surprise
from numpy.random.mtrand import randn, randint
from matplotlib.mlab import entropy

sc = surprise()

# Create two samples from a standard normal distribution
# Reshape into (# of samples, # of dimensions) array
n = 100
sample1 = randn(n).reshape(-1,1)
sample2 = randn(n).reshape(-1,1)

# Mode is 'replace', i.e. we are assuming that the two distributions
# are separately analysed posteriors.
mode = 'replace'

# Calculate entropy numbers with surprise
rel_ent, exp_rel_ent, S, sD, p = sc(sample1, sample2, mode = mode)

print('Entropy estimates for two standard normal distributions.')
print('Relative entropy D: %f'%rel_ent)
print('Expected relative entropy <D>: %f'%exp_rel_ent)
print('Surprise S: %f'%S)
print('Expected fluctuations of relative entropy sigma(D): %f'%sD)
print('p-value of Surprise: %f'%p)

# Draw random weights for the samples.
mini = 1
maxi = 10
weights1 = randint(mini, maxi, n)
weights2 = randint(mini, maxi, n)

# Calculate entropy numbers with surprise using the random weights
rel_ent, exp_rel_ent, S, sD, p = sc(sample1, sample2, mode = mode,
                                    weights1 = weights1, weights2 = weights2)

print('Entropy estimates for two standard normal distributions.')
print('(with random weights)')
print('Relative entropy D: %f'%rel_ent)
print('Expected relative entropy <D>: %f'%exp_rel_ent)
print('Surprise S: %f'%S)
print('Expected fluctuations of relative entropy sigma(D): %f'%sD)
print('p-value of Surprise: %f'%p)
