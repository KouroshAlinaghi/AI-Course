import numpy as np
import math
import matplotlib.pyplot as plt 
from scipy.stats import poisson, binom
from numpy import asarray
import tensorflow as tf

green = '#57cc99'
blue = '#22577a'
red = '#e56b6f'

line_thickness = 3

n = 250
p = 0.008
mu = n * p

width = 0.2

x_values = np.arange(0, 10)

binomial_pmf = binom.pmf(x_values, n, p)
poisson_pmf = poisson.pmf(x_values, mu)

fig, ax = plt.subplots()

ax.bar(x_values, binomial_pmf, width, label='B ~ Bin({}, {})'.format(n, p), color=green, alpha=0.7, edgecolor='black')
ax.bar(x_values - width, poisson_pmf, width, label='P ~ Pois({})'.format(mu), color=blue, alpha=0.7, edgecolor='black')

ax.set_xticks(x_values)
ax.set_xlabel('x')
ax.set_ylabel('P{X = x}')
ax.set_title("Binomail preidiction with Normal and Poisson distributions")
ax.legend()
plt.grid(True)
plt.show()

def relative_entropy(P, Q):
    res = 0
    for x in range(0, len(P)):
        res += P[x] * math.log(P[x] / Q[x])
    return res

print("D(B || P) =", relative_entropy(binomial_pmf, poisson_pmf))
print("D(P || B) =", relative_entropy(poisson_pmf, binomial_pmf))

print("D(B || P) =", np.sum(binomial_pmf * np.log(binomial_pmf / poisson_pmf)))
print("D(P || B) =", np.sum(poisson_pmf * np.log(poisson_pmf / binomial_pmf)))

n = 10
p = 0.5
mu = 4

binomial_pmf = binom.pmf(x_values, n, p)
poisson_pmf = poisson.pmf(x_values, mu)

fig, ax = plt.subplots()

ax.bar(x_values, binomial_pmf, width, label='B ~ Bin({}, {})'.format(n, p), color=green, alpha=0.7, edgecolor='black')
ax.bar(x_values - width, poisson_pmf, width, label='P ~ Pois({})'.format(mu), color=blue, alpha=0.7, edgecolor='black')

ax.set_xticks(x_values)
ax.set_xlabel('x')
ax.set_ylabel('P{X = x}')
ax.set_title("Binomail preidiction with Normal and Poisson distributions")
ax.legend()
plt.grid(True)
plt.show()

print("D(B || P) =", relative_entropy(binomial_pmf, poisson_pmf))
print("D(P || B) =", relative_entropy(poisson_pmf, binomial_pmf))

def cross_entropy(p, q):
    res = 0
    for x in range(len(p)):
        res -= p[x] * math.log(q[x])
    return res

actual_values = [0, 1, 0]
predicted_values = [0.05, 0.85, 0.10]

loss = tf.keras.losses.CategoricalCrossentropy()
loss = loss(actual_values, predicted_values)
print(loss.numpy)

print("Cross-entropy loss:", cross_entropy(actual_values, predicted_values))

actual_values = [0, 1, 0]
predicted_values = [0.85, 0.05, 0.10]

print("Cross-entropy loss:", cross_entropy(actual_values, predicted_values))

def entropy(p):
    res = 0
    for x in range(len(p)):
        res -= p[x] * math.log(p[x])
    return res

p = np.random.randint(1, 11, size=(20))
q = np.random.randint(1, 11, size=(20))

p = p / np.sum(p)
q = q / np.sum(q)

print("H(P)", entropy(p))
print("H(P, Q)", cross_entropy(p, q))
print("D_KL(P || Q)", relative_entropy(p, q))

print("H(P) + D_KL(P || Q) = ", entropy(p) + relative_entropy(p, q))
