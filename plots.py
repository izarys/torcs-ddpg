import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt

file = open('score_history.csv', 'rb')
scores = loadtxt(file, delimiter = " ")

plt.plot(scores)
plt.show()

file = open('steps_history.csv', 'rb')
steps = loadtxt(file, delimiter = " ")

plt.plot(steps)
plt.show()

