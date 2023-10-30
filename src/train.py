import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import pickle
import microsom as msom


# load input_data from file
input_data = np.loadtxt('./data/input_data.txt')

n, d = len(input_data), len(input_data[0])

x = 100
y = 100
epochs = 1000
learning_rate = 0.1

kohosom4 = msom.SOM(x=x, y=y, num_dim=d, learning_rate=learning_rate, random_seed=42)

start_time = time.time()
kohosom4.train(input_data, epochs, verbose=False)

end_time = time.time()
duration = end_time - start_time
sigma = kohosom4.sigma
learning_rate = kohosom4.learning_rate
x, y = kohosom4.map_size

print("Training duration:", duration, "seconds")
print(x,y,d,epochs,sigma,learning_rate)

weights = kohosom4.weights
plt.imshow(weights)

kohosom4.pickle_model('./models/kohosom4.pkl')