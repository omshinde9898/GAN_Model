import torch
print(torch.cuda.get_device_name())
import numpy as np
from generator import generator
import tensorflow as tf
import matplotlib.pyplot as plt

generator.load_weights('generator_150.h5')

imgs = generator.predict(tf.random.normal((16, 128,)))


fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(10,10))

for r in range(4): 
    for c in range(4): 
        ax[r][c].imshow(np.squeeze(imgs[(r+1)*(c+1)-1]))