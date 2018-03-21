

import numpy as np
from matplotlib import pyplot as plt

state = np.load('state.npy')
next_state = np.load('next_state.npy')

plt.figure(1)

for i in range(8):

    plt.subplot(241+i)
    if i<4:
        plt.imshow(state[:,:,i],cmap=None)
    else:
        plt.imshow(next_state[:,:,i-4],cmap=None)


plt.show()
