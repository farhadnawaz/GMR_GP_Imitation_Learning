import numpy as np
from fastdtw import fastdtw

test_traj = np.load('results/I/I10.npy').T
demo_traj = np.load('results/I/demo.npy')

print(test_traj.shape, demo_traj.shape)

distance, path = fastdtw(demo_traj, test_traj)
print(distance)
