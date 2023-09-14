import numpy as np
from scipy.io import loadmat # loading data from matlab
import matplotlib.pyplot as plt

SShape = np.load('IROS_dataset/SShape.npy')
print(SShape.shape)


# plt.plot(SShape[:,0], SShape[:,1])
# plt.show()


letter = 'B'  # choose a letter in the alphabet
datapath = './data/2Dletters/'
data = loadmat(datapath + '%s.mat' % letter)
demos = [d['pos'][0][0].T for d in data['demos'][0]]
demos = np.array(demos)

# 12, 200
print(demos.shape)