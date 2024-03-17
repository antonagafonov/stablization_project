import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy import odr
import os
from scipy.ndimage import gaussian_filter1d


# load trajectory
trajectory = np.load("/media/aa/huge_ssd/stablization_project/data/trajectory.npy")
trajectory_x = trajectory[:,0]

# make number of frames 10 times more
# trajectory_x = np.repeat(trajectory_x, 10)

# now smoth the trajectory with gaussian filter
trajectory_x_smoothed = gaussian_filter1d(trajectory_x, 10)

# plot both trajectories
plt.plot(trajectory_x, label="Original")
plt.plot(trajectory_x_smoothed, label="Smoothed")
plt.legend()
plt.show()