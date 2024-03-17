import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy import odr
import os
from scipy.ndimage import gaussian_filter1d
from tools import *

SMOOTHING_RADIUS = 10
maxCorners = 200

input_video_path = "data/video_jitterd_00_15_sec.mkv"
output_video_path = "data/video_jitterd_00_15_sec_jitter.mp4"
stabilized_video_path = "data/video_jitterd_00_15_sec_stabilized_" + str(SMOOTHING_RADIUS) + ".mkv"
final_stabilized_video_path = "data/video_jitterd_00_15_sec_stabilized_final_" + str(SMOOTHING_RADIUS) + ".mkv"
final_cutted_video_path = "data/video_jitterd_00_15_sec_stabilized_final_cutted" + str(SMOOTHING_RADIUS) + ".mkv"

pwd = os.getcwd()
# save into data directory
frames_path = os.path.join(pwd, "data/frames.npy")
transforms_path = os.path.join(pwd, "data/transforms.npy")
frames_clear_path = os.path.join(pwd, "data/frames_clear.npy")
trajectry_path = os.path.join(pwd, "data/trajectory.npy")

cap, fps, n_frames, w, h, out, prev, prev_gray, prev_pts, mask, transforms = load_video_show_first_frame(input_video_path,output_video_path,maxCorners = maxCorners)


# calculate jitters
transforms, frames, frames_clear = get_trajectories(transforms_path, frames_path, frames_clear_path, transforms, n_frames, cap,prev_gray, prev_pts, mask,out)
# plot jitters
plt_jitters(transforms)


trajectory,time = trajectory_compute(transforms)
# save the trajectory
np.save(trajectry_path, trajectory)

# smoothed_trajectory = median_filter(trajectory, radius=100)
smoothed_trajectory = gaus_filter(trajectory,sigma_x = 10,sigma_y = 10,sigma_theta = 10,sigma_scale = 10)
# smoothed_trajectory = smooth(trajectory) 

# Calculate difference in smoothed_trajectory and trajectory
# smoothed_trajectory = freq_smoothing(trajectory,freq_cut = 300)
difference = smoothed_trajectory - trajectory
 
# Calculate newer transformation array
transforms_smooth = transforms + difference

plot_trans_and_trans_smooth(transforms,transforms_smooth)

trajectory_smooth = smoothed_trajectory
# trajectory_smooth,time = trajectory_compute(transforms_smooth,name = 'trajectory_smooth_'+str(SMOOTHING_RADIUS)+'.png')

plot_trajectories(trajectory,trajectory_smooth,time)

write_smoth_video(input_video_path, stabilized_video_path,
                final_stabilized_video_path, final_cutted_video_path, transforms_smooth,
                    frames_clear, n_frames, w, h, fps)

