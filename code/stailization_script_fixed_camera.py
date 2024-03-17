import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

SMOOTHING_RADIUS = 200
features_to_track = 200

input_video_path = "/media/aa/huge_ssd/stablization_project/data/video_jitterd_00_15_sec.mkv"
output_video_path = "/media/aa/huge_ssd/stablization_project/data/video_jitterd_00_15_sec_jitter.mp4"

stabilized_video_path = "/media/aa/huge_ssd/stablization_project/data/video_jitterd_00_15_sec_stabilized_" + str(SMOOTHING_RADIUS) + ".mkv"
final_stabilized_video_path = "/media/aa/huge_ssd/stablization_project/data/video_jitterd_00_15_sec_stabilized_final_" + str(SMOOTHING_RADIUS) + ".mkv"
final_cutted_video_path = "/media/aa/huge_ssd/stablization_project/data/video_jitterd_00_15_sec_stabilized_final_cutted_" + str(SMOOTHING_RADIUS) + ".mkv"

frames = []

# load the video
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('M','P','4','V')

out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

# Read first frame
_, prev = cap.read() 
# covert to RGB
prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)

# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
zero_frame_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(prev)

# Pre-define transformation-store array
transforms = np.zeros((n_frames-1, 3), np.float32) 

prev_pts = cv2.goodFeaturesToTrack( prev_gray,
									maxCorners=features_to_track,
									qualityLevel=0.01,
									minDistance=30,
									blockSize=3)

zero_pts = cv2.goodFeaturesToTrack( zero_frame_gray,
									maxCorners=features_to_track,
									qualityLevel=0.01,
									minDistance=30,
									blockSize=3)

# plot prev_pts on the first frame
for pts in prev_pts:
	x, y = pts.ravel()
	x, y = int(x), int(y)
	cv2.circle(prev, (x, y), 5, (0, 0, 255), -1)
plt.imshow(prev)
plt.title("First frame with prev_pts")
plt.show()

OF_points_old = []
OF_points_new = []

for i in tqdm(range(n_frames-2)):
  
  success, curr = cap.read() 
  if not success: 
	break 

  # Convert to grayscale
  curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 
  curr_pts, status, err = cv2.calcOpticalFlowPyrLK(zero_frame_gray, curr_gray, zero_pts, None) 

  # Sanity check
  assert zero_pts.shape == curr_pts.shape 
 
  # Filter only valid points
  idx = np.where(status==1)[0]
  good_old = zero_pts[idx]
  good_new = curr_pts[idx]
   
  #Find transformation matrix
  m = cv2. estimateAffine2D(good_old, good_new)

  OF_points_old.append(good_old)
  OF_points_new.append(good_new)
  
  # Extract transformations
  dx = m[0][0][2]
  dy = m[0][1][2]
  da = np.arctan2(m[0][1][0], m[0][0][0])
 
  # Store transformation
  transforms[i] = [dx,dy,da]

  for i,(new,old) in enumerate(zip(good_new,good_old)):
	a,b = new.ravel()
	c,d = old.ravel()
	
	mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), (0,0,255), 1)
	curr = cv2.circle(curr,(int(a),int(b)),5, (0,0,255),-1)


  img = cv2.add(curr,mask)
  frames.append(img)

  # Now update the previous frame and previous points
#   prev_pts = good_new.reshape(-1,1,2)
#   prev_gray = curr_gray.copy()


  #print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))
  out.write(img)
  pass
# Release video
cap.release()
out.release()
# Close windows
cv2.destroyAllWindows()


# plot x and y translation and totation in 3 subplots
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.plot(transforms[:,0])
plt.title("Translation X")
plt.subplot(132)
plt.plot(transforms[:,1])
plt.title("Translation Y")
plt.subplot(133)
plt.plot(transforms[:,2])
plt.title("Rotation")
# save the plot
plt.savefig("/media/aa/huge_ssd/stablization_project/data/jitters.png")
plt.show()

def trajectory_compute(transforms,name = 'trajectory.png'):
	# Compute trajectory using cumulative sum of transformations
	trajectory = np.cumsum(transforms, axis=0) 
	l = len(trajectory)
	time = np.arange(0, l, 1, dtype=int)
	x = trajectory[:,0]
	y = trajectory[:,1]
	theta = trajectory[:,2]

	# plot x and y translation and totation in 3 subplots
	plt.figure(figsize=(15,5))
	plt.subplot(131)
	plt.plot(time, x)
	plt.title("Movement X")
	plt.subplot(132)
	plt.plot(time, y)
	plt.title("Movement Y")
	plt.subplot(133)
	plt.plot(time, theta)
	plt.title("Rotation")
	# save the plot
	plt.savefig("/media/aa/huge_ssd/stablization_project/data/"+name)
	plt.show()
	return trajectory

trajectory = trajectory_compute(transforms)



def movingAverage(curve, radius): 
  window_size = 2 * radius + 1
  # Define the filter 
  f = np.ones(window_size)/window_size 
  # Add padding to the boundaries 
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
  # Apply convolution 
  curve_smoothed = np.convolve(curve_pad, f, mode='same') 
  # Remove padding 
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed 

def smooth(trajectory): 
  smoothed_trajectory = np.copy(trajectory) 
  # Filter the x, y and angle curves
  for i in range(3):
	smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)

  return smoothed_trajectory

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame


# Create variable to store smoothed trajectory
smoothed_trajectory = trajectory
# smoothed_trajectory = smooth(trajectory) 

# Calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory
 
# Calculate newer transformation array
transforms_smooth = transforms
# transforms_smooth = transforms + difference

# plot x and y translation and totation in 3 subplots before smooth in green and smoth in red 
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.plot(transforms[:,0], 'g', label="before smoothing")
plt.plot(transforms_smooth[:,0], 'r', label="after smoothing")
plt.title("Translation X")
plt.legend()
plt.subplot(132)
plt.plot(transforms[:,1], 'g', label="before smoothing")
plt.plot(transforms_smooth[:,1], 'r', label="after smoothing")
plt.title("Translation Y")
plt.legend()
plt.subplot(133)
plt.plot(transforms[:,2], 'g', label="before smoothing")
plt.plot(transforms_smooth[:,2], 'r', label="after smoothing")
plt.title("Rotation")
plt.legend()
# save the plot
plt.savefig("/media/aa/huge_ssd/stablization_project/data/smoothed.png")
plt.show()

trajectory_smooth = trajectory_compute(transforms_smooth,name = 'trajectory_smooth_'+str(SMOOTHING_RADIUS)+'.png')
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter(stabilized_video_path, fourcc, fps, (2*w, h))
out2 = cv2.VideoWriter(final_stabilized_video_path, fourcc, fps, (w, h))
out3 = cv2.VideoWriter(final_cutted_video_path, fourcc, fps, (w-200, h-200))

# Write n_frames-1 transformed frames
for i in tqdm(range(n_frames-2)):
  # Read next frame
  success, frame = cap.read() 
  if not success:
	break

  # Extract transformations from the new transformation array
  dx = transforms_smooth[i,0]
  dy = transforms_smooth[i,1]
  da = transforms_smooth[i,2]

  # Reconstruct transformation matrix accordingly to new values
  m = np.zeros((2,3), np.float32)
  m[0,0] = np.cos(da)
  m[0,1] = -np.sin(da)
  m[1,0] = np.sin(da)
  m[1,1] = np.cos(da)
  m[0,2] = dx
  m[1,2] = dy

  # Apply affine wrapping to the given frame
  frame_stabilized = cv2.warpAffine(frame, m, (w,h))

  # Fix border artifacts
  frame_stabilized = fixBorder(frame_stabilized) 

  # Write the frame to the file
  frame_out = cv2.hconcat([frames[i], frame_stabilized])

  out1.write(frame_out)
  out2.write(frame_stabilized)
  out3.write(frame_stabilized[100:h-100, 100:w-100])
  #cv2_imshow(frame_out)
#   cv2.waitKey(10)
  

# Release video
cap.release()
out1.release()
out2.release()
out3.release()
# Close windows
cv2.destroyAllWindows()