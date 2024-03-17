import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy import odr
import os
from scipy.ndimage import gaussian_filter1d

def trajectory_compute(transforms,name = 'trajectory.png'):
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0) 
    l = len(trajectory)
    time = np.arange(0, l, 1, dtype=int)
    x = trajectory[:,0]
    y = trajectory[:,1]
    theta = trajectory[:,2]
    scale = trajectory[:,3]

    # plot x and y translation and totation in 3 subplots
    plt.figure(figsize=(15,5))
    plt.subplot(141)
    plt.plot(time, x)
    plt.title("Movement X")
    plt.subplot(142)
    plt.plot(time, y)
    plt.title("Movement Y")
    plt.subplot(143)
    plt.plot(time, theta)
    plt.title("Rotation")
    plt.subplot(144)
    plt.plot(scale[:-100])
    plt.title("Scale")
    # save the plot
    plt.savefig("/media/aa/huge_ssd/stablization_project/data/"+name)
    # plt.show()
    return trajectory,time

def get_trajectories(transforms_path, frames_path, frames_clear_path,
                     transforms, n_frames, cap,
                     prev_gray, prev_pts, mask,out):
    frames = []
    frames_clear = []
    # if transforms_path not exists run the following code ,else load the transforms from the file
    if not os.path.exists(transforms_path):
        OF_points_old = []
        OF_points_new = []

        # prev_pts = np.float32([kp.pt for kp in p0]).reshape(-1, 1, 2)
        for i in tqdm(range(n_frames-2)):
        
            success, curr = cap.read() 
            if not success: 
                break 

            # Convert to grayscale
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 
            #   curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 

            # Sanity check
            assert prev_pts.shape == curr_pts.shape 
            #   assert prev_pts.shape == curr_pts.shape 
            
            # Filter only valid points
            idx = np.where(status==1)[0]
            good_old = prev_pts[idx]
            good_new = curr_pts[idx]
            
            #Find transformation matrix
            m = cv2. estimateAffine2D(good_old, good_new)
            # extimate Homography
            # m = cv2.findHomography(good_old, good_new)
            
            OF_points_old.append(good_old)
            OF_points_new.append(good_new)
            
            # Extract traslation
            dx = m[0][0][2]
            dy = m[0][1][2]

            # Extract rotation angle
            da = np.arctan2(m[0][1][0], m[0][0][0])
            
            # Extract scale from matrix
            ds = np.sqrt(m[0][0][0] ** 2 + m[0][1][0] ** 2)

            # Store transformation
            transforms[i] = [dx,dy,da,ds]

            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                
                mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), (255,0,0), 1)
                curr = cv2.circle(curr,(int(a),int(b)),5, (255,0,0),-1)

            img = cv2.add(curr,mask)
            frames.append(img)
            frames_clear.append(curr)

            # Now update the previous frame and previous points
            prev_pts = good_new.reshape(-1,1,2)
            prev_gray = curr_gray.copy()

            #print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))
            out.write(img)

        np.save(transforms_path, transforms)
        # save frames and frames clear
        np.save(frames_path, frames)
        np.save(frames_clear_path, frames_clear)

    else:
        transforms = np.load(transforms_path)
        frames = np.load(frames_path)
        frames_clear = np.load(frames_clear_path)

    # Release video
    cap.release()
    out.release()

    return transforms, frames, frames_clear

def plt_jitters(transforms):
    # Close windows
    cv2.destroyAllWindows()

    # plot x and y translation and totation in 3 subplots
    plt.figure(figsize=(15,5))
    # title
    plt.suptitle("Jitters of Translsation and Rotation and cale of the video")
    plt.subplot(141)
    plt.plot(transforms[:,0])
    plt.title("Translation X")
    plt.subplot(142)
    plt.plot(transforms[:,1])
    plt.title("Translation Y")
    plt.subplot(143)
    plt.plot(transforms[:,2])
    plt.title("Rotation")
    plt.subplot(144)
    plt.plot(transforms[:-10,3])
    plt.title("Scale")
    # save the plot
    plt.savefig("/media/aa/huge_ssd/stablization_project/data/jitters.png")
    # plt.show()

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

def median_filter(trajectory, radius=10):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:,i] = np.convolve(trajectory[:,i], np.ones(radius)/radius, mode='same')
    return smoothed_trajectory

def freq_smoothing(trajectory,freq_cut = 1000):
    # fft on x,y
    x_fft = np.fft.fft(trajectory[:,0])
    plt.plot(np.abs(x_fft))
    plt.title("FFT of x")
    # plt.show()
    # now filter out the high frequency noise
    x_fft[np.abs(x_fft) < freq_cut] = 0
    # plot the fft
    # inverse fft only real part
    x_filtered = np.fft.ifft(x_fft).real

    y_fft = np.fft.fft(trajectory[:,1])
    plt.plot(np.abs(y_fft))
    plt.title("FFT of y")
    # plt.show()
    # now filter out the high frequency noise
    y_fft[np.abs(y_fft) < freq_cut] = 0
    # inverse fft
    y_filtered = np.fft.ifft(y_fft).real


    theta_fft = np.fft.fft(trajectory[:,2])
    plt.plot(np.abs(theta_fft))
    plt.title("FFT of theta")
    # plt.show()
    # now filter out the high frequency noise
    theta_fft[np.abs(theta_fft) < freq_cut] = 0
    # inverse fft
    theta_filtered = trajectory[:,2]
    # theta_filtered = np.fft.ifft(theta_fft).real
    trajectory_filtered = np.column_stack((x_filtered, y_filtered, theta_filtered))

    return trajectory_filtered

def gaus_filter(trajectory,sigma_x = 10,sigma_y = 10,sigma_theta = 10,sigma_scale = 10):
    # apply gaussian filter to trajectory
    x_filtered = gaussian_filter1d(trajectory[:,0], sigma_x)
    y_filtered = gaussian_filter1d(trajectory[:,1], sigma_y)
    theta_filtered = gaussian_filter1d(trajectory[:,2], sigma_theta)
    scale_filtered = gaussian_filter1d(trajectory[:,3], sigma_scale)


    trajectory_filtered = np.column_stack((x_filtered, y_filtered, theta_filtered, scale_filtered))
    return trajectory_filtered

def plot_trans_and_trans_smooth(transforms,transforms_smooth):
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
    # plt.show()

def plot_trajectories(trajectory,trajectory_smooth,time,pwd = os.getcwd()):
    # plot both trajectories on the same plot
    # plot x and y translation and totation in 3 subplots
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.plot(time,trajectory[:,0],color='b',label="Before Smoothing")
    plt.plot(time,trajectory_smooth[:,0],color='r',label="After Smoothing")
    plt.title("Movement X")
    plt.legend()
    plt.subplot(132)
    plt.plot(time, trajectory[:,1],color='b',label="Before Smoothing")
    plt.plot(time, trajectory_smooth[:,1],color='r',label="After Smoothing")
    plt.title("Movement Y")
    plt.legend()
    plt.subplot(133)
    plt.plot(time, trajectory[:,2],color='b',label="Before Smoothing")
    plt.plot(time, trajectory_smooth[:,2],color='r',label="After Smoothing")
    plt.title("Rotation")

    # save the plot
    plt.savefig(os.path.join(pwd, "trajectory_smooth.png"))
    # plt.show()

def write_smoth_video(input_video_path, stabilized_video_path,
                        final_stabilized_video_path, final_cutted_video_path, transforms_smooth,
                    frames_clear, n_frames, w, h, fps):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out1 = cv2.VideoWriter(stabilized_video_path, fourcc, fps, (2*w, h))
    out2 = cv2.VideoWriter(final_stabilized_video_path, fourcc, fps, (w, h))
    cutout = 100
    out3 = cv2.VideoWriter(final_cutted_video_path, fourcc, fps, (w-int(cutout*2), h-int(cutout*2)))

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
        frame_out = cv2.hconcat([frames_clear[i], frame_stabilized])
        #   frame_out = cv2.hconcat([frames[i], frame_stabilized])
        
        out1.write(frame_out)
        out2.write(frame_stabilized)
        out3.write(frame_stabilized[cutout:h-cutout, cutout:w-cutout])
        #cv2_imshow(frame_out)
        #   cv2.waitKey(10)
        

    # Release video
    cap.release()
    out1.release()
    out2.release()
    out3.release()
    # Close windows
    cv2.destroyAllWindows()

def load_video_show_first_frame(input_video_path,output_video_path,maxCorners):
    # load the video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: ", n_frames, " fps: ", fps)
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

    mask = np.zeros_like(prev)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-1, 4), np.float32) 
    # transforms = np.zeros((n_frames-1, 3), np.float32) 

    # Initialize the ORB detector
    orb = cv2.ORB_create()
    p0, _ = orb.detectAndCompute(prev_gray, None)

    prev_pts = cv2.goodFeaturesToTrack( prev_gray,
                                        maxCorners=maxCorners,
                                        qualityLevel=0.01,
                                        minDistance=30,
                                        blockSize=3)

    # print number of goodfeatures
    print("Number of good features: ", prev_pts.shape[0])
    # plot prev_pts on the first frame

    # for pts in np.float32([kp.pt for kp in p0]).reshape(-1, 1, 2):
    for pts in prev_pts:
        x, y = pts.ravel()
        x, y = int(x), int(y)
        cv2.circle(prev, (x, y), 5, (255, 0, 0), -1)
    plt.imshow(prev)
    plt.title("First frame with good points")
    plt.savefig(os.path.join(os.getcwd(), "first_frame_with_good_points.png"))
    # plt.show()
    return cap, fps, n_frames, w, h, out, prev, prev_gray, prev_pts, mask, transforms
   