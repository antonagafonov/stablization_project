import numpy as np
import cv2
import matplotlib.pyplot as plt

original_video_path = "/media/aa/huge_ssd/stablization_project/data/video_jitterd_00.mkv"
output_video_path = "/media/aa/huge_ssd/stablization_project/data/video_jitterd_00_15_sec.mkv"

cap = cv2.VideoCapture(original_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_time = total_frames / fps
print("Total frames: ", total_frames)
print("Total time: ", total_time, " seconds")

# now we need to chunk the video from 60 sec to 75 sec and save it to a new video
start_time = 60
end_time = 75
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# save the chunked video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (1920, 1080))
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
for i in range(start_frame, end_frame):
    ret, frame = cap.read()
    if ret:
        out.write(frame)
    else:
        print("Error in reading frame")
        break
