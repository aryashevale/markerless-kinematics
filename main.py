import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from scipy.signal import savgol_filter

# Ensure the root directory is in the path to allow imports from the 'src' folder
sys.path.append(os.path.abspath('.'))
from src.biomechanics import calculate_angle_3d, get_proportional_torque

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def run_inference(video_path):
    print(f"--- Starting Analysis for: {video_path} ---")
    
    if not os.path.exists(video_path):
        print(f"Error: The file {video_path} does not exist. Check the file path.")
        return

    cap = cv2.VideoCapture(video_path)
    raw_angles = []
    
    # Phase 1: Kinematic Extraction
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract spatial coordinates for the Right Leg
            h = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z]) # Hip
            k = np.array([landmarks[26].x, landmarks[26].y, landmarks[26].z]) # Knee
            a = np.array([landmarks[28].x, landmarks[28].y, landmarks[28].z]) # Ankle
            
            raw_angles.append(calculate_angle_3d(h, k, a))
            
    cap.release()
    
    if not raw_angles:
        print("Error: No pose landmarks were detected in the video.")
        return

    # Phase 2: Signal Processing
    # Ensure window length is valid for the number of extracted frames
    window_len = min(11, len(raw_angles))
    if window_len % 2 == 0: 
        window_len -= 1
        
    if window_len > 3:
        smoothed_angles = savgol_filter(raw_angles, window_length=window_len, polyorder=3)
    else:
        smoothed_angles = np.array(raw_angles)
    
    # Phase 3: Diagnosis via Robust Outlier Rejection
    # Utilizing the 95th and 5th percentiles to filter visual tracking anomalies
    upper_bound = np.percentile(smoothed_angles, 95)
    lower_bound = np.percentile(smoothed_angles, 5)
    rom = upper_bound - lower_bound
    
    print("Analysis Complete.")
    print(f"Robust Range of Motion: {rom:.2f} degrees")
    
    # Control System Output
    if rom < 65: 
        print("DIAGNOSIS: Abnormal (Stiff Knee) detected.")
        # Calculate max correction torque needed based on a healthy target of ~75 degrees flexion
        torque = get_proportional_torque(target_angle=75, actual_angle=lower_bound)
        print(f"ACTION: Assistive Torque required: {torque:.2f} Nm")
    else:
        print("DIAGNOSIS: Normal Gait detected.")
        print("ACTION: No assistance required.")
    
    print("-" * 45)

if __name__ == "__main__":
    # Test execution
    test_video = 'data/raw/stiff_knee_01.mp4'
    run_inference(test_video)