import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

# Read the video
cap = cv2.VideoCapture('IMG_7784.mp4')

# Assuming the video resolution is known or the same as the first frame for simplicity
ret, first_frame = cap.read()
if not ret:
    print("Failed to get a frame from the video.")
    cap.release()
    exit()

frame_height, frame_width = first_frame.shape[:2]
blank_background = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# Key Points for Right Arm (shoulder, elbow, wrist)
right_arm_indices = [
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 
    mp_pose.PoseLandmark.RIGHT_ELBOW.value, 
    mp_pose.PoseLandmark.RIGHT_WRIST.value
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process each frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Clear previous frame's drawings
        blank_background.fill(0)

        for index in right_arm_indices:
            landmark = results.pose_landmarks.landmark[index]
            # Convert normalized position to pixel coordinates
            x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
            # Draw circle for each right arm point
            cv2.circle(blank_background, (x, y), 5, (0, 255, 0), -1)

    # Display the frame with right arm points on a blank background
    cv2.imshow('Right Arm Points', blank_background)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
