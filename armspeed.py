import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

# Read the video
cap = cv2.VideoCapture('IMG_7784.mp4')
fps = 240  # Camera FPS

# Key Points for Right Arm (shoulder, elbow, wrist)
right_arm_indices = [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value]
previous_positions = {index: None for index in right_arm_indices}
speeds = {index: [] for index in right_arm_indices}  # To store speed data

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process each frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Extract arm keypoints and calculate speed
        for index in right_arm_indices:
            landmark = results.pose_landmarks.landmark[index]
            current_position = np.array([landmark.x, landmark.y])

            if previous_positions[index] is not None:
                displacement = np.linalg.norm(current_position - previous_positions[index])
                speed = displacement * fps  # pixels/frame * frames/second = pixels/second
                speeds[index].append(speed)

            previous_positions[index] = current_position

        # Draw the pose annotation on the frame.
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('MediaPipe Pose', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print average speeds for each key point
for index in right_arm_indices:
    avg_speed = np.mean(speeds[index]) if speeds[index] else 0
    print(f"Average speed of key point {index}: {avg_speed} pixels/second")
