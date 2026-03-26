import streamlit as st
import cv2
import mediapipe as mp

st.title("🧠 Pose Detection (MediaPipe + Streamlit)")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Start webcam
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera not working")
        break

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    # Draw landmarks
    if result.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    # Show frame in Streamlit
    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()