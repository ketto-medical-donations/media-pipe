import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("🧠 Pose Detection (Browser Camera)")

# ✅ FIXED import (old API access)
from mediapipe.python.solutions import pose
from mediapipe.python.solutions import drawing_utils

mp_pose = pose
mp_draw = drawing_utils


class PoseDetector(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if result.pose_landmarks:
            mp_draw.draw_landmarks(
                img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        return img


webrtc_streamer(key="pose", video_transformer_factory=PoseDetector)
