import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("🧠 Pose Detection (Browser Camera)")

# NEW mediapipe import
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks import python

BaseOptions = python.BaseOptions

class PoseDetector(VideoTransformerBase):
    def __init__(self):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = PoseLandmarker.create_from_options(options)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return img  # (basic version for now)

webrtc_streamer(key="pose", video_transformer_factory=PoseDetector)
