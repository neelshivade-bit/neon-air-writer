import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Neon Air Writer", layout="wide")

st.title("🎨 Neon Air Writer")
st.write("Instruction: Raise **Index Finger** to draw. Raise **Index + Middle** to stop/move.")

class VideoProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # Initialize canvas if it doesn't exist
        if self.canvas is None or self.canvas.shape != img.shape:
            self.canvas = np.zeros_like(img)

        # Process landmarks
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_img)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 8 is Index Tip, 12 is Middle Tip, 6 is Index Knuckle, 10 is Middle Knuckle
                itip = hand_landmarks.landmark[8]
                mtip = hand_landmarks.landmark[12]
                iknuckle = hand_landmarks.landmark[6]
                mknuckle = hand_landmarks.landmark[10]
                
                x, y = int(itip.x * w), int(itip.y * h)

                # DRAWING MODE: Index up, Middle down
                if itip.y < iknuckle.y and mtip.y > mknuckle.y:
                    if self.prev_x != 0 and self.prev_y != 0:
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), (0, 255, 255), 10)
                    self.prev_x, self.prev_y = x, y
                # HOVER/STOP MODE: Both fingers up or down
                else:
                    self.prev_x, self.prev_y = 0, 0
        else:
            self.prev_x, self.prev_y = 0, 0

        # Create Neon Effect
        glow = cv2.GaussianBlur(self.canvas, (13, 13), 0)
        img = cv2.addWeighted(img, 0.6, glow, 0.4, 0)
        img = cv2.add(img, self.canvas)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamer Configuration
webrtc_streamer(
    key="neon-writer-render",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
