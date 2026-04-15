import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Initialize MediaPipe
mp_hands = mp.solutions.hands

st.set_page_config(page_title="Neon Air Writer", layout="wide")
st.title("🎨 Neon Air Writer")

class VideoProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0

    def transform(self, frame):
        # This replaces the 'recv' method to bypass the 'av' library issues
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        if self.canvas is None or self.canvas.shape != img.shape:
            self.canvas = np.zeros_like(img)

        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for res in results.multi_hand_landmarks:
                itip = res.landmark[8]
                x, y = int(itip.x * img.shape[1]), int(itip.y * img.shape[0])
                
                if itip.y < res.landmark[6].y: # Index finger up
                    if self.prev_x != 0:
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), (0, 255, 255), 10)
                    self.prev_x, self.prev_y = x, y
                else:
                    self.prev_x, self.prev_y = 0, 0
        
        return cv2.addWeighted(img, 0.5, self.canvas, 0.5, 0)

webrtc_streamer(
    key="air-writer",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=VideoProcessor().transform, # Newer bypass method
    async_processing=True,
)
