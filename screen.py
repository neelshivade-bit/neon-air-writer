import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- ROBUST MEDIAPIPE INITIALIZATION ---
# Accessing solutions directly to bypass cloud-specific AttributeErrors
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Neon Air Writer", layout="wide")

st.title("🎨 Neon Air Writer")
st.markdown("#### Instructions: Use your **Index Finger** to draw. Raise your **Middle Finger** next to it to stop drawing.")

# Sidebar Settings
with st.sidebar:
    st.header("Control Panel")
    line_color = st.color_picker("Neon Color", "#FF00FF")
    line_thickness = st.slider("Brush Thickness", 1, 20, 7)
    # Convert Hex to BGR
    bgr_color = tuple(int(line_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
    
    if st.button("🗑️ Clear Canvas"):
        st.session_state["clear_canvas"] = True

class VideoProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        if self.canvas is None or self.canvas.shape != img.shape:
            self.canvas = np.zeros_like(img)

        # Handle Clear Canvas request
        if st.session_state.get("clear_canvas", False):
            self.canvas = np.zeros_like(img)
            st.session_state["clear_canvas"] = False

        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Landmark 8: Index Finger Tip | Landmark 12: Middle Finger Tip
                idx_tip = hand_landmarks.landmark[8]
                mid_tip = hand_landmarks.landmark[12]

                x, y = int(idx_tip.x * w), int(idx_tip.y * h)

                # Drawing Logic: Draw if Index is UP but Middle is DOWN
                # (Comparing Y coordinates: lower value means higher on screen)
                if idx_tip.y < hand_landmarks.landmark[6].y and mid_tip.y > hand_landmarks.landmark[10].y:
                    if self.prev_x != 0 and self.prev_y != 0:
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), bgr_color, line_thickness)
                    self.prev_x, self.prev_y = x, y
                else:
                    self.prev_x, self.prev_y = 0, 0
        else:
            self.prev_x, self.prev_y = 0, 0

        # Create Neon Glow Effect
        glow = cv2.GaussianBlur(self.canvas, (13, 13), 0)
        img = cv2.addWeighted(img, 0.7, glow, 0.3, 0)
        img = cv2.add(img, self.canvas)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC Configuration
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="neon-writer",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
