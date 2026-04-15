import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Page Config
st.set_page_config(page_title="Ultra-Smooth Neon Painter", layout="wide")

# HTTPS Warning
st.warning("⚠️ Please open this app in HTTPS (required for camera access). Use Chrome browser.")

# Session State
if "clear_canvas" not in st.session_state:
    st.session_state.clear_canvas = False

if "brush_color" not in st.session_state:
    st.session_state.brush_color = (0, 255, 255)

# Sidebar UI
with st.sidebar:
    st.title("🎨 Settings")
    margin = st.slider("Edge Sensitivity", 0, 200, 80)
    smoothness = st.slider("Smoothing", 1, 15, 3)

    color_option = st.selectbox(
        "Brush Color",
        ["Neon Rainbow", "Green", "Red", "Blue", "White"]
    )

    if st.button("🗑️ Clear Canvas"):
        st.session_state.clear_canvas = True

class PrecisionProcessor:
    def __init__(self):
        self.canvas = None
        self.plocX, self.plocY = 0, 0
        self.hue = 0
        self.lock = threading.Lock()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def get_color(self):
        if color_option == "Neon Rainbow":
            self.hue = (self.hue + 3) % 180
            color = cv2.cvtColor(
                np.uint8([[[self.hue, 255, 255]]]),
                cv2.COLOR_HSV2BGR
            )[0][0]
            return int(color[0]), int(color[1]), int(color[2])

        elif color_option == "Green":
            return (0, 255, 0)
        elif color_option == "Red":
            return (0, 0, 255)
        elif color_option == "Blue":
            return (255, 0, 0)
        elif color_option == "White":
            return (255, 255, 255)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # Thread-safe canvas reset
        with self.lock:
            if self.canvas is None or st.session_state.get("clear_canvas", False):
                self.canvas = np.zeros_like(img)
                st.session_state.clear_canvas = False

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark

            itip = lm[8]
            iknuckle = lm[6]
            mtip = lm[12]

            curr_x = np.interp(itip.x * w, (margin, w - margin), (0, w))
            curr_y = np.interp(itip.y * h, (margin, h - margin), (0, h))

            clocX = self.plocX + (curr_x - self.plocX) / smoothness
            clocY = self.plocY + (curr_y - self.plocY) / smoothness

            is_drawing = itip.y < iknuckle.y and mtip.y > lm[10].y

            if is_drawing:
                if self.plocX != 0:
                    color = self.get_color()
                    cv2.line(
                        self.canvas,
                        (int(self.plocX), int(self.plocY)),
                        (int(clocX), int(clocY)),
                        color,
                        10
                    )
                self.plocX, self.plocY = clocX, clocY
            else:
                self.plocX, self.plocY = 0, 0
                cv2.circle(img, (int(clocX), int(clocY)), 8, (0, 255, 150), 2)

        # Glow effect
        glow = cv2.GaussianBlur(self.canvas, (9, 9), 0)
        img = cv2.addWeighted(img, 1.0, glow, 1.2, 0)
        final_img = cv2.add(img, self.canvas)

        return av.VideoFrame.from_ndarray(final_img, format="bgr24")

# ✅ UPDATED WebRTC Configuration (STUN + TURN)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
})

# ✅ WebRTC Streamer (optimized)
webrtc_streamer(
    key="neon-paint-v2",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PrecisionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 30},
        },
        "audio": False,
    },
    async_processing=True,
)
