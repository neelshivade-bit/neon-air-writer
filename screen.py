import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Page Config
st.set_page_config(page_title="Ultra-Smooth Neon Painter", layout="wide")

# Persistent State Management
if "clear_canvas" not in st.session_state:
    st.session_state.clear_canvas = False

with st.sidebar:
    st.title("Settings")
    margin = st.slider("Edge Sensitivity", 0, 200, 80)
    smoothness = st.slider("Smoothing", 1, 15, 3) # Lower is faster/rawer
    if st.button("🗑️ Clear Canvas"):
        st.session_state.clear_canvas = True

class PrecisionProcessor:
    def __init__(self):
        self.canvas = None
        self.plocX, self.plocY = 0, 0
        self.hue = 0
        self.lock = threading.Lock() # Prevents flickering during canvas updates
        
        # Performance optimization: model_complexity=0 is MUCH faster for real-time
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0, 
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        with self.lock:
            if self.canvas is None or st.session_state.clear_canvas:
                self.canvas = np.zeros_like(img)
                st.session_state.clear_canvas = False

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            itip = lm[8]
            iknuckle = lm[6]
            mtip = lm[12]

            # Coordinate interpolation for full-screen reach
            curr_x = np.interp(itip.x * w, (margin, w - margin), (0, w))
            curr_y = np.interp(itip.y * h, (margin, h - margin), (0, h))

            # Exponential Smoothing
            clocX = self.plocX + (curr_x - self.plocX) / smoothness
            clocY = self.plocY + (curr_y - self.plocY) / smoothness

            # Stable Gesture: Index up, Middle finger curled
            is_drawing = itip.y < iknuckle.y and mtip.y > lm[10].y

            if is_drawing:
                if self.plocX != 0:
                    # Faster color conversion: 
                    # Instead of full cv2.cvtColor every frame, we use a simple rainbow logic
                    self.hue = (self.hue + 3) % 180
                    color = cv2.cvtColor(np.uint8([[[self.hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                    color = (int(color[0]), int(color[1]), int(color[2]))

                    cv2.line(self.canvas, (int(self.plocX), int(self.plocY)), 
                             (int(clocX), int(clocY)), color, 12)
                    cv2.line(self.canvas, (int(self.plocX), int(self.plocY)), 
                             (int(clocX), int(clocY)), (255, 255, 255), 2)
                
                self.plocX, self.plocY = clocX, clocY
            else:
                self.plocX, self.plocY = 0, 0
                cv2.circle(img, (int(clocX), int(clocY)), 8, (0, 255, 150), 2)

        # Optimization: Apply blur only to the canvas, not the whole image
        # This keeps the video background sharp and saves CPU
        glow = cv2.GaussianBlur(self.canvas, (15, 15), 0)
        img = cv2.addWeighted(img, 1.0, glow, 1.5, 0)
        final_img = cv2.add(img, self.canvas)

        return av.VideoFrame.from_ndarray(final_img, format="bgr24")

webrtc_streamer(
    key="neon-paint-v1",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PrecisionProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
