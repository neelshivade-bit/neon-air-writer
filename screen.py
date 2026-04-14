import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="High-Precision Neon", layout="wide")

with st.sidebar:
    st.header("Calibration")
    # Adjust this if you can't reach the corners of your screen
    margin = st.slider("Edge Sensitivity", 0, 250, 100, help="Higher = reach corners easier")
    smoothness = st.slider("Smoothing", 1, 20, 5)
    if st.button("🗑️ Clear Canvas"):
        st.session_state.clear_canvas = True

class PrecisionProcessor:
    def __init__(self):
        self.canvas = None
        self.plocX, self.plocY = 0, 0
        self.hue = 0
        
        # INCREASED STRENGTH: We set model_complexity to 1 for better tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1, 
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        if self.canvas is None or st.session_state.get("clear_canvas", False):
            self.canvas = np.zeros_like(img)
            st.session_state.clear_canvas = False

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            # Get the tip of the index finger (8) and the knuckle (6)
            lm = results.multi_hand_landmarks[0].landmark
            itip = lm[8]
            iknuckle = lm[6]
            mtip = lm[12] # Middle finger tip

            # --- FULL SCREEN MAPPING LOGIC ---
            # This maps a smaller center box in your camera to the full screen
            # effectively boosting 'detection' at the edges.
            curr_x = np.interp(itip.x * w, (margin, w - margin), (0, w))
            curr_y = np.interp(itip.y * h, (margin, h - margin), (0, h))

            # Smooth the movement
            clocX = self.plocX + (curr_x - self.plocX) / smoothness
            clocY = self.plocY + (curr_y - self.plocY) / smoothness

            # GESTURE: Index is UP and Middle is DOWN
            # This is the most stable 'Pen Down' gesture
            is_drawing = itip.y < iknuckle.y and mtip.y > lm[10].y

            if is_drawing:
                if self.plocX != 0:
                    # Dynamic Color
                    color_hsv = np.uint8([[[int(self.hue) % 180, 255, 255]]])
                    draw_color = tuple(map(int, cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]))

                    # Draw thick neon and thin white core
                    cv2.line(self.canvas, (int(self.plocX), int(self.plocY)), (int(clocX), int(clocY)), draw_color, 15)
                    cv2.line(self.canvas, (int(self.plocX), int(self.plocY)), (int(clocX), int(clocY)), (255, 255, 255), 3)
                
                self.plocX, self.plocY = clocX, clocY
                self.hue = (self.hue + 2) % 180
            else:
                # Hovering: Show a ghost cursor for feedback
                cv2.circle(img, (int(clocX), int(clocY)), 10, (0, 255, 100), 2)
                self.plocX, self.plocY = 0, 0

        # Create Bloom Effect
        glow = cv2.GaussianBlur(self.canvas, (21, 21), 0)
        img = cv2.addWeighted(img, 0.7, glow, 1.8, 0)
        final_img = cv2.add(img, self.canvas)

        return av.VideoFrame.from_ndarray(final_img, format="bgr24")

webrtc_streamer(
    key="precision-v6",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PrecisionProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)