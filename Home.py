import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow as tf
import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
import tempfile
from collections import Counter
from streamlit_webrtc import VideoProcessorBase, RTCConfiguration,WebRtcMode,webrtc_streamer

import streamlit as st
import cv2 as cv
from tempfile import NamedTemporaryFile
import logging
import queue
from pathlib import Path
from typing import List, NamedTuple
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# from moviepy.editor import VideoFileClip
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)
# Load the LRCN model from a local file
model_file_name = "LRCN_model_Date_Time_2024_04%d_07_33_38_Loss0.7966662645339966_Accuracy_0.8348082304000854.h5"
LRCN_model = load_model(model_file_name)

IMAGE_HEIGHT,IMAGE_WIDTH = 64,64
SEQUENCE_LENGTH = 20

CLASSES_LIST = ['RockClimbingIndoor',
'TennisSwing',
'HorseRiding',
'SkateBoarding',
'Clapping',
'Basketball',
'Biking',
'CricketShot',
'GolfSwing',
'Kayaking']

def predict_on_video(video_reader):
    predicted_labels = []
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break


        if not frame.size:
            continue

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:

            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)

            predicted_class_name = CLASSES_LIST[predicted_label]

            predicted_labels.append(predicted_class_name)


    video_reader.release()
    label_counts = Counter(predicted_labels)
    
    # Get the most common label and its count
    most_common_label, count = label_counts.most_common(1)[0]
    
    return most_common_label


def main():
    st.markdown("<h1 style='text-align: center;'>Human Activity Classification</h1>",unsafe_allow_html=True)
    st.subheader("", divider='rainbow')

    st.empty()
    

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        f = st.file_uploader("Choose a Video you want to Classify")
        most_common_label='Unknown'
        if f is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(f.read())
            vf = cv2.VideoCapture(tfile.name)
            # vf1 = cv2.VideoCapture(tfile.name)
            # stframe = st.empty()
            # while vf1.isOpened():
            #     ret, frame = vf1.read()
            #     # if frame is read correctly ret is True
            #     if not ret:
            #         print("Can't receive frame (stream end?). Exiting ...")
            #         break
            #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #     stframe.image(frame_rgb)
            label = predict_on_video(vf)
            st.subheader(label)
            # stframe = st.empty()
            bytes_data = f.getvalue()
            st.video(bytes_data)
    
    with col2:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.markdown('''
   
            <h4 style='text-align: center;'>OR</h4>
        ''',unsafe_allow_html=True)

    with col3:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        class VideoProcessor:
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                #most_common_label = predict_on_video(img)
                label = f"Predicted: Clapping"
                cv2.rectangle(img, (0,0), (640, 40), (234, 234, 77), 1)
                cv2.putText(img,label, (3,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        st.write("Open Webcam")
        webrtc_ctx = webrtc_streamer(
            key="WYH",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=VideoProcessor,
            async_processing=True,
)
if __name__ == "__main__":
    main()


