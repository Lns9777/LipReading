import streamlit as st
import os
import imageio
import cv2
import numpy as np
import tensorflow as tf
from utils import load_data, num_to_char, load_video
from modelutil import load_model

# Set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

# Set up the sidebar
with st.sidebar:
    st.image('https://media.licdn.com/dms/image/C5112AQEaCBBDXuf…eta&t=N5-zmwDxTkqLm95IN0Apesu4VSfdM8o_UufBg52sZW4')
    st.title("LipBuddy")
    st.info("This application is originally developed from the LipNet Deep learning Models.")

st.title("LipNet Full Stack App")
video_source = st.radio("Select Video Source", ('Pre-Recorded Video', "Real-Time Webcam"))

if video_source == 'Pre-Recorded Video':
    # Generating a list of options for videos
    options = os.listdir(os.path.join('data', 's1'))
    selected_video = st.selectbox("Choose a video", options)

    # Generate two columns
    col1, col2 = st.columns(2)

    if options:
        # Rendering the video
        with col1:
            st.info("The video below displays the converted video in mp4 format")
            file_path = os.path.join('data', 's1', selected_video)
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

            # Rendering inside of the app
            video = open("test_video.mp4", "rb")
            video_bytes = video.read()
            st.video(video_bytes)

        with col2:
            st.info('This is all the machine learning model sees when making a prediction')
            video, annotations = load_data(tf.convert_to_tensor(file_path))
            # imageio.mimsave('animation.gif', video, fps=10)
            # st.image('animation.gif', width=400) 

            st.info('This is the output of the machine learning model as tokens')
            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)

            # Convert prediction to text
            st.info('Decode the raw tokens into words')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)

elif video_source == 'Real-Time Webcam':
    st.info('The video below displays the real-time webcam feed')
    
    # Placeholder for the webcam feed
    placeholder = st.empty()
    
    # Load the model
    model = load_model()
    
    # OpenCV video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame[190:236, 80:220]  # Crop to the mouth region
            frames.append(frame)

            # Display the frame in the app
            placeholder.image(frame, channels="GRAY", width=400)

            if len(frames) == 75:
                video = np.array(frames)
                video = video[..., np.newaxis]
                video = (video - np.mean(video)) / np.std(video)  # Normalize
                
                # Make prediction
                yhat = model.predict(tf.expand_dims(video, axis=0))
                decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
                converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
                
                st.info('Decoded text')
                st.text(converted_prediction)

                frames = []  # Reset frames for next batch

        cap.release()
        cv2.destroyAllWindows()
